import pytest
import torch

from benchmarks.context_successor import (
    METHODS,
    candidate,
    evaluate_extrapolation_stream,
    evaluate_successor,
    extrapolation_candidate,
    extrapolation_streams,
    sample_successor_offsets,
    select_candidates,
    successor_candidate,
    successor_diagnostics,
    successor_metadata,
    successor_sanity_gate,
    train_extrapolation,
    train_successor,
)
from benchmarks.length_generalization import encode_successors, sample_text_batch


@pytest.mark.parametrize("method", METHODS)
def test_successor_baselines_train_and_use_requested_evaluation_variant(method):
    model = train_successor(method, seed=100, steps=1, batch_size=2, settings=candidate(method, 0))
    assert not model.training
    assert model.variant == method
    assert all(block.variant == method for block in model.blocks)
    if method in ("rope-linear", "rope-ntk"):
        assert all(block.rope_scale == 4.0 for block in model.blocks)
    with torch.no_grad():
        assert torch.isfinite(model(torch.zeros(1, 16, dtype=torch.long))).all()


def test_successor_candidate_adds_extended_tuning_options_without_changing_originals():
    assert successor_candidate("alibi", 0) == candidate("alibi", 0)
    assert successor_candidate("rope-log-window", 1) == candidate("rope-log-window", 1)
    assert successor_candidate("alibi", 2) == {
        "learning_rate": 0.0015,
        "tau0": 10.0,
        "rope_scale": 1.0,
    }
    assert successor_candidate("rope-linear", 2)["rope_scale"] == 8.0
    assert successor_candidate("rope-log-full", 2)["tau0"] == 15.0
    assert successor_candidate("rope-log-window", 3)["tau0"] == 2.0
    with pytest.raises(ValueError, match="unknown method or candidate"):
        successor_candidate("alibi", 3)


def test_extended_successor_candidate_trains():
    for method, index in (("alibi", 2), ("rope-log-window", 3)):
        settings = successor_candidate(method, index)
        model = train_successor(method, seed=100, steps=1, batch_size=2, settings=settings)
        assert model.variant == method
        with torch.no_grad():
            assert torch.isfinite(model(torch.zeros(1, 16, dtype=torch.long))).all()


def test_successor_tuning_selects_from_every_complete_candidate():
    records = []
    for candidate_index, correct in ((0, 5), (1, 6), (2, 7)):
        for seed in range(100, 110):
            records.append(
                {
                    "method": "alibi",
                    "candidate": candidate_index,
                    "seed": seed,
                    "steps": 500,
                    "metrics": {
                        "4_digit_held_out": {
                            "4096": {
                                "1024_4095": {"correct": correct, "total": 10},
                            }
                        }
                    },
                }
            )
    assert select_candidates(records, ["alibi"], 500) == {"alibi": 2}
    assert select_candidates(records[:20], ["alibi"], 500) == {"alibi": 1}
    with pytest.raises(ValueError, match="missing original tuning candidates"):
        select_candidates(records[10:], ["alibi"], 500)
    with pytest.raises(ValueError, match="incomplete tuning seeds"):
        select_candidates(records[:-1], ["alibi"], 500)


def test_successor_tuning_can_select_low_temperature_candidate():
    records = [
        {
            "method": "rope-log-window",
            "candidate": index,
            "seed": seed,
            "steps": 500,
            "metrics": {
                "4_digit_held_out": {"4096": {"1024_4095": {"correct": correct, "total": 10}}}
            },
        }
        for index, correct in ((0, 5), (1, 6), (3, 7))
        for seed in range(100, 110)
    ]
    assert select_candidates(records, ["rope-log-window"], 500) == {"rope-log-window": 3}


def test_successor_metadata_distinguishes_carry_and_last_digits():
    data = encode_successors(1098, 1102)
    roles, last, carry = successor_metadata(data)
    assert roles[:15].tolist() == [1, 2, 3, 4, 0] * 3
    assert last.nonzero().flatten().tolist() == [3, 8, 13, 18]
    assert carry[:5].tolist() == [-1] * 5
    assert carry[5:9].tolist() == [0, 0, 0, 0]
    assert carry[10:14].tolist() == [0, 1, 1, 0]
    assert carry[14].item() == -1


def test_successor_diagnostics_counts_roles_and_same_window_rules():
    data = encode_successors(1000, 1010)
    starts = torch.tensor([0])
    targets = data[1:41].unsqueeze(0)
    predictions = targets.clone()
    predictions[0, 10] = 9
    result = successor_diagnostics(targets, predictions, starts, successor_metadata(data))["0_63"]
    assert result["digit_2"] == {"correct": 7, "total": 8}
    assert result["delimiter"] == {"correct": 8, "total": 8}
    assert result["copy_5"] == {"correct": 28, "total": 35}
    assert result["copy_5_increment_last_oracle"] == {"correct": 35, "total": 35}
    assert result["majority_per_window_oracle"] == {"correct": 14, "total": 35}
    assert result["carry"] == {"correct": 0, "total": 0}
    assert result["non_carry"]["total"] == 29


def test_successor_diagnostics_preserves_original_samples_and_metrics():
    class Model:
        variant = "rope-full"
        train_length = 256

        def __call__(self, tokens):
            return torch.nn.functional.one_hot(tokens, num_classes=11).float()

    plain = evaluate_successor(Model(), 42, [256], 2)
    diagnosed = evaluate_successor(Model(), 42, [256], 2, diagnostics=True)
    assert {key: value for key, value in diagnosed.items() if key != "diagnostics"} == plain
    for stream in ("3_digit_train_range_cyclic", "4_digit_held_out"):
        counts = diagnosed["diagnostics"][stream]["256"]
        assert counts["0_63"]["digit_1"]["total"] > 0
        assert counts["64_255"]["copy_5"]["total"] == 2 * 192
    data = encode_successors(1000, 3000)
    generator = torch.Generator().manual_seed(42 + 256 + 110000)
    inputs, targets = sample_text_batch(data, 256, 2, generator)
    expected = successor_diagnostics(
        targets,
        inputs,
        torch.randint(
            0, len(data) - 257, (2,), generator=torch.Generator().manual_seed(42 + 256 + 110000)
        ),
        successor_metadata(data),
    )
    assert counts == expected


def test_successor_diagnostics_reject_misaligned_sampling(monkeypatch):
    class Model:
        variant = "rope-full"
        train_length = 256

    def wrong_sample(data, context, batch_size, generator):
        inputs, targets = sample_text_batch(data, context, batch_size, generator)
        return inputs, targets.roll(1, dims=1)

    monkeypatch.setattr("benchmarks.context_successor.sample_text_batch", wrong_sample)
    with pytest.raises(RuntimeError, match="sample offsets"):
        evaluate_successor(Model(), 42, [256], 1, diagnostics=True)


@pytest.mark.parametrize(
    ("axis", "train_first", "train_last", "unseen_first", "unseen_last"),
    [
        ("context", "100", "699", "700", "999"),
        ("digit_4", "0", "999", "1000", "2999"),
        ("digit_5", "0", "9999", "10000", "29999"),
    ],
)
def test_extrapolation_streams_keep_number_and_context_axes_separate(
    axis, train_first, train_last, unseen_first, unseen_last
):
    streams = extrapolation_streams(axis)
    decoded = {}
    for name in ("train", "unseen"):
        decoded[name] = "".join(
            "|" if token == 10 else str(token) for token in streams[name].tolist()
        )
    assert decoded["train"].startswith(train_first + "|")
    assert decoded["train"].endswith(train_last + "|")
    assert decoded["unseen"].startswith(unseen_first + "|")
    assert decoded["unseen"].endswith(unseen_last + "|")
    assert torch.equal(streams["seen"], streams["train"])
    with pytest.raises(ValueError, match="unknown extrapolation axis"):
        extrapolation_streams("unsupported")


def test_successor_offsets_disjoint_and_repeat_only_when_required():
    data = encode_successors(100, 200)
    train_x, train_y, train_offsets = sample_successor_offsets(
        data, 8, 100, torch.Generator().manual_seed(7), "train"
    )
    sanity_x, sanity_y, sanity_offsets = sample_successor_offsets(
        data, 8, 100, torch.Generator().manual_seed(7), "sanity"
    )
    assert set(train_offsets.tolist()).isdisjoint(sanity_offsets.tolist())
    assert all((offset ^ (offset >> 3) ^ (offset >> 6)) % 5 != 0 for offset in train_offsets)
    assert all((offset ^ (offset >> 3) ^ (offset >> 6)) % 5 == 0 for offset in sanity_offsets)
    assert set(train_offsets.remainder(5).tolist()) == set(range(5))
    assert set(sanity_offsets.remainder(5).tolist()) == set(range(5))
    assert torch.equal(train_x[:, 1:], train_y[:, :-1])
    assert torch.equal(sanity_x[:, 1:], sanity_y[:, :-1])
    assert max(train_offsets).item() + 8 < len(data)
    short_data = encode_successors(100, 110)
    _, long_targets, offsets = sample_successor_offsets(
        short_data, 80, 2, torch.Generator().manual_seed(9), "unseen"
    )
    assert long_targets[0].tolist() == [
        int(short_data[(int(offsets[0]) + index + 1) % len(short_data)]) for index in range(80)
    ]
    with pytest.raises(ValueError, match="unknown successor partition"):
        sample_successor_offsets(data, 8, 1, torch.Generator(), "invalid")


def test_successor_sanity_requires_eight_of_ten_complete_independent_seeds():
    records = [{"seed": seed, "sanity_accuracy": 0.95} for seed in range(100, 110)]
    records[0]["sanity_accuracy"] = 0.94
    records[1]["sanity_accuracy"] = 0.0
    assert successor_sanity_gate(records, range(100, 110))
    records[2]["sanity_accuracy"] = 0.9499
    assert not successor_sanity_gate(records, range(100, 110))
    with pytest.raises(ValueError, match="exactly one"):
        successor_sanity_gate(records[:-1], range(100, 110))
    with pytest.raises(ValueError, match="complete tuning or reporting"):
        successor_sanity_gate(records, range(100, 109))
    records[0]["sanity_accuracy"] = float("nan")
    with pytest.raises(ValueError, match="finite probabilities"):
        successor_sanity_gate(records, range(100, 110))


@pytest.mark.parametrize("method", ["nope", "rope-linear", "rope-ntk", "rope-log-window"])
def test_extrapolation_methods_train_with_balanced_candidate_settings(method):
    settings = extrapolation_candidate(method, 0)
    model = train_extrapolation(method, "context", 100, 1, 2, settings)
    assert model.variant == method
    assert not model.training
    assert extrapolation_candidate(method, 1)["learning_rate"] == 0.001
    if method in ("rope-linear", "rope-ntk"):
        assert all(block.rope_scale == 4.0 for block in model.blocks)
    with torch.no_grad():
        assert torch.isfinite(model(torch.zeros(1, 16, dtype=torch.long))).all()
    with pytest.raises(ValueError, match="unknown extrapolation"):
        extrapolation_candidate(method, 3)


def test_extrapolation_evaluation_aggregates_paired_samples_and_diagnostics():
    class Model:
        def __call__(self, tokens):
            return torch.nn.functional.one_hot(tokens, num_classes=11).float()

    data = extrapolation_streams("context")["unseen"]
    result = evaluate_extrapolation_stream(Model(), data, "unseen", 100, 256, 3)
    assert 0 < result["accuracy"] < 1
    assert result["perplexity"] > 1
    assert sum(bucket["total"] for bucket in result["buckets"].values()) == 3 * 256
    assert result["diagnostics"]["64_255"]["copy_5"]["total"] == 3 * 192


def test_extrapolation_training_rejects_nonfinite_loss(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.context_successor.F.cross_entropy",
        lambda *args, **kwargs: torch.tensor(float("nan")),
    )
    with pytest.raises(RuntimeError, match="non-finite extrapolation loss"):
        train_extrapolation("nope", "context", 100, 1, 1, extrapolation_candidate("nope", 0))
