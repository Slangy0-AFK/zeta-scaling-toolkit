import pytest
import torch

from benchmarks.context_ability import evaluate_needle_sanity
from benchmarks.context_char import context_benefit, evaluate_char_contexts
from benchmarks.context_needle_tuned import sanity_passed, select_candidates, train_needle
from benchmarks.context_successor import candidate, score_successor_batch


def test_needle_sanity_gate_requires_ten_seeds_and_eight_learned_models():
    records = [
        {
            "seed": seed,
            "variant": "rope-full",
            "train_length": 256,
            "steps": 3000,
            "correct": 180 if seed < 50 else 179,
            "total": 200,
        }
        for seed in range(42, 52)
    ]
    assert sanity_passed(records)
    records[0]["correct"] = 179
    assert not sanity_passed(records)
    with pytest.raises(ValueError, match="exactly seeds"):
        sanity_passed(records[:-1])
    records[0]["total"] = 199
    with pytest.raises(ValueError, match="specified gate"):
        sanity_passed(records)


def test_needle_tuning_selects_accuracy_from_complete_separate_seeds():
    records = [
        {
            "method": "rope-full",
            "candidate": index,
            "steps": 3000,
            "seed": seed,
            "metrics": {"1024": {"accuracy": index / 2}},
        }
        for seed in range(100, 110)
        for index in (0, 1)
    ]
    assert select_candidates(records, ["rope-full"], 3000) == {"rope-full": 1}
    with pytest.raises(ValueError, match="incomplete retrieval tuning"):
        select_candidates(records[:-1], ["rope-full"], 3000)


@pytest.mark.parametrize(
    "method",
    [
        "abs",
        "rope-full",
        "rope-linear",
        "rope-ntk",
        "alibi",
        "rope-log-full",
        "rope-window",
        "rope-log-window",
    ],
)
def test_needle_baselines_train_and_use_requested_evaluation_variant(method):
    model = train_needle(method, seed=42, steps=1, batch_size=2, settings=candidate(method, 0))
    assert not model.training
    assert model.variant == method
    assert all(block.variant == method for block in model.blocks)
    if method in ("rope-linear", "rope-ntk"):
        assert all(block.rope_scale == 4.0 for block in model.blocks)
    with torch.no_grad():
        assert torch.isfinite(model(torch.zeros(1, 16, dtype=torch.long))).all()


def test_context_evaluators_reject_nonfinite_logits():
    class Nonfinite(torch.nn.Module):
        positions = None

        def forward(self, inputs):
            return torch.full((*inputs.shape, 97), float("nan"))

    model = Nonfinite()
    with pytest.raises(RuntimeError, match="non-finite needle"):
        evaluate_needle_sanity(model, 42, 256, 1, 1)
    with pytest.raises(RuntimeError, match="non-finite character context"):
        context_benefit(model, torch.zeros(4000, dtype=torch.long), 42)
    with pytest.raises(RuntimeError, match="non-finite character evaluation"):
        evaluate_char_contexts(model, torch.zeros(4000, dtype=torch.long), 42, [256], 1)
    with pytest.raises(RuntimeError, match="non-finite successor"):
        score_successor_batch(
            torch.full((1, 2, 11), float("nan")), torch.zeros(1, 2, dtype=torch.long)
        )
