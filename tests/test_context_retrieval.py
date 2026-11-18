import collections
import hashlib
import json
import subprocess
import sys
from argparse import Namespace

import pytest
import torch

from benchmarks.context_retrieval import (
    COMPARISON_METHODS,
    KEY_START,
    PAIR_MARKER,
    POSITION_BUCKETS,
    QUERY_MARKER,
    VALUE_COUNT,
    VALUE_START,
    VOCABULARY_SIZE,
    _eligible_candidates,
    _experiment_identity,
    _experiment_lock,
    _json_digest,
    _manifest,
    _open_experiment,
    _paired_summary,
    _record_configuration,
    _save_model,
    _validate_result_file,
    ablate_stored_pairs,
    evaluate_model,
    evaluate_predictions,
    evaluate_random_control,
    example_registry_records,
    fingerprint_examples,
    generate_dataset,
    generate_example,
    metric_counts,
    oracle_target,
    permute_distractors,
    random_answer,
    stored_pairs,
    train_model,
    validate_example,
    write_example_registry,
)


@pytest.mark.parametrize("context_length", [256, 512, 1024, 2048, 4096])
def test_generator_creates_exact_context_length_and_oracle_label(context_length):
    example = generate_example(context_length, 16, 42, target_bucket="middle")
    assert len(example.tokens) == context_length
    assert oracle_target(example.tokens) == example.target
    assert example.target_position_bucket == "middle"
    assert example.pair_order == tuple(key for _, key, _ in stored_pairs(example.tokens))


def test_generation_is_deterministic_and_split_sensitive():
    first = generate_example(512, 16, 19, split="tune", sample_index=7)
    repeat = generate_example(512, 16, 19, split="tune", sample_index=7)
    other_split = generate_example(512, 16, 19, split="report", sample_index=7)
    assert first == repeat
    assert first.example_id == repeat.example_id
    assert first.example_id != other_split.example_id


def test_pairs_have_unique_keys_and_values_without_answer_leakage():
    example = generate_example(1024, 32, 3, target_bucket="late")
    pairs = stored_pairs(example.tokens)
    keys = [key for _, key, _ in pairs]
    values = [value for _, _, value in pairs]
    assert len(set(keys)) == len(set(values)) == 32
    assert keys.count(example.target_key) == 1
    assert example.tokens.count(example.target_key) == 2
    assert example.tokens.count(example.target) == 1
    assert all(KEY_START <= key < KEY_START + 64 for key in keys)
    assert all(VALUE_START <= value < VALUE_START + VALUE_COUNT for value in values)
    assert example.tokens[-2:] == (QUERY_MARKER, example.target_key)
    validate_example(example)


def test_counterfactual_query_selects_a_different_stored_value():
    example = generate_example(512, 16, 8)
    pairs = stored_pairs(example.tokens)
    alternate_key, alternate_value = next(
        (key, value)
        for _, key, value in pairs
        if key != example.target_key and value != example.target
    )
    counterfactual = list(example.tokens)
    counterfactual[-1] = alternate_key
    assert oracle_target(counterfactual) == alternate_value
    assert alternate_value != example.target


def test_distractor_permutation_preserves_query_answer_and_target_position():
    example = generate_example(1024, 32, 17, target_bucket="middle")
    permuted = permute_distractors(example, seed=99)
    assert oracle_target(permuted.tokens) == example.target
    assert permuted.target_position == example.target_position
    assert permuted.target_key == example.target_key
    assert permuted.pair_order != example.pair_order
    assert permuted.example_id != example.example_id


def test_pair_ablation_removes_stored_answers_and_preserves_query():
    example = generate_example(256, 16, 31)
    ablated = ablate_stored_pairs(example, seed=2)
    assert len(ablated) == len(example.tokens)
    assert PAIR_MARKER not in ablated
    assert not any(VALUE_START <= token < VALUE_START + VALUE_COUNT for token in ablated)
    assert ablated[-2:] == example.tokens[-2:]


def test_evaluation_positions_are_exactly_balanced_and_metadata_is_recorded(tmp_path):
    examples = generate_dataset("report", 44, 30, 4096, 16)
    counts = collections.Counter(example.target_position_bucket for example in examples)
    assert counts == {bucket: 10 for bucket in POSITION_BUCKETS}
    path = tmp_path / "examples.jsonl"
    write_example_registry(path, example_registry_records("report", examples))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 30
    assert all(
        {"pair_order", "target_position", "query_position", "target_distance"} <= record.keys()
        for record in records
    )


def test_train_tune_and_report_examples_are_disjoint():
    pools = {
        split: generate_dataset(split, 50 + index, 30, 256, 16)
        for index, split in enumerate(("train", "tune", "report"))
    }
    identifiers = {
        split: {example.example_id for example in examples} for split, examples in pools.items()
    }
    assert identifiers["train"].isdisjoint(identifiers["tune"])
    assert identifiers["train"].isdisjoint(identifiers["report"])
    assert identifiers["tune"].isdisjoint(identifiers["report"])
    assert len({fingerprint_examples(examples) for examples in pools.values()}) == 3


def test_registry_rejects_example_reuse_across_splits(tmp_path):
    example = generate_example(256, 16, 5, split="tune")
    path = tmp_path / "examples.jsonl"
    write_example_registry(path, example_registry_records("tune", [example]))
    with pytest.raises(ValueError, match="leakage across"):
        write_example_registry(path, example_registry_records("report", [example]))


class OracleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward_last(self, tokens):
        logits = torch.zeros(tokens.shape[0], VOCABULARY_SIZE, device=tokens.device) + self.anchor
        for row, tokens_row in enumerate(tokens.tolist()):
            logits[row, oracle_target(tokens_row)] = 1.0
        return logits


class CaptureModel(OracleModel):
    def forward_last(self, tokens):
        self.seen = tokens.clone()
        return torch.zeros(tokens.shape[0], VOCABULARY_SIZE, device=tokens.device)


def test_model_evaluator_checks_exact_retrieval_and_group_denominators():
    examples = generate_dataset("report", 11, 12, 256, 16)
    metrics = evaluate_model(OracleModel(), examples, batch_size=4)
    assert metrics["overall"] == {"correct": 12, "total": 12, "accuracy": 1.0}
    assert sum(group["total"] for group in metrics["by_target_position"].values()) == 12
    assert sum(group["total"] for group in metrics["by_target_distance"].values()) == 12
    assert sum(group["total"] for group in metrics["by_distractor_count"].values()) == 12


def test_no_retrieval_control_keeps_query_but_ablates_all_pairs():
    examples = generate_dataset("report", 12, 6, 256, 16)
    model = CaptureModel()
    metrics = evaluate_model(model, examples, batch_size=2, ablate_pairs=True)
    assert metrics["overall"]["total"] == 6
    assert PAIR_MARKER not in model.seen
    assert torch.all(model.seen[:, -2] == QUERY_MARKER)
    assert all(example.target >= VALUE_START for example in examples)


def test_random_answer_baseline_is_repeatable_and_has_valid_denominators():
    examples = generate_dataset("report", 77, 12, 256, 16)
    first = evaluate_random_control(examples, seed=9)
    second = evaluate_random_control(examples, seed=9)
    assert first == second
    assert first["overall"]["total"] == 12
    answers = [random_answer(example, seed=9) for example in examples]
    assert all(VALUE_START <= answer < VALUE_START + VALUE_COUNT for answer in answers)
    assert first["overall"]["correct"] == sum(
        answer == example.target for answer, example in zip(answers, examples)
    )


def test_metric_rejects_prediction_count_mismatch_and_invalid_token():
    examples = generate_dataset("report", 4, 3, 256, 16)
    with pytest.raises(ValueError, match="exactly one prediction"):
        metric_counts(examples, [examples[0].target])
    with pytest.raises(ValueError, match="vocabulary token IDs"):
        evaluate_predictions(examples, [VOCABULARY_SIZE, 0, 0])


def test_training_length_gate_keeps_0_95_rule_without_gating_window_diagnostic():
    records = []
    for method in ("rope-log-full", "rope-full", "alibi", "rope-log-window"):
        for candidate in (0, 1):
            for seed in range(200, 210):
                accuracy = (
                    (0.95 if seed < (208 if candidate == 0 else 207) else 0.94)
                    if method != "rope-log-window"
                    else 0.1
                )
                records.append(
                    {
                        "method": method,
                        "candidate": candidate,
                        "seed": seed,
                        "evaluation_metrics": {
                            "256:16": {"model": {"overall": {"accuracy": accuracy}}}
                        },
                    }
                )
    eligible = _eligible_candidates(records, COMPARISON_METHODS)
    assert eligible == {
        "rope-log-full": [0],
        "rope-full": [0],
        "alibi": [0],
    }


def test_existing_results_reject_changed_source_fingerprint(tmp_path, monkeypatch):
    import benchmarks.context_retrieval as benchmark

    configuration = {"steps": 2, "batch_size": 1}
    manifest_config = {
        **configuration,
        "method": "rope-log-full",
        "candidate": 0,
        "candidate_settings": {"learning_rate": 0.001, "tau0": 5.0, "rope_scale": 1.0},
        "phase": "sanity",
        "configuration_sha256": _json_digest(configuration),
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
    }
    manifest, manifest_hash = _manifest(
        manifest_config,
        seed=200,
        training_split_fingerprint="train-fingerprint",
        evaluation_split_fingerprint="sanity-fingerprint",
    )
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    result_path = tmp_path / "results.jsonl"
    record = {
        "configuration": configuration,
        "phase": "sanity",
        "method": "rope-log-full",
        "candidate": 0,
        "candidate_settings": manifest_config["candidate_settings"],
        "seed": 200,
        "training_split_fingerprint": "train-fingerprint",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "manifest": manifest,
        "manifest_sha256": manifest_hash,
    }
    record["record_sha256"] = _json_digest(record)
    result_path.write_text(json.dumps(record) + "\n")
    assert _validate_result_file(result_path, configuration) == [record]

    changed = {**record, "candidate": 1}
    result_path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="record hash mismatch"):
        _validate_result_file(result_path, configuration)
    result_path.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(
        benchmark, "_source_digests", lambda: ("changed-toolkit", "changed-benchmark")
    )
    with pytest.raises(ValueError, match="source fingerprints differ"):
        _validate_result_file(result_path, configuration)


def test_experiment_ownership_rejects_existing_output_and_changed_configuration(tmp_path):
    results = tmp_path / "run.jsonl"
    registry = tmp_path / "run.examples.jsonl"
    checkpoints = tmp_path / "run-checkpoints"
    configuration = {"steps": 3000, "batch_size": 16}
    results.write_text("historical\n")
    with pytest.raises(FileExistsError, match="unused output"):
        _open_experiment(results, registry, checkpoints, configuration, False)
    assert results.read_text() == "historical\n"
    results.unlink()
    status_path, status = _open_experiment(results, registry, checkpoints, configuration, False)
    assert status["identity"] == _experiment_identity(configuration, results, checkpoints)
    assert _open_experiment(results, registry, checkpoints, configuration, True)[0] == status_path
    with pytest.raises(ValueError, match="configuration"):
        _open_experiment(results, registry, checkpoints, {**configuration, "steps": 3001}, True)
    with pytest.raises(FileExistsError, match="unused output"):
        _open_experiment(results, registry, checkpoints, configuration, False)


def test_experiment_lock_rejects_concurrent_process(tmp_path):
    result = tmp_path / "run.jsonl"
    script = (
        "from pathlib import Path; "
        "from benchmarks.context_retrieval import _experiment_lock; "
        f"p=Path({str(result)!r}); "
        "lock=_experiment_lock(p); lock.__enter__()"
    )
    with _experiment_lock(result):
        process = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert process.returncode != 0
    assert "experiment is already locked" in process.stderr


def test_phase_status_is_incomplete_on_interrupt_and_completed_on_resume(tmp_path, monkeypatch):
    import benchmarks.context_retrieval as benchmark

    output = tmp_path / "run.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["context_retrieval", "--phase", "sanity", "--output", str(output)]
    )
    monkeypatch.setattr(
        benchmark, "_run_training_phase", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        benchmark.main()
    status_path = tmp_path / "run.experiment.json"
    assert json.loads(status_path.read_text())["phases"]["sanity"] == "incomplete"
    monkeypatch.setattr(benchmark, "_run_training_phase", lambda *args: None)
    monkeypatch.setattr(
        sys, "argv", ["context_retrieval", "--phase", "sanity", "--output", str(output), "--resume"]
    )
    benchmark.main()
    assert json.loads(status_path.read_text())["phases"]["sanity"] == "completed"


@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
def test_interrupted_trial_resumes_checkpoint_without_training_or_duplicate_record(
    tmp_path, monkeypatch
):
    import benchmarks.context_retrieval as benchmark
    from benchmarks.length_generalization import SequenceLM

    examples = generate_dataset("train", 1, 3, 256, 16)
    inputs = torch.tensor([example.tokens for example in examples], dtype=torch.long)
    targets = torch.tensor([example.target for example in examples], dtype=torch.long)
    monkeypatch.setattr(benchmark, "_build_training_pool", lambda: (inputs, targets, examples))
    monkeypatch.setattr(benchmark, "SANITY_SEEDS", (200,))
    monkeypatch.setattr(benchmark, "CANDIDATES", (0,))
    monkeypatch.setattr(
        benchmark,
        "train_model",
        lambda *args: SequenceLM(VOCABULARY_SIZE, "rope-full", False, 256, 10.0, 10000.0),
    )
    args = Namespace(
        phase="sanity",
        methods=["rope-full"],
        steps=3,
        batch_size=1,
        eval_examples=3,
        eval_batch_size=1,
        device="cpu",
    )
    configuration = {"steps": 3, "batch_size": 1}
    records = tmp_path / "run.jsonl"
    registry = tmp_path / "run.examples.jsonl"
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setattr(
        benchmark, "_record_metrics", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        benchmark._run_training_phase(args, configuration, records, registry, checkpoints)
    assert not records.exists()
    assert (checkpoints / "sanity-rope-full-0-200.pt").exists()
    sidecar = checkpoints / "sanity-rope-full-0-200.manifest.json"
    original_sidecar = sidecar.read_text()
    sidecar.unlink()
    with pytest.raises(FileExistsError, match="unverifiable orphan"):
        benchmark._run_training_phase(args, configuration, records, registry, checkpoints)
    sidecar.write_text("{}")
    with pytest.raises(ValueError, match="checkpoint manifest or provenance mismatch"):
        benchmark._run_training_phase(args, configuration, records, registry, checkpoints)
    sidecar.write_text(original_sidecar)
    monkeypatch.setattr(
        benchmark, "train_model", lambda *args: pytest.fail("completed fit must not be retrained")
    )
    monkeypatch.setattr(benchmark, "_record_metrics", lambda *args: {"model": {"overall": {}}})
    benchmark._run_training_phase(args, configuration, records, registry, checkpoints)
    benchmark._run_training_phase(args, configuration, records, registry, checkpoints)
    rows = benchmark._validate_result_file(records, configuration)
    assert len(rows) == 1
    assert (
        rows[0]["manifest"]["experiment_config"]["checkpoint_sha256"]
        == rows[0]["checkpoint_sha256"]
    )
    checkpoint = checkpoints / "sanity-rope-full-0-200.pt"
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        benchmark._validate_result_file(records, configuration)


def test_summary_reports_paired_per_seed_results_and_confidence_interval_shape():
    records = []
    for method, offset in (
        ("rope-log-full", -0.1),
        ("rope-full", 0.0),
        ("alibi", 0.05),
        ("rope-log-window", -0.05),
    ):
        for seed in range(400, 410):
            records.append(
                {
                    "method": method,
                    "candidate": 0,
                    "seed": seed,
                    "evaluation_metrics": {
                        "4096:16": {
                            "model": {
                                "overall": {"accuracy": 0.5 + offset},
                                "by_target_position": {
                                    "early": {"accuracy": 0.4 + offset},
                                    "middle": {"accuracy": 0.5 + offset},
                                    "late": {"accuracy": 0.6 + offset},
                                },
                                "by_target_distance": {"0_63": {"accuracy": 0.5 + offset}},
                                "by_distractor_count": {"15": {"accuracy": 0.5 + offset}},
                            }
                        }
                    },
                }
            )
    summary = _paired_summary(records)
    comparison = summary["comparisons"]["rope-log-full_vs_rope-full"]["4096:16:overall"]
    assert comparison["seed_count"] == 10
    assert comparison["target_wins"] == 0
    assert len(comparison["per_seed_differences"]) == 10
    assert len(comparison["paired_95pct_seed_bootstrap_ci"]) == 2
    assert "4096:16:by_target_position:late" in summary["comparisons"]["rope-log-full_vs_rope-full"]


@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
def test_model_checkpoint_is_portable_cpu_state(tmp_path):
    model = CaptureModel()
    path = tmp_path / "model.pt"
    digest = _save_model(model, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert state["anchor"].device.type == "cpu"
    assert torch.equal(state["anchor"], model.anchor)


def test_record_configuration_tracks_device_and_numerical_settings():
    args = Namespace(
        steps=3000,
        batch_size=16,
        eval_batch_size=8,
        eval_examples=300,
        methods=list(COMPARISON_METHODS),
        device="cuda",
    )
    configuration = _record_configuration(args)
    assert configuration["device"] == "cuda"
    assert configuration["steps"] == 3000
    assert configuration["deterministic_algorithms"] == torch.are_deterministic_algorithms_enabled()
    assert configuration["cuda_matmul_allow_tf32"] == torch.backends.cuda.matmul.allow_tf32


def test_cuda_request_fails_closed_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.context_retrieval",
            "--phase",
            "sanity",
            "--device",
            "cuda",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "CUDA is unavailable" in result.stderr


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training_evaluation_and_checkpoint_round_trip(tmp_path):
    from benchmarks.length_generalization import SequenceLM

    examples = generate_dataset("hardware-test", 2, 3, 256, 16)
    inputs = torch.tensor([example.tokens for example in examples], dtype=torch.long)
    targets = torch.tensor([example.target for example in examples], dtype=torch.long)
    model = train_model("rope-log-full", 0, 200, 1, 2, inputs, targets, torch.device("cuda"))
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    assert evaluate_model(model, examples, batch_size=3)["overall"]["total"] == 3
    path = tmp_path / "model.pt"
    _save_model(model, path)
    restored = SequenceLM(VOCABULARY_SIZE, "rope-log-full", False, 256, 5.0, 10000.0)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    for name, value in model.state_dict().items():
        assert torch.equal(value.cpu(), restored.state_dict()[name])
