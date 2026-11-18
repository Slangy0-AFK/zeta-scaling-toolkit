import hashlib
import json
import subprocess
import sys

import pytest
import torch

from benchmarks import context_extrapolation as runner


def sanity_records():
    return [
        {
            "method": method,
            "candidate": candidate,
            "seed": seed,
            "sanity_accuracy": 0.96 if method == "alibi" and candidate == 1 else 0.5,
        }
        for method in runner.EXTRAPOLATION_METHODS
        for candidate in runner.CANDIDATES
        for seed in runner.TUNING_SEEDS
    ]


def test_eligibility_requires_complete_equal_budget_and_blocks_undertrained_candidates():
    assert tuple(runner.TUNING_SEEDS) == tuple(range(100, 110))
    assert tuple(runner.REPORTING_SEEDS) == tuple(range(42, 52))
    records = sanity_records()
    eligible = runner.eligible_candidates(records)
    assert eligible["alibi"] == [1]
    assert all(not eligible[method] for method in runner.EXTRAPOLATION_METHODS if method != "alibi")
    with pytest.raises(ValueError, match="both candidates"):
        runner.eligible_candidates(records[:-1])
    with pytest.raises(ValueError, match="both candidates"):
        runner.eligible_candidates(records + records[:1])


@pytest.mark.parametrize("axis", ["context", "digit_4"])
def test_frozen_selection_rejects_missing_or_ineligible_trials(axis):
    eligible = {method: [] for method in runner.EXTRAPOLATION_METHODS}
    eligible["alibi"] = [0, 1]
    records = [
        {
            "method": "alibi",
            "candidate": candidate,
            "seed": seed,
            "unseen": {
                "256": {"accuracy": 0.5 + candidate * 0.1},
                "4096": {"buckets": {"1024_4095": {"correct": 5 + candidate, "total": 10}}},
            },
        }
        for candidate in runner.CANDIDATES
        for seed in runner.TUNING_SEEDS
    ]
    assert runner.select_frozen_candidates(records, eligible, axis) == {"alibi": 1}
    with pytest.raises(ValueError, match="incomplete"):
        runner.select_frozen_candidates(records[:-1], eligible, axis)
    with pytest.raises(ValueError, match="incomplete"):
        runner.select_frozen_candidates(records + records[:1], eligible, axis)


def test_record_loading_rejects_duplicates_and_configuration_mismatch(tmp_path):
    path = tmp_path / "records.jsonl"
    config = {"axis": "context"}
    assert runner.load_records(path, config) == []
    record = {"axis": "context", "method": "alibi", "candidate": 0, "seed": 100}
    path.write_text(json.dumps(record) + "\n")
    assert runner.load_records(path, config) == [record]
    with pytest.raises(ValueError, match="incompatible"):
        runner.load_records(path, {"axis": "digit_4"})
    path.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        runner.load_records(path, config)


def test_runner_sanity_resumes_and_tuning_requires_complete_gate(tmp_path, monkeypatch):
    class Model:
        def state_dict(self):
            return {}

    calls = []
    monkeypatch.setattr(runner, "train_extrapolation", lambda *args: calls.append(args) or Model())
    monkeypatch.setattr(
        runner,
        "evaluate_extrapolation_stream",
        lambda *args, **kwargs: {"accuracy": 0.9, "perplexity": 1.5},
    )
    args = [
        "context_extrapolation",
        "--phase",
        "sanity",
        "--axis",
        "context",
        "--work-dir",
        str(tmp_path),
        "--max-runs",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    runner.main()
    runner.main()
    records = (tmp_path / "context-500-sanity.jsonl").read_text().splitlines()
    assert len(records) == 2
    assert [json.loads(line)["seed"] for line in records] == [100, 101]
    assert len(calls) == 2
    monkeypatch.setattr(sys, "argv", args[:2] + ["tune"] + args[3:])
    with pytest.raises(ValueError, match="both candidates"):
        runner.main()


def test_runner_rejects_2000_without_complete_500_sanity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_extrapolation",
            "--phase",
            "sanity",
            "--axis",
            "digit_4",
            "--steps",
            "2000",
            "--work-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(ValueError, match="both candidates"):
        runner.main()


def test_reporting_uses_tuning_baseline_and_never_ranks_undertrained_methods():
    selected = {"alibi": 1, "rope-full": 1, "rope-log-window": 0}
    tuning = [
        {
            "method": method,
            "candidate": index,
            "seed": seed,
            "unseen": {"256": {"accuracy": score}},
        }
        for method, index, score in (
            ("alibi", 1, 0.8),
            ("rope-full", 1, 0.7),
            ("rope-log-window", 0, 0.9),
        )
        for seed in runner.TUNING_SEEDS
    ]

    def metric(accuracy):
        return {
            "accuracy": accuracy,
            "buckets": {"0_63": {"correct": int(accuracy * 100), "total": 100}},
            "diagnostics": {
                "0_63": {
                    "digit_1": {"correct": int(accuracy * 100), "total": 100},
                    "carry": {"correct": 0, "total": 0},
                }
            },
        }

    records = [
        {
            "method": method,
            "candidate": index,
            "seed": seed,
            "sanity_accuracy": 0.96 if method != "rope-full" else 0.8,
            "in_distribution": {"perplexity": 1.0},
            "seen": {"256": metric(score)},
            "unseen": {"256": metric(score)},
        }
        for method, index, score in (
            ("alibi", 1, 0.8),
            ("rope-full", 1, 0.7),
            ("rope-log-window", 0, 0.9),
        )
        for seed in runner.REPORTING_SEEDS
    ]
    summary = runner.reporting_summary(records, tuning, selected, "digit_4")
    assert summary["baseline"] == "alibi"
    assert summary["methods"]["rope-full"]["status"] == "undertrained"
    assert "streams" not in summary["methods"]["rope-full"]
    variant = summary["methods"]["rope-log-window"]
    assert variant["streams"]["unseen"]["256"]["overall"]["ci_low"] == pytest.approx(0.1)
    assert variant["diagnostics"]["unseen"]["256"]["0_63"]["carry"] is None
    assert runner.decision_from_summaries({500: summary, 2000: summary}, "digit_4")[
        "rope-log-window"
    ]
    assert not runner.decision_from_summaries({500: summary}, "digit_4")["rope-log-window"]
    variant["streams"]["unseen"]["256"]["overall"]["ci_low"] = -0.01
    assert not runner.decision_from_summaries({500: summary, 2000: summary}, "digit_4")[
        "rope-log-window"
    ]
    with pytest.raises(ValueError, match="frozen candidates"):
        runner.reporting_summary(records[:-1], tuning, selected, "digit_4")
    with pytest.raises(ValueError, match="ten finite"):
        runner.paired_results([1.0], [1.0])
    variant["streams"]["unseen"]["4096"] = {
        "1024_4095": {"ci_low": 0.1},
    }
    assert runner.decision_from_summaries({500: summary, 2000: summary}, "context")[
        "rope-log-window"
    ]
    with pytest.raises(ValueError, match="finite probability"):
        runner.selection_score({"unseen": {"256": {"accuracy": float("nan")}}}, "digit_4")


def test_evaluate_streams_respects_context_and_digit_lengths(monkeypatch):
    calls = []

    def evaluate(model, data, partition, seed, length, samples, *, least_significant_digit_first):
        calls.append((data, partition, seed, length, samples, least_significant_digit_first))
        return {"length": length}

    monkeypatch.setattr(runner, "evaluate_extrapolation_stream", evaluate)
    streams = {"seen": "seen_data", "unseen": "unseen_data"}
    result = runner.evaluate_streams(None, streams, "context", 100, 8)
    assert set(result["seen"]) == {"256", "512", "1024", "4096"}
    assert calls[0] == ("seen_data", "sanity", 100, 256, 8, False)
    assert calls[-1] == ("unseen_data", "unseen", 100, 4096, 8, False)
    assert set(runner.evaluate_streams(None, streams, "digit_5", 100, 8)["unseen"]) == {"256"}
    assert calls[-1][-1]


@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
def test_load_model_restores_requested_evaluation_variant(tmp_path):
    settings = runner.extrapolation_candidate("rope-linear", 0)
    model = runner.train_extrapolation("rope-linear", "context", 100, 1, 1, settings)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    loaded = runner.load_model(path, "rope-linear", 0)
    tokens = torch.arange(16).unsqueeze(0).remainder(11)
    with torch.no_grad():
        torch.testing.assert_close(loaded(tokens), model(tokens))


def test_runner_tuning_checks_checkpoints_and_reporting_uses_frozen_seed_set(
    tmp_path, monkeypatch, capsys
):
    config = {
        "axis": "digit_4",
        "steps": 500,
        "batch_size": 16,
        "samples": 8,
        "protocol": 1,
        "source_sha256": runner.source_digest(),
    }
    records = sanity_records()
    checkpoint = tmp_path / "digit_4-500-alibi-1-100.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for record in records:
        record.update(config)
        record["in_distribution"] = {"accuracy": record["sanity_accuracy"], "perplexity": 1.0}
        record["checkpoint_sha256"] = digest
    (tmp_path / "digit_4-500-sanity.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    args = [
        "context_extrapolation",
        "--phase",
        "tune",
        "--axis",
        "digit_4",
        "--work-dir",
        str(tmp_path),
        "--max-runs",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner, "load_model", lambda *args: None)
    monkeypatch.setattr(
        runner, "evaluate_streams", lambda *args: {"unseen": {"256": {"accuracy": 0.7}}}
    )
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checkpoint does not match"):
        runner.main()
    checkpoint.write_bytes(b"checkpoint")
    runner.main()
    tuning_path = tmp_path / "digit_4-500-tuning.jsonl"
    first = json.loads(tuning_path.read_text())
    assert (first["method"], first["candidate"], first["seed"]) == ("alibi", 1, 100)
    tuning_path.write_text(
        "".join(json.dumps({**first, "seed": seed}) + "\n" for seed in runner.TUNING_SEEDS)
    )
    calls = []
    monkeypatch.setattr(runner, "train_extrapolation", lambda *args: calls.append(args) or None)
    monkeypatch.setattr(
        runner,
        "evaluate_extrapolation_stream",
        lambda *args, **kwargs: {"accuracy": 0.96, "perplexity": 1.0},
    )
    monkeypatch.setattr(sys, "argv", args[:2] + ["report"] + args[3:])
    runner.main()
    reporting_path = tmp_path / "digit_4-500-reporting.jsonl"
    first = json.loads(reporting_path.read_text())
    assert (first["method"], first["candidate"], first["seed"]) == ("alibi", 1, 42)
    assert calls[0][2] == 42
    reporting_path.write_text(
        "".join(json.dumps({**first, "seed": seed}) + "\n" for seed in runner.REPORTING_SEEDS)
    )
    monkeypatch.setattr(runner, "reporting_summary", lambda *args: {"baseline": "alibi"})
    monkeypatch.setattr(sys, "argv", args[:2] + ["summary"] + args[3:])
    runner.main()
    assert '"baseline": "alibi"' in capsys.readouterr().out


def test_runner_script_help_is_executable():
    result = subprocess.run(
        [sys.executable, "benchmarks/context_extrapolation.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--phase" in result.stdout


def test_2000_step_phase_requires_prior_gates_and_uses_frozen_candidates(tmp_path, monkeypatch):
    class Model:
        def state_dict(self):
            return {}

    args = [
        "context_extrapolation",
        "--phase",
        "sanity",
        "--axis",
        "context",
        "--steps",
        "2000",
        "--work-dir",
        str(tmp_path),
        "--max-runs",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    eligible = {method: [] for method in runner.EXTRAPOLATION_METHODS}
    monkeypatch.setattr(runner, "eligible_candidates", lambda *args: eligible)
    with pytest.raises(ValueError, match="passing toolkit and baseline"):
        runner.main()
    eligible["alibi"] = [1]
    eligible["rope-log-window"] = [0]
    monkeypatch.setattr(
        runner,
        "select_frozen_candidates",
        lambda *args: {"alibi": 1, "rope-log-window": 0},
    )
    calls = []
    monkeypatch.setattr(runner, "train_extrapolation", lambda *args: calls.append(args) or Model())
    monkeypatch.setattr(
        runner,
        "evaluate_extrapolation_stream",
        lambda *args, **kwargs: {"accuracy": 0.96, "perplexity": 1.0},
    )
    runner.main()
    record = json.loads((tmp_path / "context-2000-sanity.jsonl").read_text())
    assert (record["method"], record["candidate"], record["seed"]) == ("alibi", 1, 100)
    assert calls[0][3] == 2000


def test_budget_summary_and_decision_keep_500_candidates_frozen(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(runner, "load_records", lambda path, config: [])
    monkeypatch.setattr(
        runner,
        "eligible_candidates",
        lambda records, candidates: seen.append(candidates) or {"alibi": [1]},
    )
    monkeypatch.setattr(runner, "select_frozen_candidates", lambda *args: {"alibi": 1})
    monkeypatch.setattr(runner, "reporting_summary", lambda *args: {"methods": {}})
    monkeypatch.setattr(runner, "decision_from_summaries", lambda *args: {"rope-log-window": False})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_extrapolation",
            "--phase",
            "decision",
            "--axis",
            "context",
            "--work-dir",
            str(tmp_path),
        ],
    )
    runner.main()
    assert seen[0]["alibi"] == [0, 1]
    assert seen[1]["alibi"] == [1]
    assert seen[1]["nope"] == []
    assert '"rope-log-window": false' in capsys.readouterr().out
