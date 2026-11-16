import pytest

from benchmarks.context_char import LENGTHS, PREFIX_LENGTHS, VARIANTS
from benchmarks.context_report import (
    character_table,
    needle_table,
    paired_summary,
    successor_table,
    tuned_character_table,
)
from benchmarks.context_successor import METHODS


def test_paired_summary_uses_matched_seeds_and_rejects_nonfinite_values():
    baseline = {seed: float(seed) for seed in range(42, 52)}
    values = {seed: value + 1 for seed, value in baseline.items()}
    assert paired_summary(values, baseline) == (47.5, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="exactly seeds"):
        paired_summary({42: 1.0}, baseline)
    values[42] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        paired_summary(values, baseline)


def test_character_table_requires_complete_paired_trials(monkeypatch):
    monkeypatch.setattr("benchmarks.context_report.bootstrap_interval", lambda values: (0.0, 0.0))
    records = []
    buckets = {name: 2.0 for name in ("0_63", "64_255", "256_1023", "1024_4095", "4096_plus")}
    metrics = {
        "perplexity": {str(length): 5.0 for length in LENGTHS},
        "position_nll": {"16384": buckets},
        "context_nll": {str(length): 2.0 for length in PREFIX_LENGTHS},
    }
    for variant in VARIANTS:
        for seed in range(42, 52):
            records.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "steps": 500,
                    "train_length": 256,
                    "window": 64,
                    "sinks": 4,
                    "batch_size": 16,
                    "eval_batch_size": 2,
                    "lengths": list(LENGTHS),
                    "dataset_sha256": "test",
                    "metrics": {"full": metrics, "window": metrics},
                }
            )
    table = character_table(records)
    assert "| rope-log-full | full | 16384 | 5.0000 | +0.0000 [+0.0000, +0.0000] |" in table
    assert "| rope-window | window | 4096_plus | 2.0000 |" in table
    buckets["4096_plus"] = float("nan")
    with pytest.raises(ValueError, match="non-finite position"):
        character_table(records)
    buckets["4096_plus"] = 2.0
    with pytest.raises(ValueError, match="every variant"):
        character_table(records[:-1])
    records[0]["steps"] = 2000
    with pytest.raises(ValueError, match="inconsistent"):
        character_table(records)


def test_successor_table_compares_paired_accuracy_to_tuning_selected_baseline(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.context_report.select_candidates",
        lambda records, methods, steps: dict.fromkeys(methods, 0),
    )
    monkeypatch.setattr(
        "benchmarks.context_report.bootstrap_interval", lambda values: (min(values), max(values))
    )
    records = []
    tuning = []
    for method in METHODS:
        correct = {"rope-full": 8, "rope-log-full": 9}.get(method, 6)
        counts = {
            "correct": correct,
            "total": 10,
            "digit_correct": 6,
            "digit_total": 8,
            "nll_sum": 10.0,
        }
        metrics = {
            digit_range: {
                str(length): {"0_63": counts, "1024_4095": counts}
                for length in (256, 512, 1024, 4096)
            }
            for digit_range in ("3_digit_train_range_cyclic", "4_digit_held_out")
        }
        for seed in range(42, 52):
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "candidate": 0,
                    "steps": 500,
                    "batch_size": 16,
                    "samples": 8,
                    "metrics": metrics,
                }
            )
        tuning.append({"method": method, "candidate": 0, "steps": 500, "metrics": metrics})
    table = successor_table(records, tuning)
    assert "baseline selected on tuning seeds: `rope-full`" in table
    assert "| rope-log-full | 4_digit_held_out | 4096 | 1024_4095 | 0.9000 | +0.1000" in table
    assert "| abs | 4_digit_held_out | 512 |" not in table
    with pytest.raises(ValueError, match="every method"):
        successor_table(records[:-1], tuning)


def test_tuned_character_table_selects_baseline_without_reporting_seeds(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.context_report.select_char_candidates",
        lambda records, methods, steps: dict.fromkeys(methods, 0),
    )
    monkeypatch.setattr(
        "benchmarks.context_report.bootstrap_interval", lambda values: (min(values), max(values))
    )
    records = []
    tuning = []
    for method in METHODS:
        loss = {"rope-full": 1.0, "rope-log-full": 0.9}.get(method, 2.0)
        for seed in range(42, 52):
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "candidate": 0,
                    "steps": 500,
                    "batch_size": 16,
                    "eval_batch_size": 2,
                    "dataset_sha256": "test",
                    "metrics": {
                        "context_nll": {str(length): loss for length in PREFIX_LENGTHS},
                        "perplexity": {"256": loss + 1.0},
                    },
                }
            )
        tuning.append({"method": method, "candidate": 0, "steps": 500, "tuning_nll": loss})
    table = tuned_character_table(records, tuning)
    assert "baseline selected on tuning seeds: `rope-full`" in table
    assert "| rope-log-full | 1024 | 0.9000 | -0.1000" in table
    assert "| abs | 1024 |" not in table
    with pytest.raises(ValueError, match="every method"):
        tuned_character_table(records[:-1], tuning)


def test_needle_table_requires_sanity_and_uses_paired_tuned_baseline(monkeypatch):
    monkeypatch.setattr("benchmarks.context_report.sanity_passed", lambda records: True)
    monkeypatch.setattr(
        "benchmarks.context_report.select_needle_candidates",
        lambda records, methods, steps: dict.fromkeys(methods, 0),
    )
    monkeypatch.setattr(
        "benchmarks.context_report.bootstrap_interval", lambda values: (min(values), max(values))
    )
    records = []
    tuning = []
    for method in METHODS:
        accuracy = {"rope-full": 0.8, "rope-log-full": 0.9}.get(method, 0.6)
        metrics = {str(length): {"accuracy": accuracy} for length in (256, 512, 1024, 4096)}
        for seed in range(42, 52):
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "candidate": 0,
                    "steps": 500,
                    "batch_size": 16,
                    "eval_queries": 200,
                    "metrics": metrics,
                }
            )
        tuning.append({"method": method, "candidate": 0, "steps": 500, "metrics": metrics})
    table = needle_table(records, tuning, [])
    assert "baseline selected on tuning seeds: `rope-full`" in table
    assert "| rope-log-full | 1024 | 0.9000 | +0.1000" in table
    assert "| abs | 1024 |" not in table
    monkeypatch.setattr("benchmarks.context_report.sanity_passed", lambda records: False)
    with pytest.raises(ValueError, match="sanity gate failed"):
        needle_table(records, tuning, [])
