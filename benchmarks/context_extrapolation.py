"""Gated, disjoint-seed successor extrapolation experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .context_successor import (
        EXTRAPOLATION_AXES,
        EXTRAPOLATION_METHODS,
        evaluate_extrapolation_stream,
        extrapolation_candidate,
        extrapolation_streams,
        successor_sanity_gate,
        train_extrapolation,
    )
    from .length_generalization import SequenceLM, bootstrap_interval, positive_int
else:
    from context_successor import (
        EXTRAPOLATION_AXES,
        EXTRAPOLATION_METHODS,
        evaluate_extrapolation_stream,
        extrapolation_candidate,
        extrapolation_streams,
        successor_sanity_gate,
        train_extrapolation,
    )
    from length_generalization import SequenceLM, bootstrap_interval, positive_int

TUNING_SEEDS = range(100, 110)
REPORTING_SEEDS = range(42, 52)
CANDIDATES = (0, 1)
BASELINES = ("nope", "rope-full", "rope-linear", "rope-ntk", "alibi", "rope-window")
TOOLKIT = ("rope-log-full", "rope-log-window")


def load_records(path: Path, configuration: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(
        any(record.get(key) != value for key, value in configuration.items()) for record in records
    ):
        raise ValueError(f"incompatible experiment records: {path}")
    keys = [(r["method"], r["candidate"], r["seed"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate experiment records: {path}")
    return records


def eligible_candidates(
    records: list[dict[str, Any]], candidates_by_method: dict[str, list[int]] | None = None
) -> dict[str, list[int]]:
    if candidates_by_method is None:
        candidates_by_method = {method: list(CANDIDATES) for method in EXTRAPOLATION_METHODS}
    expected = {
        (method, candidate, seed)
        for method in EXTRAPOLATION_METHODS
        for candidate in candidates_by_method[method]
        for seed in TUNING_SEEDS
    }
    if {(r["method"], r["candidate"], r["seed"]) for r in records} != expected or len(
        records
    ) != len(expected):
        raise ValueError("sanity tuning requires both candidates and all tuning seeds per method")
    return {
        method: [
            candidate
            for candidate in candidates_by_method[method]
            if successor_sanity_gate(
                [
                    record
                    for record in records
                    if record["method"] == method and record["candidate"] == candidate
                ],
                TUNING_SEEDS,
            )
        ]
        for method in EXTRAPOLATION_METHODS
    }


def selection_score(record: dict[str, Any], axis: str) -> float:
    if axis == "context":
        bucket = record["unseen"]["4096"]["buckets"]["1024_4095"]
        value = bucket["correct"] / bucket["total"]
    else:
        value = record["unseen"]["256"]["accuracy"]
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("tuning accuracy must be a finite probability")
    return value


def select_frozen_candidates(
    records: list[dict[str, Any]], eligible: dict[str, list[int]], axis: str
) -> dict[str, int]:
    expected = {
        (method, candidate, seed)
        for method, candidates in eligible.items()
        for candidate in candidates
        for seed in TUNING_SEEDS
    }
    if {(r["method"], r["candidate"], r["seed"]) for r in records} != expected or len(
        records
    ) != len(expected):
        raise ValueError("tuning evaluation is incomplete or contains ineligible candidates")
    return {
        method: max(
            candidates,
            key=lambda index: statistics.mean(
                selection_score(record, axis)
                for record in records
                if record["method"] == method and record["candidate"] == index
            ),
        )
        for method, candidates in eligible.items()
        if candidates
    }


def paired_results(values: list[float], reference: list[float]) -> dict[str, float]:
    if (
        len(values) != 10
        or len(reference) != 10
        or not all(math.isfinite(value) for value in (*values, *reference))
    ):
        raise ValueError("paired results require ten finite matched-seed values")
    differences = [value - baseline for value, baseline in zip(values, reference)]
    lower, upper = bootstrap_interval(differences)
    return {
        "mean": statistics.mean(values),
        "difference": statistics.mean(differences),
        "ci_low": lower,
        "ci_high": upper,
    }


def reporting_summary(
    records: list[dict[str, Any]],
    tuning: list[dict[str, Any]],
    selected: dict[str, int],
    axis: str,
) -> dict[str, Any]:
    expected = {
        (method, index, seed) for method, index in selected.items() for seed in REPORTING_SEEDS
    }
    if {(r["method"], r["candidate"], r["seed"]) for r in records} != expected or len(
        records
    ) != len(expected):
        raise ValueError("reporting requires exactly the frozen candidates and reporting seeds")
    trials = {
        method: sorted(
            (record for record in records if record["method"] == method), key=lambda r: r["seed"]
        )
        for method in selected
    }
    valid = {
        method: successor_sanity_gate(rows, REPORTING_SEEDS) for method, rows in trials.items()
    }
    baselines = [method for method in BASELINES if method in selected]
    baseline = (
        max(
            baselines,
            key=lambda method: statistics.mean(
                selection_score(record, axis)
                for record in tuning
                if record["method"] == method and record["candidate"] == selected[method]
            ),
        )
        if baselines
        else None
    )
    result: dict[str, Any] = {"baseline": baseline, "methods": {}}
    for method, rows in trials.items():
        sanity = [row["sanity_accuracy"] for row in rows]
        entry: dict[str, Any] = {
            "in_distribution_accuracy": statistics.mean(sanity),
            "sanity_seeds_passed": sum(value >= 0.95 for value in sanity),
            "status": "valid" if valid[method] else "undertrained",
        }
        result["methods"][method] = entry
        if not valid[method] or baseline is None or not valid[baseline]:
            continue
        reference = trials[baseline]
        perplexity = [row["in_distribution"]["perplexity"] for row in rows]
        reference_ppl = [row["in_distribution"]["perplexity"] for row in reference]
        entry["training_perplexity"] = paired_results(perplexity, reference_ppl)
        entry["relative_training_perplexity"] = (
            statistics.mean(perplexity) / statistics.mean(reference_ppl) - 1
        )
        entry["streams"] = {}
        entry["diagnostics"] = {}
        for stream in ("seen", "unseen"):
            entry["streams"][stream] = {}
            entry["diagnostics"][stream] = {}
            for length, metrics in rows[0][stream].items():
                entry["streams"][stream][length] = {}
                entry["diagnostics"][stream][length] = {}
                entry["streams"][stream][length]["overall"] = paired_results(
                    [row[stream][length]["accuracy"] for row in rows],
                    [row[stream][length]["accuracy"] for row in reference],
                )
                for bucket in metrics["buckets"]:
                    values = [
                        row[stream][length]["buckets"][bucket]["correct"]
                        / row[stream][length]["buckets"][bucket]["total"]
                        for row in rows
                    ]
                    ref_values = [
                        row[stream][length]["buckets"][bucket]["correct"]
                        / row[stream][length]["buckets"][bucket]["total"]
                        for row in reference
                    ]
                    entry["streams"][stream][length][bucket] = paired_results(values, ref_values)
                for bucket, groups in metrics["diagnostics"].items():
                    entry["diagnostics"][stream][length][bucket] = {}
                    for group in groups:
                        counts = [row[stream][length]["diagnostics"][bucket][group] for row in rows]
                        reference_counts = [
                            row[stream][length]["diagnostics"][bucket][group] for row in reference
                        ]
                        if all(count["total"] for count in (*counts, *reference_counts)):
                            entry["diagnostics"][stream][length][bucket][group] = paired_results(
                                [count["correct"] / count["total"] for count in counts],
                                [count["correct"] / count["total"] for count in reference_counts],
                            )
                        else:
                            entry["diagnostics"][stream][length][bucket][group] = None
    return result


def decision_from_summaries(summaries: dict[int, dict[str, Any]], axis: str) -> dict[str, bool]:
    decisions = {}
    for method in TOOLKIT:
        outcomes = []
        for steps in (500, 2000):
            summary = summaries.get(steps, {})
            entry = summary.get("methods", {}).get(method, {})
            if entry.get("status") != "valid" or "streams" not in entry:
                outcomes.append(False)
                continue
            if axis == "context":
                accuracy = entry["streams"]["unseen"]["4096"]["1024_4095"]
            else:
                accuracy = entry["streams"]["unseen"]["256"]["overall"]
            outcomes.append(
                accuracy["ci_low"] > 0 and abs(entry["relative_training_perplexity"]) <= 0.01
            )
        decisions[method] = all(outcomes)
    return decisions


def source_digest() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("context_extrapolation.py", "context_successor.py", "length_generalization.py"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def load_budget_summary(
    work_dir: Path, config: dict[str, Any], candidates_by_method: dict[str, list[int]]
) -> tuple[dict[str, Any], dict[str, int]]:
    prefix = f"{config['axis']}-{config['steps']}"
    sanity = load_records(work_dir / f"{prefix}-sanity.jsonl", config)
    eligible = eligible_candidates(sanity, candidates_by_method)
    tuning = load_records(work_dir / f"{prefix}-tuning.jsonl", config)
    selected = select_frozen_candidates(tuning, eligible, config["axis"])
    reporting = load_records(work_dir / f"{prefix}-reporting.jsonl", config)
    return reporting_summary(reporting, tuning, selected, config["axis"]), selected


def evaluate_streams(
    model: SequenceLM, streams: dict[str, torch.Tensor], axis: str, seed: int, samples: int
) -> dict[str, Any]:
    lengths = (256, 512, 1024, 4096) if axis == "context" else (256,)
    return {
        name: {
            str(length): evaluate_extrapolation_stream(
                model,
                streams[name],
                "sanity" if name == "seen" else "unseen",
                seed,
                length,
                samples,
            )
            for length in lengths
        }
        for name in ("seen", "unseen")
    }


def load_model(path: Path, method: str, candidate: int) -> SequenceLM:
    settings = extrapolation_candidate(method, candidate)
    model = SequenceLM(11, method, False, 256, settings["tau0"], 10000.0, rope_scale=4.0)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("sanity", "tune", "report", "summary", "decision"), required=True
    )
    parser.add_argument("--axis", choices=EXTRAPOLATION_AXES, required=True)
    parser.add_argument("--steps", type=positive_int, choices=(500, 2000), default=500)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--samples", type=positive_int, default=8)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-runs", type=positive_int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.axis}-{args.steps}"
    config = {
        "axis": args.axis,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "samples": args.samples,
        "protocol": 1,
        "source_sha256": source_digest(),
    }
    sanity_path = args.work_dir / f"{prefix}-sanity.jsonl"
    tuning_path = args.work_dir / f"{prefix}-tuning.jsonl"
    report_path = args.work_dir / f"{prefix}-reporting.jsonl"
    sanity = load_records(sanity_path, config)
    candidates_by_method = {method: list(CANDIDATES) for method in EXTRAPOLATION_METHODS}
    if args.phase == "decision":
        first, selected_500 = load_budget_summary(
            args.work_dir, {**config, "steps": 500}, candidates_by_method
        )
        frozen = {
            method: [selected_500[method]] if method in selected_500 else []
            for method in EXTRAPOLATION_METHODS
        }
        second, _ = load_budget_summary(args.work_dir, {**config, "steps": 2000}, frozen)
        print(json.dumps(decision_from_summaries({500: first, 2000: second}, args.axis)))
        return
    if args.steps == 2000:
        prior_config = {**config, "steps": 500}
        prior = load_records(args.work_dir / f"{args.axis}-500-sanity.jsonl", prior_config)
        prior_eligible = eligible_candidates(prior)
        if not any(prior_eligible[method] for method in TOOLKIT) or not any(
            prior_eligible[method] for method in BASELINES
        ):
            raise ValueError(
                "2000 steps require a passing toolkit and baseline 500-step sanity gate"
            )
        prior_tuning = load_records(args.work_dir / f"{args.axis}-500-tuning.jsonl", prior_config)
        prior_selected = select_frozen_candidates(prior_tuning, prior_eligible, args.axis)
        candidates_by_method = {
            method: [prior_selected[method]] if method in prior_selected else []
            for method in EXTRAPOLATION_METHODS
        }
    eligible = eligible_candidates(sanity, candidates_by_method) if args.phase != "sanity" else None
    tuning = load_records(tuning_path, config)
    selected = (
        select_frozen_candidates(tuning, eligible, args.axis)
        if args.phase in ("report", "summary") and eligible is not None
        else None
    )
    if args.phase == "summary":
        assert selected is not None
        reporting = load_records(report_path, config)
        print(json.dumps(reporting_summary(reporting, tuning, selected, args.axis), sort_keys=True))
        return
    destination = {"sanity": sanity_path, "tune": tuning_path, "report": report_path}[args.phase]
    existing = load_records(destination, config)
    completed = {(r["method"], r["candidate"], r["seed"]) for r in existing}
    streams = extrapolation_streams(args.axis)
    seeds = REPORTING_SEEDS if args.phase == "report" else TUNING_SEEDS
    runs = 0
    with destination.open("a", encoding="utf-8") as output:
        for method in EXTRAPOLATION_METHODS:
            if args.phase == "sanity":
                candidates = candidates_by_method[method]
            elif args.phase == "tune":
                assert eligible is not None
                candidates = eligible[method]
            else:
                assert selected is not None
                candidates = [selected[method]] if method in selected else []
            for index in candidates:
                for seed in seeds:
                    if (method, index, seed) in completed:
                        continue
                    settings = extrapolation_candidate(method, index)
                    checkpoint = args.work_dir / f"{prefix}-{method}-{index}-{seed}.pt"
                    if args.phase == "tune":
                        sanity_record = next(
                            r
                            for r in sanity
                            if (r["method"], r["candidate"], r["seed"]) == (method, index, seed)
                        )
                        if (
                            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                            != sanity_record["checkpoint_sha256"]
                        ):
                            raise ValueError(
                                f"checkpoint does not match sanity record: {checkpoint}"
                            )
                        model = load_model(checkpoint, method, index)
                        in_distribution = sanity_record["in_distribution"]
                    else:
                        model = train_extrapolation(
                            method, args.axis, seed, args.steps, args.batch_size, settings
                        )
                        in_distribution = evaluate_extrapolation_stream(
                            model, streams["train"], "sanity", seed, 256, args.samples
                        )
                        if args.phase == "sanity":
                            temporary = checkpoint.with_suffix(".tmp")
                            torch.save(model.state_dict(), temporary)
                            temporary.replace(checkpoint)
                    record = {
                        **config,
                        "method": method,
                        "candidate": index,
                        "seed": seed,
                        "settings": settings,
                        "in_distribution": in_distribution,
                        "sanity_accuracy": in_distribution["accuracy"],
                        "environment": {
                            "python": platform.python_version(),
                            "torch": torch.__version__,
                            "platform": platform.platform(),
                            "torch_threads": torch.get_num_threads(),
                        },
                    }
                    if args.phase != "sanity":
                        record.update(
                            evaluate_streams(model, streams, args.axis, seed, args.samples)
                        )
                    else:
                        record["checkpoint_sha256"] = hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest()
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                    print(
                        json.dumps(
                            {
                                "phase": args.phase,
                                "method": method,
                                "candidate": index,
                                "seed": seed,
                            }
                        ),
                        flush=True,
                    )
                    runs += 1
                    if args.max_runs is not None and runs >= args.max_runs:
                        return


if __name__ == "__main__":
    main()
