"""Resumable tuning and evaluation of successor counting."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zeta_scaling_toolkit import AsymptoticLRScheduler

if __package__:
    from .context_char import BUCKETS
    from .context_reuse import reuse_rope_trial
    from .length_generalization import (
        SequenceLM,
        encode_successors,
        positive_int,
        sample_text_batch,
    )
else:
    from context_char import BUCKETS
    from context_reuse import reuse_rope_trial
    from length_generalization import SequenceLM, encode_successors, positive_int, sample_text_batch

METHODS = (
    "abs",
    "rope-full",
    "rope-linear",
    "rope-ntk",
    "alibi",
    "rope-log-full",
    "rope-window",
    "rope-log-window",
)
LENGTHS = (256, 512, 1024, 4096)


def candidate(method: str, index: int) -> dict[str, float]:
    if method not in METHODS or index not in (0, 1):
        raise ValueError("unknown method or candidate")
    return {
        "learning_rate": (0.0007, 0.001)[index]
        if method in ("abs", "rope-full", "alibi", "rope-window")
        else 0.001,
        "tau0": (5.0, 10.0)[index] if method in ("rope-log-full", "rope-log-window") else 10.0,
        "rope_scale": (4.0, 16.0)[index] if method in ("rope-linear", "rope-ntk") else 1.0,
    }


def train_successor(
    method: str, seed: int, steps: int, batch_size: int, settings: dict[str, float]
) -> SequenceLM:
    data = encode_successors(0, 1000)
    torch.manual_seed(seed)
    training_variant = "rope-full" if method in ("rope-linear", "rope-ntk") else method
    model = SequenceLM(11, training_variant, False, 256, settings["tau0"], 10000.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=0.1, total_steps=steps)
    generator = torch.Generator().manual_seed(seed + 50000)
    for _ in range(steps):
        inputs, targets = sample_text_batch(data, 256, batch_size, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs).reshape(-1, 11), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite successor loss for {method}/{seed}")
        loss.backward()
        optimizer.step()
        scheduler.step()
    model.eval()
    if method in ("rope-linear", "rope-ntk"):
        model.variant = method
        for block in model.blocks:
            block.variant = method
            block.rope_scale = settings["rope_scale"]
    return model


def score_successor_batch(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, dict[str, float | int]]:
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite successor evaluation logits")
    predictions = logits.argmax(dim=-1)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    scores = {}
    for start, end in BUCKETS:
        selected = targets[:, start:end]
        if not selected.numel():
            continue
        correct = predictions[:, start:end] == selected
        digits = selected != 10
        name = f"{start}_{end - 1}" if end is not None else f"{start}_plus"
        scores[name] = {
            "correct": int(correct.sum()),
            "total": selected.numel(),
            "digit_correct": int((correct & digits).sum()),
            "digit_total": int(digits.sum()),
            "nll_sum": float(losses[:, start:end].sum()),
        }
    return scores


def evaluate_successor(
    model: SequenceLM,
    seed: int,
    lengths: list[int],
    samples: int,
    held_out_only: bool = False,
) -> dict[str, Any]:
    streams = {
        "3_digit_train_range_cyclic": encode_successors(100, 1000).repeat(2),
        "4_digit_held_out": encode_successors(1000, 3000),
    }
    results: dict[str, Any] = {}
    with torch.no_grad():
        for digit_length, data in streams.items():
            if held_out_only and digit_length != "4_digit_held_out":
                continue
            results[digit_length] = {}
            for length in lengths:
                if model.variant == "abs" and length > model.train_length:
                    continue
                generator = torch.Generator().manual_seed(seed + length + 110000)
                totals: dict[str, dict[str, float | int]] = {}
                for start in range(0, samples, 2):
                    inputs, targets = sample_text_batch(
                        data, length, min(2, samples - start), generator
                    )
                    scores = score_successor_batch(model(inputs), targets)
                    for bucket, counts in scores.items():
                        if bucket not in totals:
                            totals[bucket] = {key: 0 for key in counts}
                        for key, value in counts.items():
                            totals[bucket][key] += value
                results[digit_length][str(length)] = totals
    return results


def select_candidates(
    records: list[dict[str, Any]], methods: list[str], steps: int
) -> dict[str, int]:
    selected = {}
    for method in methods:
        means = []
        for index in (0, 1):
            trials = [
                record
                for record in records
                if record["method"] == method
                and record["candidate"] == index
                and record["steps"] == steps
            ]
            if {record["seed"] for record in trials} != set(range(100, 110)) or len(trials) != 10:
                raise ValueError(f"incomplete tuning seeds for {method} candidate {index}")
            length = "256" if method == "abs" else "4096"
            accuracies = []
            for record in trials:
                buckets = record["metrics"]["4_digit_held_out"][length]
                scores = list(buckets.values()) if method == "abs" else [buckets["1024_4095"]]
                accuracies.append(
                    sum(score["correct"] for score in scores)
                    / sum(score["total"] for score in scores)
                )
            means.append(sum(accuracies) / len(accuracies))
        selected[method] = max((0, 1), key=lambda index: means[index])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=500)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--samples", type=positive_int, default=8)
    parser.add_argument("--lengths", nargs="+", type=positive_int, default=list(LENGTHS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 110)))
    parser.add_argument("--candidates", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--tuning-records", type=Path)
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--max-runs", type=positive_int)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/context-ability-successor-tuning500.jsonl")
    )
    args = parser.parse_args()
    if (
        len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
        or len(set(args.methods)) != len(args.methods)
        or len(set(args.candidates)) != len(args.candidates)
    ):
        parser.error("seeds must be distinct and nonnegative")
    if len(set(args.lengths)) != len(args.lengths) or max(args.lengths) + 1 >= 7200:
        parser.error("contexts must be distinct and shorter than the evaluation streams")
    if args.tuning and args.tuning_records is not None:
        parser.error("tuning and selected-candidate reporting are separate runs")
    selected = None
    if args.tuning_records is not None:
        if set(args.seeds) & set(range(100, 110)):
            parser.error("reporting seeds must be disjoint from tuning seeds")
        tuning_records = [
            json.loads(line) for line in args.tuning_records.read_text().splitlines() if line
        ]
        if any(
            not record["tuning"]
            or record["steps"] != args.steps
            or record["batch_size"] != args.batch_size
            or record["samples"] != args.samples
            or record["settings"] != candidate(record["method"], record["candidate"])
            for record in tuning_records
        ):
            raise ValueError("tuning records do not match the reporting configuration")
        selected = select_candidates(tuning_records, args.methods, args.steps)
    torch.set_num_threads(1)
    configuration = {
        "task": "successor",
        "steps": args.steps,
        "train_length": 256,
        "batch_size": args.batch_size,
        "samples": args.samples,
        "lengths": args.lengths,
        "tuning": args.tuning,
    }
    records: list[dict[str, Any]] = []
    if args.output.exists():
        records = [json.loads(line) for line in args.output.read_text().splitlines() if line]
    if any(any(record[key] != value for key, value in configuration.items()) for record in records):
        raise ValueError("output contains a different experiment configuration")
    completed = {(record["method"], record["candidate"], record["seed"]) for record in records}
    cache: dict[int, SequenceLM] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_runs = 0
    with args.output.open("a", encoding="utf-8") as output:
        for method in args.methods:
            for index in [selected[method]] if selected is not None else args.candidates:
                for seed in args.seeds:
                    if (method, index, seed) in completed:
                        continue
                    settings = candidate(method, index)
                    model = reuse_rope_trial(
                        cache,
                        method,
                        seed,
                        settings,
                        lambda: train_successor(
                            method, seed, args.steps, args.batch_size, settings
                        ),
                    )
                    lengths = [256 if method == "abs" else 4096] if args.tuning else args.lengths
                    metrics = evaluate_successor(
                        model, seed, lengths, args.samples, held_out_only=args.tuning
                    )
                    record = {
                        **configuration,
                        "method": method,
                        "candidate": index,
                        "settings": settings,
                        "seed": seed,
                        "metrics": metrics,
                        "environment": {
                            "python": platform.python_version(),
                            "torch": torch.__version__,
                            "platform": platform.platform(),
                            "torch_threads": torch.get_num_threads(),
                        },
                    }
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                    completed.add((method, index, seed))
                    print(
                        json.dumps({"method": method, "candidate": index, "seed": seed}), flush=True
                    )
                    new_runs += 1
                    if args.max_runs is not None and new_runs >= args.max_runs:
                        return


if __name__ == "__main__":
    main()
