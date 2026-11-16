"""Resumable tuning and evaluation of successor counting."""

from __future__ import annotations

import argparse
import json
import math
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
EXTRAPOLATION_AXES = ("context", "digit_4", "digit_5")
EXTRAPOLATION_METHODS = (
    "nope",
    "rope-full",
    "rope-linear",
    "rope-ntk",
    "alibi",
    "rope-window",
    "rope-log-full",
    "rope-log-window",
)


def extrapolation_streams(axis: str) -> dict[str, torch.Tensor]:
    if axis == "context":
        train = encode_successors(100, 700)
        unseen = encode_successors(700, 1000)
    elif axis == "digit_4":
        train = encode_successors(0, 1000)
        unseen = encode_successors(1000, 3000)
    elif axis == "digit_5":
        train = encode_successors(0, 10000)
        unseen = encode_successors(10000, 30000)
    else:
        raise ValueError(f"unknown extrapolation axis: {axis}")
    return {"train": train, "seen": train, "unseen": unseen}


def sample_successor_offsets(
    data: torch.Tensor,
    context: int,
    batch_size: int,
    generator: torch.Generator,
    partition: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if partition not in ("train", "sanity", "unseen"):
        raise ValueError(f"unknown successor partition: {partition}")
    available = len(data) - context if context < len(data) else len(data)
    offsets = torch.arange(available)
    held_out = (offsets ^ (offsets >> 3) ^ (offsets >> 6)).remainder(5) == 0
    if partition == "train":
        offsets = offsets[~held_out]
    elif partition == "sanity":
        offsets = offsets[held_out]
    selected = offsets[torch.randint(len(offsets), (batch_size,), generator=generator)]
    repeated = data.repeat(math.ceil((len(data) + context + 1) / len(data)))
    samples = repeated[selected[:, None] + torch.arange(context + 1)]
    return samples[:, :-1], samples[:, 1:], selected


def successor_sanity_gate(records: list[dict[str, Any]], seeds: range) -> bool:
    if set(seeds) not in (set(range(100, 110)), set(range(42, 52))):
        raise ValueError("sanity gate requires a complete tuning or reporting seed set")
    if len(records) != 10 or {record["seed"] for record in records} != set(seeds):
        raise ValueError("sanity gate requires exactly one record per seed")
    accuracies = [record["sanity_accuracy"] for record in records]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in accuracies):
        raise ValueError("sanity accuracies must be finite probabilities")
    return sum(value >= 0.95 for value in accuracies) >= 8


def extrapolation_candidate(method: str, index: int) -> dict[str, float]:
    if method not in EXTRAPOLATION_METHODS or index not in (0, 1):
        raise ValueError("unknown extrapolation method or candidate")
    return {"learning_rate": (0.0007, 0.001)[index], "tau0": 10.0, "rope_scale": 4.0}


def train_extrapolation(
    method: str, axis: str, seed: int, steps: int, batch_size: int, settings: dict[str, float]
) -> SequenceLM:
    data = extrapolation_streams(axis)["train"]
    torch.manual_seed(seed)
    training_variant = "rope-full" if method in ("rope-linear", "rope-ntk") else method
    model = SequenceLM(11, training_variant, False, 256, settings["tau0"], 10000.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=0.1, total_steps=steps)
    generator = torch.Generator().manual_seed(seed + 50000)
    for _ in range(steps):
        inputs, targets, _ = sample_successor_offsets(data, 256, batch_size, generator, "train")
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs).reshape(-1, 11), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite extrapolation loss for {axis}/{method}/{seed}")
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


def successor_candidate(method: str, index: int) -> dict[str, float]:
    if index in (0, 1):
        return candidate(method, index)
    if index == 3 and method in ("rope-log-full", "rope-log-window"):
        settings = candidate(method, 0)
        settings["tau0"] = 2.0
        return settings
    if method not in METHODS or index != 2:
        raise ValueError("unknown method or candidate")
    settings = candidate(method, 0)
    if method in ("abs", "rope-full", "alibi", "rope-window"):
        settings["learning_rate"] = 0.0015
    elif method in ("rope-linear", "rope-ntk"):
        settings["rope_scale"] = 8.0
    else:
        settings["tau0"] = 15.0
    return settings


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


def successor_metadata(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    roles = torch.zeros_like(data)
    last_digits = torch.zeros_like(data, dtype=torch.bool)
    carry = torch.full_like(data, -1)
    previous: list[int] = []
    number: list[int] = []
    start = 0
    for position, token in enumerate(data.tolist()):
        if token != 10:
            number.append(token)
            continue
        if number:
            roles[start:position] = torch.arange(1, len(number) + 1)
            last_digits[position - 1] = True
            if (
                len(number) == len(previous)
                and int("".join(map(str, number))) == int("".join(map(str, previous))) + 1
            ):
                for index, (digit, old_digit) in enumerate(zip(number, previous)):
                    carry[start + index] = int(index < len(number) - 1 and digit != old_digit)
        previous = number
        number = []
        start = position + 1
    return roles, last_digits, carry


def successor_diagnostics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    starts: torch.Tensor,
    metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, dict[str, dict[str, int]]]:
    positions = starts[:, None] + torch.arange(targets.shape[1])[None, :] + 1
    roles, last_digits, carry = (values[positions] for values in metadata)
    correct = predictions == targets
    results: dict[str, dict[str, dict[str, int]]] = {}
    for start, end in BUCKETS:
        selected = targets[:, start:end]
        if not selected.numel():
            continue
        name = f"{start}_{end - 1}" if end is not None else f"{start}_plus"
        selected_roles = roles[:, start:end]
        selected_carry = carry[:, start:end]
        selected_correct = correct[:, start:end]
        counts = {}
        for role in range(max(4, int(roles.max())) + 1):
            mask = selected_roles == role
            label = "delimiter" if role == 0 else f"digit_{role}"
            counts[label] = {
                "correct": int((selected_correct & mask).sum()),
                "total": int(mask.sum()),
            }
        for label, value in (("carry", 1), ("non_carry", 0)):
            mask = selected_carry == value
            counts[label] = {
                "correct": int((selected_correct & mask).sum()),
                "total": int(mask.sum()),
            }

        rule_start = max(start, 5)
        rule_end = targets.shape[1] if end is None else min(end, targets.shape[1])
        if rule_start < rule_end:
            actual = targets[:, rule_start:rule_end]
            copied = targets[:, rule_start - 5 : rule_end - 5]
            incremented = torch.where(
                last_digits[:, rule_start:rule_end], (copied + 1) % 10, copied
            )
            majority = sum(int(torch.bincount(row, minlength=11).max()) for row in actual)
            counts["majority_per_window_oracle"] = {"correct": majority, "total": actual.numel()}
            counts["copy_5"] = {"correct": int((copied == actual).sum()), "total": actual.numel()}
            counts["copy_5_increment_last_oracle"] = {
                "correct": int((incremented == actual).sum()),
                "total": actual.numel(),
            }
        results[name] = counts
    return results


def evaluate_extrapolation_stream(
    model: SequenceLM,
    data: torch.Tensor,
    partition: str,
    seed: int,
    length: int,
    samples: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed + length + 110000)
    repeated = data.repeat(math.ceil((len(data) + length + 1) / len(data)))
    metadata = successor_metadata(repeated)
    totals: dict[str, dict[str, float | int]] = {}
    diagnostics: dict[str, dict[str, dict[str, int]]] = {}
    with torch.no_grad():
        for start in range(0, samples, 2):
            inputs, targets, offsets = sample_successor_offsets(
                data, length, min(2, samples - start), generator, partition
            )
            logits = model(inputs)
            for bucket, counts in score_successor_batch(logits, targets).items():
                accumulated = totals.setdefault(bucket, {key: 0 for key in counts})
                for key, value in counts.items():
                    accumulated[key] += value
            for bucket, groups in successor_diagnostics(
                targets, logits.argmax(dim=-1), offsets, metadata
            ).items():
                for group, counts in groups.items():
                    accumulated_diagnostic = diagnostics.setdefault(bucket, {}).setdefault(
                        group, {"correct": 0, "total": 0}
                    )
                    for key, value in counts.items():
                        accumulated_diagnostic[key] += value
    total = sum(bucket["total"] for bucket in totals.values())
    return {
        "accuracy": sum(bucket["correct"] for bucket in totals.values()) / total,
        "perplexity": math.exp(sum(bucket["nll_sum"] for bucket in totals.values()) / total),
        "buckets": totals,
        "diagnostics": diagnostics,
    }


def evaluate_successor(
    model: SequenceLM,
    seed: int,
    lengths: list[int],
    samples: int,
    held_out_only: bool = False,
    diagnostics: bool = False,
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
            metadata = successor_metadata(data) if diagnostics else None
            for length in lengths:
                if model.variant == "abs" and length > model.train_length:
                    continue
                generator = torch.Generator().manual_seed(seed + length + 110000)
                totals: dict[str, dict[str, float | int]] = {}
                diagnostic_totals: dict[str, dict[str, dict[str, int]]] = {}
                for start in range(0, samples, 2):
                    count = min(2, samples - start)
                    if metadata is not None:
                        mirror = torch.Generator()
                        mirror.set_state(generator.get_state())
                        starts = torch.randint(
                            0, len(data) - length - 1, (count,), generator=mirror
                        )
                    inputs, targets = sample_text_batch(data, length, count, generator)
                    if metadata is not None:
                        positions = starts[:, None] + torch.arange(length + 1)
                        sampled = data[positions]
                        if not torch.equal(sampled[:, :-1], inputs) or not torch.equal(
                            sampled[:, 1:], targets
                        ):
                            raise RuntimeError("successor diagnostic sample offsets do not match")
                    logits = model(inputs)
                    scores = score_successor_batch(logits, targets)
                    if metadata is not None:
                        batch_diagnostics = successor_diagnostics(
                            targets, logits.argmax(dim=-1), starts, metadata
                        )
                        for bucket, groups in batch_diagnostics.items():
                            for group, values in groups.items():
                                accumulated = diagnostic_totals.setdefault(bucket, {}).setdefault(
                                    group, {"correct": 0, "total": 0}
                                )
                                for key, value in values.items():
                                    accumulated[key] += value
                    for bucket, counts in scores.items():
                        if bucket not in totals:
                            totals[bucket] = {key: 0 for key in counts}
                        for key, value in counts.items():
                            totals[bucket][key] += value
                results[digit_length][str(length)] = totals
                if diagnostics:
                    results.setdefault("diagnostics", {}).setdefault(digit_length, {})[
                        str(length)
                    ] = diagnostic_totals
    return results


def select_candidates(
    records: list[dict[str, Any]], methods: list[str], steps: int
) -> dict[str, int]:
    selected = {}
    for method in methods:
        indices = sorted(
            {
                record["candidate"]
                for record in records
                if record["method"] == method and record["steps"] == steps
            }
        )
        if not {0, 1}.issubset(indices):
            raise ValueError(f"missing original tuning candidates for {method}")
        means: dict[int, float] = {}
        for index in indices:
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
            means[index] = sum(accuracies) / len(accuracies)
        selected[method] = max(indices, key=lambda index: means[index])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=500)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--samples", type=positive_int, default=8)
    parser.add_argument("--lengths", nargs="+", type=positive_int, default=list(LENGTHS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 110)))
    parser.add_argument("--candidates", nargs="+", type=int, choices=(0, 1, 2, 3), default=[0, 1])
    parser.add_argument("--tuning-records", type=Path)
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
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
    if (
        args.tuning_records is None
        and 3 in args.candidates
        and any(method not in ("rope-log-full", "rope-log-window") for method in args.methods)
    ):
        parser.error("candidate 3 is only available for rope-log-full and rope-log-window")
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
            or record["settings"] != successor_candidate(record["method"], record["candidate"])
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
    if args.diagnostics:
        configuration["diagnostics"] = True
    records: list[dict[str, Any]] = []
    if args.output.exists():
        records = [json.loads(line) for line in args.output.read_text().splitlines() if line]
    if any(
        any(record.get(key) != value for key, value in configuration.items())
        or record.get("diagnostics", False) != args.diagnostics
        for record in records
    ):
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
                    settings = successor_candidate(method, index)
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
                        model,
                        seed,
                        lengths,
                        args.samples,
                        held_out_only=args.tuning,
                        **({"diagnostics": True} if args.diagnostics else {}),
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
                    if args.diagnostics:
                        print(json.dumps({"diagnostics": metrics["diagnostics"]}), flush=True)
                    new_runs += 1
                    if args.max_runs is not None and new_runs >= args.max_runs:
                        return


if __name__ == "__main__":
    main()
