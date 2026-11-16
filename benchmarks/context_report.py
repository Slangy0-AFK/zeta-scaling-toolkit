"""Reproduce paired Stage 1 character tables from complete JSONL records."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

if __package__:
    from .context_char import LENGTHS, PREFIX_LENGTHS, VARIANTS
    from .context_char_tuned import select_candidates as select_char_candidates
    from .context_needle_tuned import sanity_passed
    from .context_needle_tuned import select_candidates as select_needle_candidates
    from .context_successor import METHODS, select_candidates
    from .length_generalization import bootstrap_interval
else:
    from context_char import LENGTHS, PREFIX_LENGTHS, VARIANTS
    from context_char_tuned import select_candidates as select_char_candidates
    from context_needle_tuned import sanity_passed
    from context_needle_tuned import select_candidates as select_needle_candidates
    from context_successor import METHODS, select_candidates
    from length_generalization import bootstrap_interval

SEEDS = tuple(range(42, 52))


def paired_summary(
    values: dict[int, float], baseline: dict[int, float]
) -> tuple[float, float, float, float]:
    if set(values) != set(SEEDS) or set(baseline) != set(SEEDS):
        raise ValueError("paired reporting requires exactly seeds 42 through 51")
    if not all(math.isfinite(value) for value in (*values.values(), *baseline.values())):
        raise ValueError("paired reporting requires finite measurements")
    differences = [values[seed] - baseline[seed] for seed in SEEDS]
    lower, upper = bootstrap_interval(differences)
    return statistics.mean(values.values()), statistics.mean(differences), lower, upper


def character_table(records: list[dict[str, Any]]) -> str:
    indexed = {(record["variant"], record["seed"]): record for record in records}
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if set(indexed) != expected or len(records) != len(expected):
        raise ValueError("character records must contain every variant and seed exactly once")
    configuration_keys = (
        "steps",
        "train_length",
        "window",
        "sinks",
        "batch_size",
        "eval_batch_size",
        "lengths",
        "dataset_sha256",
    )
    if any(any(record[key] != records[0][key] for key in configuration_keys) for record in records):
        raise ValueError("character records contain inconsistent experiment configurations")

    def values(variant: str, mode: str, metric: str, key: str) -> dict[int, float]:
        return {seed: float(indexed[variant, seed]["metrics"][mode][metric][key]) for seed in SEEDS}

    lines = [
        "| Training | Evaluation | Context | PPL | Difference vs RoPE+log full [95% CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for mode in ("full", "window"):
            for length in LENGTHS:
                key = str(length)
                mean, difference, lower, upper = paired_summary(
                    values(variant, mode, "perplexity", key),
                    values("rope-log-full", "full", "perplexity", key),
                )
                lines.append(
                    f"| {variant} | {mode} | {length} | {mean:.4f} | "
                    f"{difference:+.4f} [{lower:+.4f}, {upper:+.4f}] |"
                )
    lines += [
        "",
        "| Training | Evaluation | Position bucket | Mean NLL at context 16384 |",
        "|---|---|---|---:|",
    ]
    for variant in VARIANTS:
        for mode in ("full", "window"):
            for bucket in ("0_63", "64_255", "256_1023", "1024_4095", "4096_plus"):
                mean = statistics.mean(
                    indexed[variant, seed]["metrics"][mode]["position_nll"]["16384"][bucket]
                    for seed in SEEDS
                )
                if not math.isfinite(mean):
                    raise ValueError(f"non-finite position loss for {variant}/{mode}/{bucket}")
                lines.append(f"| {variant} | {mode} | {bucket} | {mean:.4f} |")
    lines += [
        "",
        "| Training | Evaluation | Prefix | NLL | Difference vs RoPE+log full [95% CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for mode in ("full", "window"):
            for length in PREFIX_LENGTHS:
                key = str(length)
                mean, difference, lower, upper = paired_summary(
                    values(variant, mode, "context_nll", key),
                    values("rope-log-full", "full", "context_nll", key),
                )
                lines.append(
                    f"| {variant} | {mode} | {length} | {mean:.4f} | "
                    f"{difference:+.4f} [{lower:+.4f}, {upper:+.4f}] |"
                )
    return "\n".join(lines)


def successor_table(records: list[dict[str, Any]], tuning: list[dict[str, Any]]) -> str:
    if not records:
        raise ValueError("successor records are empty")
    steps = records[0]["steps"]
    selected = select_candidates(tuning, list(METHODS), steps)
    indexed = {(record["method"], record["seed"]): record for record in records}
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(indexed) != expected or len(records) != len(expected):
        raise ValueError("successor records must contain every method and seed exactly once")
    if any(
        record["candidate"] != selected[record["method"]]
        or record["steps"] != steps
        or record["batch_size"] != records[0]["batch_size"]
        or record["samples"] != records[0]["samples"]
        for record in records
    ):
        raise ValueError("successor records do not match selected candidates or configuration")

    def tuning_accuracy(method: str) -> float:
        trials = [
            record
            for record in tuning
            if record["method"] == method
            and record["candidate"] == selected[method]
            and record["steps"] == steps
        ]
        return statistics.mean(
            record["metrics"]["4_digit_held_out"]["4096"]["1024_4095"]["correct"]
            / record["metrics"]["4_digit_held_out"]["4096"]["1024_4095"]["total"]
            for record in trials
        )

    baseline = max(("rope-full", "rope-linear", "rope-ntk", "alibi"), key=tuning_accuracy)
    lines = [
        f"Strongest eligible baseline selected on tuning seeds: `{baseline}`.",
        "",
        "| Method | Digit range | Context | Position bucket | Accuracy | "
        f"Difference vs {baseline} [95% CI] | Digit-only accuracy | "
        f"Digit-only difference vs {baseline} [95% CI] |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for digit_range in ("3_digit_train_range_cyclic", "4_digit_held_out"):
            for length in (256, 512, 1024, 4096):
                if method == "abs" and length > 256:
                    continue
                first = indexed[method, SEEDS[0]]["metrics"][digit_range][str(length)]
                for bucket in first:

                    def score(name: str, seed: int) -> dict[str, Any]:
                        return indexed[name, seed]["metrics"][digit_range][str(length)][bucket]

                    values = {
                        seed: score(method, seed)["correct"] / score(method, seed)["total"]
                        for seed in SEEDS
                    }
                    reference = {
                        seed: score(baseline, seed)["correct"] / score(baseline, seed)["total"]
                        for seed in SEEDS
                    }
                    mean, difference, lower, upper = paired_summary(values, reference)
                    digits = {
                        seed: score(method, seed)["digit_correct"]
                        / score(method, seed)["digit_total"]
                        for seed in SEEDS
                    }
                    reference_digits = {
                        seed: score(baseline, seed)["digit_correct"]
                        / score(baseline, seed)["digit_total"]
                        for seed in SEEDS
                    }
                    digit_accuracy, digit_difference, digit_lower, digit_upper = paired_summary(
                        digits, reference_digits
                    )
                    lines.append(
                        f"| {method} | {digit_range} | {length} | {bucket} | {mean:.4f} | "
                        f"{difference:+.4f} [{lower:+.4f}, {upper:+.4f}] | {digit_accuracy:.4f} | "
                        f"{digit_difference:+.4f} [{digit_lower:+.4f}, {digit_upper:+.4f}] |"
                    )

    lines += [
        "",
        f"| Method | Four-digit PPL at 256 | Difference vs {baseline} [95% CI] | Relative difference of mean PPL |",
        "|---|---:|---:|---:|",
    ]

    def perplexities(method: str) -> dict[int, float]:
        values = {}
        for seed in SEEDS:
            buckets = indexed[method, seed]["metrics"]["4_digit_held_out"]["256"].values()
            totals = list(buckets)
            values[seed] = math.exp(
                sum(bucket["nll_sum"] for bucket in totals)
                / sum(bucket["total"] for bucket in totals)
            )
        return values

    reference = perplexities(baseline)
    for method in METHODS:
        mean, difference, lower, upper = paired_summary(perplexities(method), reference)
        relative = 100.0 * (mean / statistics.mean(reference.values()) - 1.0)
        lines.append(
            f"| {method} | {mean:.4f} | {difference:+.4f} "
            f"[{lower:+.4f}, {upper:+.4f}] | {relative:+.2f}% |"
        )
    return "\n".join(lines)


def tuned_character_table(records: list[dict[str, Any]], tuning: list[dict[str, Any]]) -> str:
    if not records:
        raise ValueError("tuned character records are empty")
    steps = records[0]["steps"]
    selected = select_char_candidates(tuning, list(METHODS), steps)
    indexed = {(record["method"], record["seed"]): record for record in records}
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(indexed) != expected or len(records) != len(expected):
        raise ValueError("tuned character records must contain every method and seed exactly once")
    if any(
        record["candidate"] != selected[record["method"]]
        or record["steps"] != steps
        or record["batch_size"] != records[0]["batch_size"]
        or record["eval_batch_size"] != records[0]["eval_batch_size"]
        or record["dataset_sha256"] != records[0]["dataset_sha256"]
        for record in records
    ):
        raise ValueError(
            "tuned character records do not match selected candidates or configuration"
        )

    def tuning_loss(method: str) -> float:
        return statistics.mean(
            record["tuning_nll"]
            for record in tuning
            if record["method"] == method
            and record["candidate"] == selected[method]
            and record["steps"] == steps
        )

    baseline = min(("rope-full", "rope-linear", "rope-ntk", "alibi"), key=tuning_loss)
    lines = [
        f"Strongest eligible character baseline selected on tuning seeds: `{baseline}`.",
        "",
        f"| Method | Prefix | Late-position NLL | Difference vs {baseline} [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        if method == "abs":
            continue
        for length in PREFIX_LENGTHS:
            values = {
                seed: indexed[method, seed]["metrics"]["context_nll"][str(length)] for seed in SEEDS
            }
            reference = {
                seed: indexed[baseline, seed]["metrics"]["context_nll"][str(length)]
                for seed in SEEDS
            }
            mean, difference, lower, upper = paired_summary(values, reference)
            lines.append(
                f"| {method} | {length} | {mean:.4f} | "
                f"{difference:+.4f} [{lower:+.4f}, {upper:+.4f}] |"
            )
    lines += [
        "",
        "| Method | PPL at 256 | "
        f"Difference vs {baseline} [95% CI] | Relative difference of mean PPL |",
        "|---|---:|---:|---:|",
    ]
    reference = {seed: indexed[baseline, seed]["metrics"]["perplexity"]["256"] for seed in SEEDS}
    for method in METHODS:
        values = {seed: indexed[method, seed]["metrics"]["perplexity"]["256"] for seed in SEEDS}
        mean, difference, lower, upper = paired_summary(values, reference)
        relative = 100.0 * (mean / statistics.mean(reference.values()) - 1.0)
        lines.append(
            f"| {method} | {mean:.4f} | {difference:+.4f} "
            f"[{lower:+.4f}, {upper:+.4f}] | {relative:+.2f}% |"
        )
    return "\n".join(lines)


def needle_table(
    records: list[dict[str, Any]], tuning: list[dict[str, Any]], sanity: list[dict[str, Any]]
) -> str:
    if not sanity_passed(sanity):
        raise ValueError("retrieval sanity gate failed; no retrieval comparison is valid")
    if not records:
        raise ValueError("retrieval records are empty")
    steps = records[0]["steps"]
    selected = select_needle_candidates(tuning, list(METHODS), steps)
    indexed = {(record["method"], record["seed"]): record for record in records}
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(indexed) != expected or len(records) != len(expected):
        raise ValueError("retrieval records must contain every method and seed exactly once")
    if any(
        record["candidate"] != selected[record["method"]]
        or record["steps"] != steps
        or record["batch_size"] != records[0]["batch_size"]
        or record["eval_queries"] != records[0]["eval_queries"]
        for record in records
    ):
        raise ValueError("retrieval records do not match selected candidates or configuration")

    def tuning_accuracy(method: str) -> float:
        return statistics.mean(
            record["metrics"]["1024"]["accuracy"]
            for record in tuning
            if record["method"] == method
            and record["candidate"] == selected[method]
            and record["steps"] == steps
        )

    baseline = max(("rope-full", "rope-linear", "rope-ntk", "alibi"), key=tuning_accuracy)
    lines = [
        f"Strongest eligible retrieval baseline selected on tuning seeds: `{baseline}`.",
        "",
        f"| Method | Context | Accuracy | Difference vs {baseline} [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        for length in (256, 512, 1024, 4096):
            if method == "abs" and length > 256:
                continue
            values = {
                seed: indexed[method, seed]["metrics"][str(length)]["accuracy"] for seed in SEEDS
            }
            reference = {
                seed: indexed[baseline, seed]["metrics"][str(length)]["accuracy"] for seed in SEEDS
            }
            mean, difference, lower, upper = paired_summary(values, reference)
            lines.append(
                f"| {method} | {length} | {mean:.4f} | "
                f"{difference:+.4f} [{lower:+.4f}, {upper:+.4f}] |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--successor-records", type=Path)
    parser.add_argument("--successor-tuning", type=Path)
    parser.add_argument("--char-baselines", type=Path)
    parser.add_argument("--char-tuning", type=Path)
    parser.add_argument("--needle-records", type=Path)
    parser.add_argument("--needle-tuning", type=Path)
    parser.add_argument("--needle-sanity", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line]
    print(character_table(records))
    if bool(args.successor_records) != bool(args.successor_tuning):
        parser.error("successor records and tuning records must be supplied together")
    if args.successor_records is not None and args.successor_tuning is not None:
        successor = [
            json.loads(line) for line in args.successor_records.read_text().splitlines() if line
        ]
        tuning = [
            json.loads(line) for line in args.successor_tuning.read_text().splitlines() if line
        ]
        print("\n" + successor_table(successor, tuning))
    if bool(args.char_baselines) != bool(args.char_tuning):
        parser.error("character baseline and tuning records must be supplied together")
    if args.char_baselines is not None and args.char_tuning is not None:
        baselines = [
            json.loads(line) for line in args.char_baselines.read_text().splitlines() if line
        ]
        tuning = [json.loads(line) for line in args.char_tuning.read_text().splitlines() if line]
        print("\n" + tuned_character_table(baselines, tuning))
    needle_paths = (args.needle_records, args.needle_tuning, args.needle_sanity)
    if any(needle_paths) and not all(needle_paths):
        parser.error("needle records, tuning records, and sanity records must be supplied together")
    if all(path is not None for path in needle_paths):
        needle, tuning, sanity = [
            [json.loads(line) for line in path.read_text().splitlines() if line]
            for path in needle_paths
            if path is not None
        ]
        print("\n" + needle_table(needle, tuning, sanity))


if __name__ == "__main__":
    main()
