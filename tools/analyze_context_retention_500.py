"""Analyze completed hosted 500-step context-ability records without training."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.context_successor import METHODS, select_candidates  # noqa: E402
from benchmarks.length_generalization import bootstrap_interval  # noqa: E402

SEEDS = tuple(range(42, 52))
TUNING_SEEDS = tuple(range(100, 110))
LENGTHS = (256, 512, 1024, 4096)
TASKS = (
    ("3_digit_train_range_cyclic", "Seen 3-digit training-range"),
    ("4_digit_held_out", "Unseen 4-digit held-out"),
)
BASELINES = ("rope-full", "rope-linear", "rope-ntk", "alibi")
TARGETS = ("rope-log-full", "rope-log-window")
COLORS = {
    "abs": "#777777",
    "rope-full": "#2878b5",
    "rope-linear": "#62a744",
    "rope-ntk": "#8755a1",
    "alibi": "#222222",
    "rope-log-full": "#d34a44",
    "rope-window": "#35a878",
    "rope-log-window": "#ec8c2f",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_records(
    records: list[dict[str, Any]], tuning: list[dict[str, Any]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, int]]:
    keys = [(record["method"], record["seed"]) for record in records]
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if len(records) != len(expected) or set(keys) != expected:
        raise ValueError("report records must contain each method and seed exactly once")

    tuning_keys = [(record["method"], record["candidate"], record["seed"]) for record in tuning]
    expected_tuning = {
        (method, candidate, seed)
        for method in METHODS
        for candidate in (0, 1)
        for seed in TUNING_SEEDS
    }
    if len(tuning) != len(expected_tuning) or set(tuning_keys) != expected_tuning:
        raise ValueError("tuning records must contain both candidates and all tuning seeds")

    selected = select_candidates(tuning, list(METHODS), 500)
    if any(
        record["steps"] != 500
        or record["train_length"] != 256
        or record["batch_size"] != 16
        or record["samples"] != 8
        or record["tuning"]
        or record["candidate"] != selected[record["method"]]
        for record in records
    ):
        raise ValueError("report records do not match the selected 500-step protocol")

    indexed = {(record["method"], record["seed"]): record for record in records}
    for method in METHODS:
        expected_lengths = {"256"} if method == "abs" else {str(n) for n in LENGTHS}
        for seed in SEEDS:
            record = indexed[method, seed]
            for task, _ in TASKS:
                if set(record["metrics"][task]) != expected_lengths:
                    raise ValueError(f"incomplete contexts for {method}/{seed}/{task}")
    return indexed, selected


def score(
    record: dict[str, Any], task: str, length: int, bucket: str | None = None
) -> tuple[float, float, float, float]:
    buckets = record["metrics"][task][str(length)]
    selected = [buckets[bucket]] if bucket is not None else list(buckets.values())
    total = sum(item["total"] for item in selected)
    correct = sum(item["correct"] for item in selected)
    digit_total = sum(item["digit_total"] for item in selected)
    digit_correct = sum(item["digit_correct"] for item in selected)
    nll_sum = sum(item["nll_sum"] for item in selected)
    mean_nll = nll_sum / total
    return correct / total, digit_correct / digit_total, mean_nll, math.exp(mean_nll)


def average(values: list[float]) -> float:
    return statistics.mean(values)


def paired_text(values: list[float]) -> str:
    lower, upper = bootstrap_interval(values)
    return f"{average(values):+.4f} [{lower:+.4f}, {upper:+.4f}]"


def candidate_settings_table(tuning: list[dict[str, Any]]) -> str:
    lines = ["| Method | Candidate 0 settings | Candidate 1 settings |", "|---|---|---|"]
    for method in METHODS:
        settings_by_candidate = []
        for candidate in (0, 1):
            settings = {
                json.dumps(record["settings"], sort_keys=True, separators=(",", ":"))
                for record in tuning
                if record["method"] == method and record["candidate"] == candidate
            }
            if len(settings) != 1:
                raise ValueError(f"inconsistent settings for {method} candidate {candidate}")
            settings_by_candidate.append(f"`{settings.pop()}`")
        lines.append(f"| {method} | {settings_by_candidate[0]} | {settings_by_candidate[1]} |")
    return "\n".join(lines)


def build_csv(indexed: dict[tuple[str, int], dict[str, Any]], path: Path) -> None:
    rows = []
    for task, task_label in TASKS:
        for method in METHODS:
            for length in LENGTHS:
                if str(length) not in indexed[method, SEEDS[0]]["metrics"][task]:
                    continue
                first = indexed[method, SEEDS[0]]["metrics"][task][str(length)]
                buckets: list[str | None] = [
                    None,
                    *sorted(first, key=lambda x: int(x.split("_")[0])),
                ]
                for bucket in buckets:
                    values = [score(indexed[method, seed], task, length, bucket) for seed in SEEDS]
                    rows.append(
                        {
                            "task": task,
                            "task_label": task_label,
                            "method": method,
                            "context_tokens": length,
                            "position_bucket": bucket or "overall",
                            "seeds": len(values),
                            "accuracy": average([value[0] for value in values]),
                            "digit_only_accuracy": average([value[1] for value in values]),
                            "mean_nll": average([value[2] for value in values]),
                            "mean_perplexity": average([value[3] for value in values]),
                        }
                    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def context_table(indexed: dict[tuple[str, int], dict[str, Any]], task: str) -> str:
    lines = [
        "| Method | 256 tokens | 512 tokens | 1,024 tokens | 4,096 tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = []
        for length in LENGTHS:
            if str(length) not in indexed[method, SEEDS[0]]["metrics"][task]:
                cells.append("n/a")
                continue
            values = [score(indexed[method, seed], task, length) for seed in SEEDS]
            cells.append(
                f"{fmt(average([value[0] for value in values]))} acc / "
                f"{average([value[3] for value in values]):.2f} PPL"
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def context_change_table(indexed: dict[tuple[str, int], dict[str, Any]], task: str) -> str:
    lines = [
        "| Method | 512 vs 256 | 1,024 vs 256 | 4,096 vs 256 |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = []
        for length in LENGTHS[1:]:
            if str(length) not in indexed[method, SEEDS[0]]["metrics"][task]:
                cells.append("n/a")
                continue
            changes = [
                (
                    score(indexed[method, seed], task, length)[0]
                    - score(indexed[method, seed], task, LENGTHS[0])[0],
                    score(indexed[method, seed], task, length)[3]
                    - score(indexed[method, seed], task, LENGTHS[0])[3],
                )
                for seed in SEEDS
            ]
            cells.append(
                f"{average([change[0] for change in changes]):+.4f} acc / "
                f"{average([change[1] for change in changes]):+.2f} PPL"
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def position_table(indexed: dict[tuple[str, int], dict[str, Any]], task: str) -> str:
    lines = [
        "| Method | 0–63 (early) | 64–255 | 256–1,023 (middle) | 1,024–4,095 (late) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        if "4096" not in indexed[method, SEEDS[0]]["metrics"][task]:
            continue
        buckets = indexed[method, SEEDS[0]]["metrics"][task]["4096"]
        cells = []
        for bucket in ("0_63", "64_255", "256_1023", "1024_4095"):
            if bucket not in buckets:
                cells.append("n/a")
                continue
            values = [score(indexed[method, seed], task, 4096, bucket) for seed in SEEDS]
            cells.append(
                f"{average([value[0] for value in values]):.4f} acc / "
                f"{average([value[1] for value in values]):.4f} digit"
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison_table(indexed: dict[tuple[str, int], dict[str, Any]]) -> str:
    lines = [
        "| Method | Baseline | Mean accuracy (late) | Paired accuracy diff [95% CI] | "
        "Seeds favoring method | Digit-only diff [95% CI] | Mean PPL (late) | "
        "Paired PPL diff [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    task, _ = TASKS[1]
    for method in TARGETS:
        for baseline in BASELINES:
            target_scores = [
                score(indexed[method, seed], task, 4096, "1024_4095") for seed in SEEDS
            ]
            baseline_scores = [
                score(indexed[baseline, seed], task, 4096, "1024_4095") for seed in SEEDS
            ]
            accuracy_diff = [
                target[0] - reference[0]
                for target, reference in zip(target_scores, baseline_scores)
            ]
            digit_diff = [
                target[1] - reference[1]
                for target, reference in zip(target_scores, baseline_scores)
            ]
            ppl_diff = [
                target[3] - reference[3]
                for target, reference in zip(target_scores, baseline_scores)
            ]
            lines.append(
                f"| {method} | {baseline} | {average([value[0] for value in target_scores]):.4f} | "
                f"{paired_text(accuracy_diff)} | {sum(value > 0 for value in accuracy_diff)}/10 | "
                f"{paired_text(digit_diff)} | {average([value[3] for value in target_scores]):.2f} | "
                f"{paired_text(ppl_diff)} |"
            )
    return "\n".join(lines)


def deterioration_table(indexed: dict[tuple[str, int], dict[str, Any]]) -> str:
    lines = [
        "| Method | Late (1,024–4,095) minus 64–255 accuracy | "
        "Paired 95% CI | Seeds with lower late accuracy |",
        "|---|---:|---:|---:|",
    ]
    task, _ = TASKS[1]
    for method in TARGETS:
        changes = [
            score(indexed[method, seed], task, 4096, "1024_4095")[0]
            - score(indexed[method, seed], task, 4096, "64_255")[0]
            for seed in SEEDS
        ]
        lines.append(
            f"| {method} | {average(changes):+.4f} | "
            f"[{bootstrap_interval(changes)[0]:+.4f}, {bootstrap_interval(changes)[1]:+.4f}] | "
            f"{sum(change < 0 for change in changes)}/10 |"
        )
    return "\n".join(lines)


def hardware_summary(records: list[dict[str, Any]], hardware_path: Path) -> tuple[str, str]:
    platforms = Counter(record["environment"]["platform"] for record in records)
    hardware = json.loads(hardware_path.read_text())
    models_by_method: dict[str, Counter[str]] = {}
    models_by_key: dict[tuple[str, int], str] = {}
    for method in METHODS:
        models: Counter[str] = Counter()
        for seed in SEEDS:
            details = hardware.get(f"successor-report-{method}-{seed}", {})
            model = details.get("Model name", "not recorded")
            models[model] += 1
            models_by_key[method, seed] = model
        models_by_method[method] = models

    platform_lines = [
        f"- `{platform}`: {count} records" for platform, count in sorted(platforms.items())
    ]
    cpu_lines = [
        f"- {method}: " + "; ".join(f"{count} × {model}" for model, count in models.items())
        for method, models in models_by_method.items()
    ]
    for target in TARGETS:
        same = sum(models_by_key[target, seed] == models_by_key["alibi", seed] for seed in SEEDS)
        cpu_lines.append(f"- Same-CPU seed matches vs ALiBi: {target} {same}/10")
    return "\n".join(platform_lines), "\n".join(cpu_lines)


def char_summary(records: list[dict[str, Any]]) -> str:
    expected = {
        (variant, seed)
        for variant in ("rope-full", "rope-window", "rope-log-full", "rope-log-window")
        for seed in SEEDS
    }
    keys = [(record["variant"], record["seed"]) for record in records]
    if len(records) != 40 or set(keys) != expected:
        raise ValueError("character records must contain four variants and ten seeds each")
    lines = [
        "| Training variant | Attention evaluation | 256 PPL | 512 PPL | 1,024 PPL | 4,096 PPL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in ("rope-full", "rope-window", "rope-log-full", "rope-log-window"):
        for mode in ("full", "window"):
            cells = [
                statistics.mean(
                    next(
                        record
                        for record in records
                        if record["variant"] == variant and record["seed"] == seed
                    )["metrics"][mode]["perplexity"][str(length)]
                    for seed in SEEDS
                )
                for length in LENGTHS
            ]
            lines.append(
                f"| {variant} | {mode} | " + " | ".join(f"{cell:.4f}" for cell in cells) + " |"
            )
    return "\n".join(lines)


def svg_curve(indexed: dict[tuple[str, int], dict[str, Any]], path: Path) -> None:
    width, height = 1000, 720
    left, right = 95, 960
    panel_top = (85, 390)
    panel_height = 230
    x_values = {length: left + (math.log2(length) - 8) / 4 * (right - left) for length in LENGTHS}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:22px;font-weight:bold}"
        ".axis{font-size:13px}.legend{font-size:12px}</style>",
        '<text x="50" y="34" class="title">500-step successor accuracy by context length</text>',
    ]
    for panel_index, (task, title) in enumerate(TASKS):
        top = panel_top[panel_index]
        bottom = top + panel_height
        parts.append(f'<text x="{left}" y="{top - 22}" class="axis">{html.escape(title)}</text>')
        for tick in (0, 0.25, 0.5, 0.75, 1):
            y = bottom - tick * panel_height
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                f'stroke="#dddddd"/><text x="{left - 12}" y="{y + 4:.1f}" '
                f'text-anchor="end" class="axis">{tick:.2f}</text>'
            )
        for length, x in x_values.items():
            parts.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
                f'stroke="#eeeeee"/><text x="{x:.1f}" y="{bottom + 20}" '
                f'text-anchor="middle" class="axis">{length}</text>'
            )
        for method in METHODS:
            points = []
            for length in LENGTHS:
                if str(length) not in indexed[method, SEEDS[0]]["metrics"][task]:
                    continue
                scores = [score(indexed[method, seed], task, length)[0] for seed in SEEDS]
                x = x_values[length]
                y = bottom - average(scores) * panel_height
                points.append((x, y))
            if not points:
                continue
            color = COLORS[method]
            point_list = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(
                f'<polyline points="{point_list}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
            parts.extend(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>' for x, y in points
            )
        if panel_index == 1:
            parts.append(
                f'<text x="{(left + right) / 2:.1f}" y="{bottom + 42}" '
                'text-anchor="middle" class="axis">Evaluation context tokens (log scale)</text>'
            )
    legend_y = 686
    for index, method in enumerate(METHODS):
        x = 95 + (index % 4) * 220
        y = legend_y + (index // 4) * 17
        parts.append(
            f'<line x1="{x}" y1="{y - 4}" x2="{x + 25}" y2="{y - 4}" '
            f'stroke="{COLORS[method]}" stroke-width="3"/>'
            f'<text x="{x + 32}" y="{y}" class="legend">{html.escape(method)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_report(
    records: list[dict[str, Any]],
    indexed: dict[tuple[str, int], dict[str, Any]],
    selected: dict[str, int],
    tuning: list[dict[str, Any]],
    char_records: list[dict[str, Any]],
    hardware_path: Path,
    csv_path: Path,
    svg_path: Path,
) -> str:
    platforms, cpus = hardware_summary(records, hardware_path)
    candidates = "\n".join(f"| {method} | {selected[method]} |" for method in METHODS)
    tables = "\n\n".join(f"### {label}\n\n{context_table(indexed, task)}" for task, label in TASKS)
    changes = "\n\n".join(
        f"### {label}\n\n{context_change_table(indexed, task)}" for task, label in TASKS
    )
    positions = "\n\n".join(
        f"### 4,096-token positions — {label}\n\n{position_table(indexed, task)}"
        for task, label in TASKS
    )
    config = records[0]
    return f"""# Completed 500-step context-retention analysis

**Assessment: NO-GO for claiming useful distant-context retrieval.** The
completed benchmark shows substantial task- and position-dependent differences,
but its next-successor task does not establish that correct predictions depend
on retrieving information from far back in the context. The separate planted
needle sanity check failed the pre-existing gate, so no successful retrieval
claim is supported.

## Experiment identity and integrity

- Decision-grade cohort: hosted **Stage 1 context ability 500** workflow,
  [run 43780419668](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions/runs/43780419668),
  commit [`d58685833085b45c41d78f10fa3e3d6150e1a773`](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/commit/d58685833085b45c41d78f10fa3e3d6150e1a773).
- Reporting data: [successor records](./context-ability-successor500-hosted.jsonl),
  {len(records)} rows, {len(METHODS)} methods × 10 reporting seeds (42–51),
  no missing or duplicate method/seed keys.
- Tuning data: [successor tuning records](./context-ability-successor-tuning500-hosted.jsonl),
  {len(tuning)} rows, 8 methods × 2 candidates × 10 separate tuning seeds (100–109).
- Protocol: {config["steps"]} optimization steps, train length
  {config["train_length"]}, batch size {config["batch_size"]}, {config["samples"]}
  evaluation samples per seed; Python 3.14.10, PyTorch 2.15.0+cpu, one thread.
- Reporting candidates selected on tuning seeds:

| Method | Candidate index |
|---|---:|
{candidates}

The two tuning candidates per method, with settings copied from the raw tuning
records:

{candidate_settings_table(tuning)}

- Two successor evaluation partitions are kept distinct. `3_digit_train_range_cyclic`
  is numbers 100–999 repeated twice and is seen/training-range. `4_digit_held_out`
  is numbers 1,000–2,999, held out from training but reused across tuning and
  reporting; disjoint seeds do not make the number-range samples independent.
- The records contain no per-record immutable experiment manifest or source
  digest. The linked Actions run and commit identify the workflow/code version;
  source/environment provenance is therefore less complete than a manifest-backed
  run. [Hardware metadata](./context-ability-hosted-hardware.json) shows multiple
  CPU models; the paired intervals below pair by seed, not by identical CPU.

Hosted report platform strings:
{platforms}

Recorded successor-report CPU models:
{cpus}

The local checkpointed [`context-500-*` records](./context-extrapolation-resume/)
are a separate protocol (protocol 1, Python 3.14.2 / PyTorch 2.15.0+cu132,
one Azure host, context-axis train 100–699 versus unseen 700–999). Its reporting
file contains only ALiBi and `rope-window` for seeds 42–51; it does not contain
the requested log-RoPE comparisons. Its 480 checkpoint files and source/checkpoint
hashes are not mixed into these hosted Stage 1 measurements.

## Context-retention table

Each cell is mean per-seed **accuracy / perplexity** over all scored positions.
Perplexity is `exp(mean token NLL)` calculated per seed, then averaged across the
10 seeds. These are next-token/successor scores, not retrieval scores. `abs`
cannot be evaluated beyond its 256-token position table.

{tables}

## Change as context increases

The following are paired per-seed changes from 256 tokens; accuracy changes
are absolute fractions and PPL changes are target minus 256-token PPL.

{changes}

The complete position-bucket measurements for every context length, including
NLL and digit-only accuracy, are in [{csv_path.name}]({csv_path.name}).

## Context-retention curve

![Mean successor accuracy by context length](./{svg_path.name})

The upper panel is the seen cyclic stream; the lower is the held-out four-digit
stream. The x-axis is logarithmic. The curve includes accuracy over separators,
so digit-only accuracy is also provided in the CSV.

## Position dependence at 4,096 tokens

Each value is mean seed-level total-token accuracy / digit-only accuracy. The
position buckets are the benchmark's fixed token ranges; at a 512-token context,
the `256–1,023` bucket represents only positions 256–511.

{positions}

## Paired late-position comparison against baselines

For four-digit held-out data at positions 1,024–4,095, differences are paired
by reporting seed. CIs are 10,000 seed-bootstrap resamples; `wins` counts seeds
where the log-RoPE method's total-token accuracy is higher. Perplexity
differences are target minus baseline, so positive means worse.

{comparison_table(indexed)}

Position change within the 4,096-token held-out sequence:

{deterioration_table(indexed)}

At this context the log-full method drops materially in the late bucket
relative to positions 64–255. Log-window has little late-vs-64–255 accuracy
drop, but remains below ALiBi and the absolute scores do not prove distant
retrieval. No post-hoc numerical “collapse” threshold is imposed here.

## Held-out character control

The separate Tiny Shakespeare character control evaluates the held-out text
split. Values below are mean per-seed perplexity, with training variant
separated from full/window attention evaluation:

{char_summary(char_records)}

The source records use a 256-token training context; worsening full-attention
perplexity at longer lengths and the flatter window-evaluation scores describe
position/context degradation, not evidence that a specific distant fact was
retrieved.

## Retrieval gate and interpretation

The workflow's planted-needle sanity records are
[here](./context-ability-needle-sanity-hosted.jsonl). Only 6/10 seeds met its
0.9 accuracy threshold, below the required 8/10, so retrieval comparisons were
skipped. Moreover, that sanity task has one planted key/value pair and does
not require matching the query key, as documented in [RESULTS.md](./RESULTS.md).
The successor task asks for the next item in a deterministic number stream and
can be solved from local sequence structure; high scores on the seen cyclic
stream do not establish use of information far back in the 4,096-token input.

**Direct answer:** No. `rope-log-window` scored 0.4418 at late held-out positions
(digit-only 0.3629) versus ALiBi 0.5214 (digit-only 0.4018); paired difference
−0.0796, 95% CI [−0.1053, −0.0540], and it beat ALiBi on only 1/10 reporting
seeds. `rope-log-full` scored 0.2285 (digit-only 0.1477), difference −0.2930,
95% CI [−0.3210, −0.2624], beating ALiBi on 0/10 seeds. The log-window seen
cyclic score (0.9530 late) is encouraging for in-range next-token behavior but
is not held-out retrieval evidence. Overall: **NO-GO** for the claim that Zeta
preserves useful information over 4,096 tokens; evidence supports a clear
held-out late-position deficit for both log methods versus the selected
baseline, with stronger deterioration for log-full.

## Next scientifically justified experiment

Do not treat the current failed sanity run as evidence of retrieval. Before
another comparative training study, define a query-dependent associative
retrieval sanity task where the target value can only be identified by matching
the query key to its distant key/value pair; include distractors and balance
target position/distance. First establish the existing sanity gate on the
existing model/training protocol. Only if it passes should an independently
held-out retrieval comparison be considered. Keep tuning and reporting
contents disjoint, and record a manifest/source digest with each run. This is
a recommendation only; no training was started for this analysis.

## Reproduction

All commands below are analysis/reporting only; they do not train models.

```bash
python tools/analyze_context_retention_500.py
python benchmarks/context_report.py benchmarks/context-ability-char500-hosted.jsonl \\
  --successor-records benchmarks/context-ability-successor500-hosted.jsonl \\
  --successor-tuning benchmarks/context-ability-successor-tuning500-hosted.jsonl \\
  --char-baselines benchmarks/context-ability-char-baselines500-hosted.jsonl \\
  --char-tuning benchmarks/context-ability-char-tuning500-hosted.jsonl
```

The analyzer writes this Markdown report, its long-form CSV, and the SVG curve.
It validates cohort completeness and candidate selection before producing
output. Original JSONL records, checkpoint archives, and [RESULTS.md](./RESULTS.md)
were not modified. No experiment, 2,000-step run, release, or publication was
initiated.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "benchmarks/context-ability-successor500-hosted.jsonl",
    )
    parser.add_argument(
        "--tuning",
        type=Path,
        default=ROOT / "benchmarks/context-ability-successor-tuning500-hosted.jsonl",
    )
    parser.add_argument(
        "--char-records",
        type=Path,
        default=ROOT / "benchmarks/context-ability-char500-hosted.jsonl",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=ROOT / "benchmarks/context-ability-hosted-hardware.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "benchmarks/context-retention-500-analysis",
    )
    args = parser.parse_args()

    records = read_jsonl(args.records)
    tuning = read_jsonl(args.tuning)
    char_records = read_jsonl(args.char_records)
    indexed, selected = validate_records(records, tuning)
    csv_path = args.output_prefix.with_suffix(".csv")
    svg_path = args.output_prefix.with_suffix(".svg")
    report_path = args.output_prefix.with_suffix(".md")
    build_csv(indexed, csv_path)
    svg_curve(indexed, svg_path)
    report = build_report(
        records, indexed, selected, tuning, char_records, args.hardware, csv_path, svg_path
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}, {csv_path}, and {svg_path}")


if __name__ == "__main__":
    main()
