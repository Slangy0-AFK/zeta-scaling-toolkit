"""Stage A validation for a proposed independent retrieval experiment.

The symbolic controls validate the dataset; they are not neural-model results.
No Zeta persistent retrieval adapter exists, so comparative stages are blocked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = 1
VALIDATION_SEED = 9001
VALIDATION_LENGTHS = (20, 50)
WINDOW_TURNS = 8
MISSING = "NOT_FOUND"
ROOT = Path(__file__).resolve().parents[2]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def generate_conversation(length: int, seed: int, split: str) -> list[dict[str, Any]]:
    """Produce fact/update turns followed by five balanced queries, an update query and a miss."""
    if length < 20 or seed < 0 or not split:
        raise ValueError("length must be at least 20; seed and split must be valid")
    material = f"independent-retrieval-v{VERSION}:{split}:{seed}:{length}"
    rng = random.Random(int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big"))
    count = length - 8
    ids = rng.sample(range(1 << 40), 2 * count + 2)
    keys = [f"key_{value:010x}" for value in ids[:count]]
    values = [f"value_{value:010x}" for value in ids[count : 2 * count]]
    update_value = f"value_{ids[-2]:010x}"
    missing_key = f"key_{ids[-1]:010x}"
    positions = [round(index * (count - 1) / 4) for index in range(5)]
    if len(set(positions)) != 5:
        raise ValueError("insufficient distinct target positions")
    update_index = next(index for index in range(count) if index not in positions)
    turns: list[dict[str, Any]] = [
        {
            "kind": "fact",
            "key": key,
            "value": value,
            "filler": f"note_{rng.getrandbits(32):08x}",
        }
        for key, value in zip(keys, values)
    ]
    turns.append(
        {
            "kind": "fact",
            "key": keys[update_index],
            "value": update_value,
            "filler": f"note_{rng.getrandbits(32):08x}",
        }
    )
    query_keys = [keys[index] for index in positions] + [keys[update_index], missing_key]
    rng.shuffle(query_keys)
    turns.extend({"kind": "query", "key": key} for key in query_keys)
    return turns


def oracle(turns: list[dict[str, Any]], query_index: int) -> str:
    if turns[query_index]["kind"] != "query":
        raise ValueError("query_index must identify a query")
    key = turns[query_index]["key"]
    for turn in reversed(turns[:query_index]):
        if turn["kind"] == "fact" and turn["key"] == key:
            return turn["value"]
    return MISSING


def serialize(turn: dict[str, Any]) -> str:
    if turn["kind"] == "fact":
        return f"FACT {turn['key']} {turn['value']} {turn['filler']}"
    if turn["kind"] == "query":
        return f"QUERY {turn['key']}"
    raise ValueError("unknown turn kind")


def symbolic_lookup(history: list[dict[str, Any]], key: str) -> str:
    latest = {turn["key"]: turn["value"] for turn in history if turn["kind"] == "fact"}
    return latest.get(key, MISSING)


def evaluate(turns: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rows = []
    for index, turn in enumerate(turns):
        if turn["kind"] != "query":
            continue
        history = turns[:index]
        target = oracle(turns, index)
        candidates = sorted({item["value"] for item in history if item["kind"] == "fact"})
        rng = random.Random(int(digest([seed, index, "random-control"])[:16], 16))
        outputs = {
            "full_history_symbolic": symbolic_lookup(history, turn["key"]),
            "fixed_window_symbolic": symbolic_lookup(history[-WINDOW_TURNS:], turn["key"]),
            "random_control": rng.choice(candidates),
            "no_retrieval": MISSING,
        }
        target_indices = [
            pos
            for pos, fact in enumerate(history)
            if fact["kind"] == "fact" and fact["key"] == turn["key"]
        ]
        latest_position = target_indices[-1] if target_indices else None
        for method, prediction in outputs.items():
            supplied = (
                history
                if method in ("full_history_symbolic", "random_control")
                else history[-WINDOW_TURNS:]
                if method == "fixed_window_symbolic"
                else []
            )
            input_tokens = sum(len(serialize(item).split()) for item in [*supplied, turn])
            rows.append(
                {
                    "system": method,
                    "conversation_length": len(turns),
                    "seed": seed,
                    "query_index": index,
                    "target_position": latest_position,
                    "distance": index - latest_position if latest_position is not None else None,
                    "negative": target == MISSING,
                    "target": target,
                    "prediction": prediction,
                    "correct": prediction == target,
                    "symbolic_input_tokens": input_tokens,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for method in sorted({row["system"] for row in rows}):
        group = [row for row in rows if row["system"] == method]
        positives = [row for row in group if not row["negative"]]
        negatives = [row for row in group if row["negative"]]
        correct = sum(row["correct"] for row in group)
        result[method] = {
            "correct": correct,
            "total": len(group),
            "accuracy": correct / len(group),
            "positive_correct": sum(row["correct"] for row in positives),
            "positive_total": len(positives),
            "negative_correct": sum(row["correct"] for row in negatives),
            "negative_total": len(negatives),
            "symbolic_input_tokens": sum(row["symbolic_input_tokens"] for row in group),
        }
    return result


def paired_differences(rows: list[dict[str, Any]], left: str, right: str) -> list[int]:
    groups: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {left: {}, right: {}}
    for row in rows:
        if row["system"] in groups:
            key = (row["seed"], row["conversation_length"], row["query_index"])
            if key in groups[row["system"]]:
                raise ValueError("duplicate paired observation")
            groups[row["system"]][key] = row
    if not groups[left] or groups[left].keys() != groups[right].keys():
        raise ValueError("paired systems must have the same nonempty query set")
    if any(groups[left][key]["target"] != groups[right][key]["target"] for key in groups[left]):
        raise ValueError("paired systems must use identical targets")
    return [
        int(groups[left][key]["correct"]) - int(groups[right][key]["correct"])
        for key in sorted(groups[left])
    ]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE
    ).strip()


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def run_validation(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    _write_json(output / "status.json", {"stage": "A", "status": "incomplete", "started": start})
    try:
        began = time.perf_counter()
        cpu_start = time.process_time()
        datasets = [
            {
                "length": length,
                "seed": VALIDATION_SEED,
                "split": "validation",
                "turns": generate_conversation(length, VALIDATION_SEED, "validation"),
            }
            for length in VALIDATION_LENGTHS
        ]
        rows = [row for dataset in datasets for row in evaluate(dataset["turns"], dataset["seed"])]
        summary = summarize(rows)
        if summary["full_history_symbolic"]["correct"] != summary["full_history_symbolic"]["total"]:
            raise RuntimeError("symbolic full-history control disagrees with independent oracle")
        paired = paired_differences(rows, "fixed_window_symbolic", "full_history_symbolic")
        elapsed = time.perf_counter() - began
        cpu = time.process_time() - cpu_start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        results = {
            "stage": "A",
            "status": "completed_validation_only",
            "scientific_comparison": False,
            "zeta_system": "unavailable: no integrated query-dependent persistent-context implementation",
            "hypotheses": {key: "inconclusive" for key in ("H1", "H2", "H3")},
            "datasets": len(datasets),
            "queries": len(rows) // 4,
            "validation_seed": VALIDATION_SEED,
            "lengths": list(VALIDATION_LENGTHS),
            "systems": summary,
            "paired_window_minus_full_correct_count": sum(paired),
            "confidence_interval": None,
            "confidence_interval_reason": "one validation seed; no inferential reporting",
            "elapsed_seconds": elapsed,
            "process_cpu_seconds": cpu,
            "process_peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
            "rows": rows,
        }
        _write_json(output / "dataset.json", datasets)
        _write_json(output / "RESULTS.json", results)
        with (output / "PER_SEED_RESULTS.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=["seed", "system", "correct", "total", "accuracy"]
            )
            writer.writeheader()
            for method, metrics in summary.items():
                writer.writerow(
                    {
                        "seed": VALIDATION_SEED,
                        "system": method,
                        **{key: metrics[key] for key in ("correct", "total", "accuracy")},
                    }
                )
        with (output / "EFFICIENCY_COMPARISON.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["system", "symbolic_input_tokens"])
            writer.writeheader()
            for method, metrics in summary.items():
                writer.writerow(
                    {"system": method, "symbolic_input_tokens": metrics["symbolic_input_tokens"]}
                )
        artifacts = (
            "dataset.json",
            "RESULTS.json",
            "PER_SEED_RESULTS.csv",
            "EFFICIENCY_COMPARISON.csv",
        )
        manifest = {
            "stage": "A",
            "git_commit": _git("rev-parse", "HEAD"),
            "git_status_porcelain": _git("status", "--porcelain"),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "pytorch": importlib.metadata.version("torch"),
            "platform": platform.platform(),
            "cpu_logical": os.cpu_count(),
            "cpu_affinity": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "seed": VALIDATION_SEED,
            "split": "validation",
            "lengths": list(VALIDATION_LENGTHS),
            "window_turns": WINDOW_TURNS,
            "dataset_fingerprint": digest(datasets),
            "started": start,
            "completed": datetime.now(timezone.utc).isoformat(),
            "output_sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in artifacts
            },
        }
        _write_json(output / "manifest.json", manifest)
        _write_json(
            output / "status.json",
            {
                "stage": "A",
                "status": "completed_validation_only",
                "started": start,
                "completed": manifest["completed"],
            },
        )
        print(
            json.dumps(
                {"output": str(output), "status": results["status"], "queries": results["queries"]}
            )
        )
    except BaseException:
        _write_json(
            output / "status.json", {"stage": "A", "status": "incomplete", "started": start}
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_validation(args.output_dir)


if __name__ == "__main__":
    main()
