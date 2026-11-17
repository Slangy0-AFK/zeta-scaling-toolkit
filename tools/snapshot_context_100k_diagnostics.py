"""Snapshot existing 100K diagnostics without touching the running experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> tuple[list[dict], str, bool]:
    data = path.read_bytes()
    incomplete = bool(data) and not data.endswith(b"\n")
    lines = data.splitlines()
    if incomplete:
        lines = lines[:-1]
    return (
        [json.loads(line) for line in lines if line.strip()],
        hashlib.sha256(data).hexdigest(),
        incomplete,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    output = args.output_dir.resolve()
    if output.is_relative_to(experiment):
        parser.error("output must be outside the active experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    outputs = [output / "snapshot.json", output / "checkpoints.csv"]
    if any(path.exists() for path in outputs):
        parser.error("snapshot outputs already exist; choose a fresh directory")

    progress, progress_hash, progress_partial = read_jsonl(experiment / "progress.jsonl")
    health, health_hash, health_partial = read_jsonl(args.health)
    if not progress or not health:
        raise ValueError("progress and health logs must contain complete records")
    checks = {
        check["processed_tokens"]: check for event in health for check in event["new_checkpoints"]
    }
    correctness_path = experiment / "CORRECTNESS_RESULTS.json"
    correctness = json.loads(correctness_path.read_text())
    paths = [
        experiment / "run_experiment.py",
        experiment / "compat_adapter.py",
        experiment / "requirements-lock.txt",
        correctness_path,
        ROOT / "tools/monitor_context_100k.py",
        Path(__file__),
    ]
    manifest = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_snapshot": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preflight_source_commit": correctness["source_commit"],
        "progress_file": str(experiment / "progress.jsonl"),
        "progress_sha256_at_read": progress_hash,
        "progress_had_incomplete_trailing_line": progress_partial,
        "health_file": str(args.health.resolve()),
        "health_sha256_at_read": health_hash,
        "health_had_incomplete_trailing_line": health_partial,
        "file_sha256": {str(path): sha256(path) for path in paths},
        "model_id": correctness["model_id"],
        "model_revision": correctness["model_revision"],
        "weights_sha256_recorded": correctness["weights_sha256"],
        "tokenizer_sha256_recorded": correctness["tokenizer_sha256"],
        "stream_sha256_recorded": correctness["stream_sha256"],
        "corpus_sha256_recorded": correctness["corpus_sha256"],
        "pid": health[-1]["pid"],
        "cpu_count_visible": os.cpu_count(),
        "limitations": [
            "Progress and health were read separately and may represent different instants.",
            "Recorded model hashes are preflight provenance, not newly computed hashes.",
            "Checkpoint interval rates are not per-chunk rates.",
        ],
    }
    snapshot = {"manifest": manifest, "progress": progress, "health": health}
    fields = [
        "processed_tokens",
        "last_position",
        "elapsed_seconds",
        "tokens_per_second",
        "interval_seconds",
        "interval_tokens_per_second",
        "peak_rss_mib",
        "available_memory_mib",
        "cache_bytes",
        "checkpoint",
        "checkpoint_sha256",
        "monitor_status",
        "monitor_issues",
    ]
    rows = []
    previous = 0
    for entry in progress:
        count = entry["processed_tokens"]
        seconds = entry.get("interval_seconds")
        check = checks.get(count)
        row = {name: entry.get(name) for name in fields}
        row["interval_tokens_per_second"] = (
            (count - previous) / seconds if seconds and count > previous else None
        )
        row["checkpoint_sha256"] = check["checkpoint_sha256"] if check else None
        row["monitor_status"] = ("FAIL" if check["issues"] else "PASS") if check else "NOT TESTED"
        row["monitor_issues"] = "; ".join(check["issues"]) if check else ""
        rows.append(row)
        previous = count
    with outputs[0].open("x", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2, sort_keys=True)
        file.write("\n")
    with outputs[1].open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Snapshot written to {output}")


if __name__ == "__main__":
    main()
