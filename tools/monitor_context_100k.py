"""Read-only checkpoint and process diagnostics for the 100K experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open

STAGES = (8193, 10000, 16384, 25000, 50000, 65537, 75000, 90000, 100000)
INTERVAL = 2048
SINKS = 64
WINDOW = 64
CONFIG_SHA256 = "224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee"


def expected_positions(end: int) -> list[int]:
    return list(range(min(end, SINKS))) + list(range(max(SINKS, end - WINDOW), end))


def scheduled_positions(end: int) -> list[int]:
    return sorted(set(range(INTERVAL, end + 1, INTERVAL)) | {x for x in STAGES if x <= end})


def process_identity(pid: int) -> tuple[str, str] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    return fields[0], fields[19]


def read_progress(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    data = path.read_bytes()
    incomplete = bool(data) and not data.endswith(b"\n")
    lines = data.splitlines()
    if incomplete:
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()], incomplete


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Monitor:
    def __init__(self, root: Path, config_path: Path, pid: int, stall_seconds: float):
        self.root = root.resolve()
        self.pid = pid
        self.stall_seconds = stall_seconds
        self.identity = process_identity(pid)
        if self.identity is None or self.identity[0] == "Z":
            raise ValueError(f"experiment process {pid} is not running")
        self.preflight = json.loads((self.root / "CORRECTNESS_RESULTS.json").read_text())
        self.config = json.loads(config_path.read_text())
        if digest(config_path) != CONFIG_SHA256:
            raise ValueError("model configuration hash does not match the pinned experiment")
        self.layers = self.config["num_hidden_layers"]
        self.heads = self.config["num_key_value_heads"]
        self.head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        self.records: list[dict[str, Any]] = []
        self.fingerprints: dict[str, tuple[int, int]] = {}
        self.previous_positions: list[int] | None = None
        self.previous_tensors: dict[str, np.ndarray] = {}
        self.last_checkpoint_time = time.time()
        self.terminal_status: str | None = None

    def check_checkpoint(self, entry: dict[str, Any]) -> dict[str, Any]:
        end = entry["processed_tokens"]
        path = (self.root / entry["checkpoint"]).resolve()
        if not path.is_relative_to(self.root / "checkpoints"):
            raise ValueError("checkpoint path escapes the checkpoint directory")
        issues: list[str] = []
        positions_expected = expected_positions(end)
        current_tensors: dict[str, np.ndarray] = {}
        with safe_open(str(path), framework="numpy", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            for name, expected in (
                ("next_position", str(end)),
                ("stream_sha256", self.preflight["stream_sha256"]),
                ("model_sha256", self.preflight["weights_sha256"]),
                ("model_revision", self.preflight["model_revision"]),
            ):
                if metadata.get(name) != expected:
                    issues.append(f"checkpoint metadata mismatch: {name}")
            keys = set(checkpoint.keys())
            expected_keys = {"positions"} | {
                f"layer_{index}_{kind}"
                for index in range(self.layers)
                for kind in ("keys", "values")
            }
            if keys != expected_keys:
                issues.append("checkpoint layer tensors are missing or unexpected")
            position_tensor = checkpoint.get_tensor("positions")
            positions = position_tensor.tolist()
            if position_tensor.dtype != np.int64:
                issues.append("checkpoint position tensor is not int64")
            if positions != positions_expected:
                issues.append("invalid, missing, duplicated, or misplaced cache positions")
            if entry.get("retained_positions") != positions_expected:
                issues.append("progress cache positions do not match the expected policy")
            expected_shape = (1, self.heads, len(positions_expected), self.head_dim)
            tensor_bytes = 0
            shared = sorted(set(positions) & set(self.previous_positions or []))
            previous_indices = (
                [self.previous_positions.index(position) for position in shared]
                if self.previous_positions is not None
                else []
            )
            current_indices = [positions.index(position) for position in shared]
            for name in sorted(expected_keys - {"positions"}):
                if name not in keys:
                    continue
                tensor = checkpoint.get_tensor(name)
                tensor_bytes += tensor.nbytes
                if tensor.shape != expected_shape:
                    issues.append(f"unexpected tensor shape: {name}")
                if tensor.dtype != np.float32:
                    issues.append(f"unexpected tensor dtype: {name}")
                if not np.isfinite(tensor).all():
                    issues.append(f"nonfinite cache tensor: {name}")
                if (
                    shared
                    and name in self.previous_tensors
                    and tensor.shape == expected_shape
                    and not np.array_equal(
                        tensor[..., current_indices, :],
                        self.previous_tensors[name][..., previous_indices, :],
                    )
                ):
                    issues.append(f"retained cache state changed across checkpoints: {name}")
                current_tensors[name] = np.array(tensor, copy=True)
            expected_bytes = tensor_bytes + len(positions) * 8 * self.layers
            if entry.get("cache_bytes") != expected_bytes:
                issues.append("reported cache bytes do not match checkpoint tensors")
        if entry.get("first_position") != 0 or entry.get("last_position") != end - 1:
            issues.append("progress position bounds are inconsistent")
        if entry.get("stream_sha256") != self.preflight["stream_sha256"]:
            issues.append("progress stream fingerprint mismatch")
        if len(positions) > SINKS + WINDOW:
            issues.append("cache exceeds configured sink and window bounds")
        self.previous_positions = positions
        self.previous_tensors = current_tensors
        stat = path.stat()
        self.fingerprints[str(path)] = (stat.st_size, stat.st_mtime_ns)
        return {
            "processed_tokens": end,
            "last_position": entry.get("last_position"),
            "checkpoint": str(path),
            "checkpoint_sha256": digest(path),
            "elapsed_seconds": entry.get("elapsed_seconds"),
            "tokens_per_second": entry.get("tokens_per_second"),
            "cache_bytes": entry.get("cache_bytes"),
            "retained_positions": len(positions),
            "shared_positions_checked": len(shared),
            "issues": issues,
        }

    def observe(self) -> dict[str, Any]:
        now = time.time()
        issues: list[str] = []
        checks: list[dict[str, Any]] = []
        try:
            records, partial = read_progress(self.root / "progress.jsonl")
            if records[: len(self.records)] != self.records:
                issues.append("previously observed progress was modified or truncated")
            elif len(records) > len(self.records):
                counts = [record["processed_tokens"] for record in records]
                if counts != scheduled_positions(counts[-1]):
                    issues.append(
                        "checkpoint sequence has missing, duplicated, or unexpected positions"
                    )
                for record in records[len(self.records) :]:
                    checks.append(self.check_checkpoint(record))
                self.records = records
                self.last_checkpoint_time = (self.root / "progress.jsonl").stat().st_mtime
        except (OSError, ValueError, KeyError, TypeError, IndexError, SafetensorError) as exc:
            issues.append(f"checkpoint/progress validation failed: {type(exc).__name__}: {exc}")
            partial = False
        for path, fingerprint in self.fingerprints.items():
            try:
                stat = Path(path).stat()
                if (stat.st_size, stat.st_mtime_ns) != fingerprint:
                    issues.append(f"previously validated checkpoint changed: {path}")
            except FileNotFoundError:
                issues.append(f"previously validated checkpoint missing: {path}")
        identity = process_identity(self.pid)
        alive = identity is not None and identity[0] != "Z" and identity[1] == self.identity[1]
        resource_path = self.root / "resource-results.json"
        resource = None
        if resource_path.exists():
            try:
                resource = json.loads(resource_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"cannot read resource results: {exc}")
        if not alive:
            self.terminal_status = (
                resource.get("status", "missing_status") if resource else "missing_status"
            )
            if self.terminal_status != "completed":
                issues.append(
                    f"experiment stopped without successful completion: {self.terminal_status}"
                )
        latest = self.records[-1]["processed_tokens"] if self.records else 0
        age = now - self.last_checkpoint_time
        stalled = alive and latest < 100000 and age > self.stall_seconds
        if stalled:
            issues.append("no checkpoint observed within the configured stall threshold")
        for check in checks:
            issues.extend(
                f"position {check['processed_tokens']}: {issue}" for issue in check["issues"]
            )
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pid": self.pid,
            "process_alive": alive,
            "terminal_status": self.terminal_status,
            "latest_processed_tokens": latest,
            "checkpoint_age_seconds": age,
            "suspected_stall": stalled,
            "incomplete_progress_line": partial,
            "new_checkpoints": checks,
            "issues": issues,
            "preflight_rope": self.preflight.get("high_position_rope"),
            "limitations": [
                "Checkpoints do not prove every intervening token was processed exactly once.",
                "Live logits and positional encodings are not observable from checkpoints.",
                "Resume equivalence cannot be established without a separate comparison.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=60)
    parser.add_argument("--stall-seconds", type=float, default=1200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.stall_seconds <= 0:
        parser.error("interval and stall thresholds must be positive")
    output = args.output.resolve()
    if output.is_relative_to(args.experiment_dir.resolve()):
        parser.error("diagnostics output must be outside the active experiment directory")
    monitor = Monitor(args.experiment_dir, args.model_config, args.pid, args.stall_seconds)
    with output.open("x", encoding="utf-8") as log:
        while True:
            result = monitor.observe()
            log.write(json.dumps(result, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())
            print(
                f"diagnostics: {result['latest_processed_tokens']} tokens; "
                f"{len(result['issues'])} issue(s); alive={result['process_alive']}",
                flush=True,
            )
            if args.once or not result["process_alive"]:
                break
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
