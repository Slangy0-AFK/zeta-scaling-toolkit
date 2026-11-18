"""Record pinned Stability Toolkit recommendations for new Zeta checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_PROGRESS_LINE_BYTES = 4096
MAX_CHECKPOINT_CHARS = 256


def load_toolkit(path: Path) -> Any:
    source = path / "crash_fix.py"
    spec = importlib.util.spec_from_file_location("zeta_stability_advisor", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Stability Toolkit from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StabilityAdvisoryObserver:
    """Read checkpoint summaries and append advisory-only resource recommendations."""

    def __init__(self, run_dir: Path, toolkit: Any, *, poll_seconds: float = 1.0) -> None:
        if not 0 < poll_seconds <= 60:
            raise ValueError("poll interval must be greater than 0 and at most 60 seconds")
        self.run_dir = run_dir
        self.progress_path = run_dir / "progress.jsonl"
        self.output_path = run_dir / "stability-recommendations.jsonl"
        self.toolkit = toolkit
        self.poll_seconds = poll_seconds
        self.offset = 0
        self.last_processed = 0
        if self.output_path.exists():
            lines = self.output_path.read_text(encoding="utf-8").splitlines()
            if lines:
                previous = json.loads(lines[-1])
                self.last_processed = previous["processed_tokens"]
        self.replaying_existing = self.last_processed > 0

    def poll_once(self) -> int:
        if not self.progress_path.exists():
            return 0
        accepted = 0
        with self.progress_path.open("rb") as stream:
            stream.seek(self.offset)
            while True:
                line_start = stream.tell()
                raw = stream.readline(MAX_PROGRESS_LINE_BYTES + 1)
                if not raw:
                    break
                if len(raw) > MAX_PROGRESS_LINE_BYTES:
                    raise ValueError("progress record exceeds the configured size bound")
                if not raw.endswith(b"\n"):
                    stream.seek(line_start)
                    break
                self.offset = stream.tell()
                entry = json.loads(raw)
                if not isinstance(entry, dict):
                    raise ValueError("progress record must be a JSON object")
                processed = entry.get("processed_tokens")
                if (
                    isinstance(processed, bool)
                    or not isinstance(processed, int)
                    or not 1 <= processed <= 100_000
                    or entry.get("last_position") != processed - 1
                ):
                    raise ValueError("invalid checkpoint token progression")
                checkpoint = entry.get("checkpoint")
                if not isinstance(checkpoint, str) or len(checkpoint) > MAX_CHECKPOINT_CHARS:
                    raise ValueError("invalid checkpoint reference")
                stream_hash = entry.get("stream_sha256")
                if not isinstance(stream_hash, str) or len(stream_hash) != 64:
                    raise ValueError("invalid stream hash")
                int(stream_hash, 16)
                if self.replaying_existing and processed <= self.last_processed:
                    continue
                if processed < self.last_processed:
                    raise ValueError("checkpoint progress moved backwards")
                if processed == self.last_processed:
                    continue
                self.replaying_existing = False

                cpu_cores = self.toolkit.detect_cpu_cores()
                available_memory_mb = self.toolkit.detect_available_memory_mb()
                started = time.perf_counter()
                recommendation = self.toolkit.get_recommendation(
                    100_000,
                    cpu_cores=cpu_cores,
                    available_memory_mb=available_memory_mb,
                )
                record = {
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "processed_tokens": processed,
                    "last_position": processed - 1,
                    "stream_sha256": stream_hash,
                    "checkpoint": checkpoint,
                    "tokens_per_second": entry.get("tokens_per_second"),
                    "cpu_cores": cpu_cores,
                    "available_memory_mb": available_memory_mb,
                    "recommendation": recommendation,
                    "applied_to_zeta": False,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                with self.output_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                self.last_processed = processed
                accepted += 1
        return accepted

    def run(self) -> None:
        while True:
            self.poll_once()
            time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--toolkit-path", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    observer = StabilityAdvisoryObserver(
        args.run_dir.resolve(),
        load_toolkit(args.toolkit_path.resolve()),
        poll_seconds=args.poll_seconds,
    )
    if args.once:
        print(json.dumps({"recorded": observer.poll_once()}))
        return
    try:
        observer.run()
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
