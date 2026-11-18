"""Focused tests for the bounded out-of-process progress observer."""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from telemetry_sidecar import ProgressTelemetry, normalize_progress

STREAM_HASH = "a" * 64


def record(position: int) -> dict[str, Any]:
    return {
        "processed_tokens": position,
        "first_position": 0,
        "last_position": position - 1,
        "stream_sha256": STREAM_HASH,
        "checkpoint": f"checkpoints/position-{position:06d}.safetensors",
        "retained_positions": list(range(min(position, 4))),
        "cache_bytes": 256,
        "tokens_per_second": 3.0,
        "available_memory_mib": 4096.0,
        "peak_rss_mib": 512.0,
    }


class ProgressTelemetryTests(unittest.TestCase):
    def test_normalization_keeps_only_small_read_only_fields(self):
        event = normalize_progress(record(64))
        self.assertEqual(event["processed_tokens"], 64)
        self.assertEqual(event["last_position"], 63)
        self.assertNotIn("logits", event)
        self.assertNotIn("input_ids", event)

    def test_invalid_position_and_unbounded_fields_are_rejected(self):
        invalid = record(64)
        invalid["last_position"] = 62
        with self.assertRaises(ValueError):
            normalize_progress(invalid)
        oversized = record(64)
        oversized["retained_positions"] = list(range(129))
        with self.assertRaises(ValueError):
            normalize_progress(oversized)

    def test_queue_is_bounded_and_reports_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / "progress.jsonl"
            progress.write_text("".join(json.dumps(record(i * 64)) + "\n" for i in range(1, 10)))
            telemetry = ProgressTelemetry(progress, root / "status.json", None, queue_capacity=2)
            self.assertEqual(telemetry.poll_once(), 2)
            self.assertLessEqual(telemetry.queue.qsize(), 2)
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["dropped"], 0)
            self.assertEqual(status["backpressure_events"], 1)
            self.assertEqual(status["last_deferred_tokens"], 192)
            self.assertEqual(status["queue_high_water"], 2)

    def test_source_is_unchanged_and_transient_failures_recover_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / "progress.jsonl"
            progress.write_text(
                "".join(json.dumps(record(position)) + "\n" for position in (64, 128, 192))
            )
            original_hash = hashlib.sha256(progress.read_bytes()).hexdigest()
            attempts: list[int] = []
            lock = threading.Lock()

            def flaky_sender(event: dict[str, Any]) -> None:
                with lock:
                    attempts.append(event["processed_tokens"])
                    attempt = len(attempts)
                if attempt <= 2:
                    raise ConnectionError("controlled outage")

            telemetry = ProgressTelemetry(
                progress, root / "status.json", flaky_sender, poll_seconds=0.01
            )
            self.assertEqual(telemetry.poll_once(), 3)
            telemetry.start()
            self.assertTrue(telemetry.wait_until_idle(3.0))
            telemetry.stop()
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["delivered"], 3)
            self.assertEqual(status["failed_requests"], 2)
            self.assertEqual(status["last_delivered_tokens"], 192)
            self.assertLessEqual(status["queue_high_water"], status["queue_capacity"])
            self.assertEqual(hashlib.sha256(progress.read_bytes()).hexdigest(), original_hash)
            self.assertEqual(attempts.count(64), 3)
            self.assertEqual(attempts[-2:], [128, 192])

    def test_partial_jsonl_record_waits_for_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / "progress.jsonl"
            complete = json.dumps(record(64)) + "\n"
            partial = json.dumps(record(128))
            progress.write_text(complete + partial)
            telemetry = ProgressTelemetry(progress, root / "status.json", None)
            self.assertEqual(telemetry.poll_once(), 1)
            with progress.open("a") as stream:
                stream.write("\n")
            self.assertEqual(telemetry.poll_once(), 1)
            self.assertEqual(telemetry.queue.qsize(), 2)

    def test_run_once_drains_more_records_than_queue_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / "progress.jsonl"
            progress.write_text("".join(json.dumps(record(i * 64)) + "\n" for i in range(1, 10)))
            received = []
            telemetry = ProgressTelemetry(
                progress, root / "status.json", received.append, queue_capacity=2
            )
            result = telemetry.run_once(timeout=3)
            self.assertEqual(
                [event["processed_tokens"] for event in received], list(range(64, 577, 64))
            )
            self.assertEqual(result["last_delivered_tokens"], 576)
            self.assertEqual(result["dropped"], 0)

    def test_run_once_recovers_when_queue_was_already_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / "progress.jsonl"
            progress.write_text("".join(json.dumps(record(i * 64)) + "\n" for i in range(1, 5)))
            received = []
            telemetry = ProgressTelemetry(
                progress, root / "status.json", received.append, queue_capacity=2
            )
            self.assertEqual(telemetry.poll_once(), 2)
            result = telemetry.run_once(timeout=3)
            self.assertEqual([event["processed_tokens"] for event in received], [64, 128, 192, 256])
            self.assertEqual(result["last_delivered_tokens"], 256)

    def test_queue_capacity_cannot_exceed_fixed_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                ProgressTelemetry(root / "progress", root / "status", None, queue_capacity=129)

    def test_stop_during_failure_does_not_skip_queued_records_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / "progress.jsonl"
            progress.write_text(
                "".join(json.dumps(record(position)) + "\n" for position in (64, 128))
            )
            attempted = threading.Event()

            def unavailable(event):
                attempted.set()
                raise ConnectionError("controlled outage")

            telemetry = ProgressTelemetry(
                progress, root / "status.json", unavailable, poll_seconds=0.01
            )
            telemetry.start()
            self.assertTrue(attempted.wait(2))
            telemetry.stop()
            self.assertEqual(
                json.loads((root / "status.json").read_text())["last_delivered_tokens"], 0
            )
            received = []
            recovered = ProgressTelemetry(progress, root / "status.json", received.append)
            recovered.run_once(timeout=3)
            self.assertEqual([event["processed_tokens"] for event in received], [64, 128])

    def test_oversized_record_is_skipped_without_blocking_following_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / "progress.jsonl"
            progress.write_text("x" * 5000 + "\n" + json.dumps(record(64)) + "\n")
            received = []
            telemetry = ProgressTelemetry(progress, root / "status.json", received.append)
            result = telemetry.run_once(timeout=3)
            self.assertEqual(result["malformed_records"], 1)
            self.assertEqual([event["processed_tokens"] for event in received], [64])

    def test_isolated_observer_throughput_and_peak_rss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / "progress.jsonl"
            positions = set(range(2048, 100_001, 2048))
            positions.update((8193, 10000, 25000, 50000, 65537, 75000, 90000, 100000))
            progress.write_text(
                "".join(json.dumps(record(position)) + "\n" for position in sorted(positions)),
                encoding="utf-8",
            )
            script = Path(__file__).with_name("telemetry_sidecar.py")
            started = time.perf_counter()
            result = subprocess.run(
                [sys.executable, str(script), "--run-dir", str(root), "--once", "--local-only"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            elapsed = time.perf_counter() - started
            child_peak_rss_mib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
            status = json.loads(result.stdout)
            self.assertEqual(status["delivered"], len(positions))
            self.assertEqual(status["last_delivered_tokens"], 100_000)
            self.assertLessEqual(status["queue_high_water"], status["queue_capacity"])
            print(
                "telemetry_probe="
                f"events:{len(positions)},elapsed_s:{elapsed:.4f},"
                f"events_per_s:{len(positions) / elapsed:.1f},"
                f"accounted_token_span_per_s:{100_000 / elapsed:.0f},"
                f"child_peak_rss_mib:{child_peak_rss_mib:.1f},"
                f"queue_high_water:{status['queue_high_water']}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
