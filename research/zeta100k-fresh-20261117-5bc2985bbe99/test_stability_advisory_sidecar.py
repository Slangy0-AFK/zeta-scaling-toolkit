"""Focused tests for advisory-only resource monitoring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from stability_advisory_sidecar import StabilityAdvisoryObserver


class StabilityAdvisoryObserverTests(unittest.TestCase):
    def test_records_a_read_only_advisory_for_a_new_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = {
                "processed_tokens": 2048,
                "last_position": 2047,
                "stream_sha256": "a" * 64,
                "checkpoint": "checkpoints/position-002048.safetensors",
                "tokens_per_second": 10.0,
            }
            (root / "progress.jsonl").write_text(json.dumps(progress) + "\n")
            toolkit = SimpleNamespace(
                detect_cpu_cores=lambda: 4,
                detect_available_memory_mb=lambda: 10_000.0,
                get_recommendation=lambda tokens, **kwargs: {
                    "workload_tokens": tokens,
                    "cpu_cores": kwargs["cpu_cores"],
                },
            )
            observer = StabilityAdvisoryObserver(root, toolkit)

            self.assertEqual(observer.poll_once(), 1)
            record = json.loads((root / "stability-recommendations.jsonl").read_text())

            self.assertEqual(record["processed_tokens"], 2048)
            self.assertEqual(record["last_position"], 2047)
            self.assertFalse(record["applied_to_zeta"])
            self.assertEqual(record["recommendation"]["cpu_cores"], 4)
            self.assertEqual(observer.poll_once(), 0)

    def test_waits_for_a_complete_jsonl_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = {
                "processed_tokens": 64,
                "last_position": 63,
                "stream_sha256": "b" * 64,
                "checkpoint": "checkpoints/position-000064.safetensors",
            }
            (root / "progress.jsonl").write_text(json.dumps(progress))
            toolkit = SimpleNamespace(
                detect_cpu_cores=lambda: 4,
                detect_available_memory_mb=lambda: 10_000.0,
                get_recommendation=lambda tokens, **kwargs: {},
            )
            observer = StabilityAdvisoryObserver(root, toolkit)

            self.assertEqual(observer.poll_once(), 0)
            with (root / "progress.jsonl").open("a") as stream:
                stream.write("\n")
            self.assertEqual(observer.poll_once(), 1)

    def test_restart_skips_records_already_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [
                {
                    "processed_tokens": position,
                    "last_position": position - 1,
                    "stream_sha256": "d" * 64,
                    "checkpoint": f"checkpoints/position-{position:06d}.safetensors",
                }
                for position in (2048, 4096, 6144)
            ]
            (root / "progress.jsonl").write_text(
                "".join(json.dumps(entry) + "\n" for entry in entries)
            )
            (root / "stability-recommendations.jsonl").write_text(
                json.dumps({"processed_tokens": 4096}) + "\n"
            )
            toolkit = SimpleNamespace(
                detect_cpu_cores=lambda: 4,
                detect_available_memory_mb=lambda: 10_000.0,
                get_recommendation=lambda tokens, **kwargs: {},
            )
            observer = StabilityAdvisoryObserver(root, toolkit)

            self.assertEqual(observer.poll_once(), 1)
            records = [
                json.loads(line)
                for line in (root / "stability-recommendations.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["processed_tokens"] for record in records], [4096, 6144]
            )

    def test_rejects_inconsistent_checkpoint_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = {
                "processed_tokens": 64,
                "last_position": 60,
                "stream_sha256": "c" * 64,
                "checkpoint": "checkpoints/position-000064.safetensors",
            }
            (root / "progress.jsonl").write_text(json.dumps(progress) + "\n")
            toolkit = SimpleNamespace()
            observer = StabilityAdvisoryObserver(root, toolkit)

            with self.assertRaisesRegex(ValueError, "token progression"):
                observer.poll_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
