"""Verify run-level service failures are durably recorded and stop admission."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import integration_control

RUN_ID = "zeta100k-fresh-20261117-5bc2985bbe99-failure-test"
CONTROL = Path(__file__).with_name("integration_control.py")


class IntegrationFailureRecordTests(unittest.TestCase):
    def test_completion_updates_admitted_pipeline_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "integration-state.json").write_text(
                json.dumps({"services": {"mathmesh": {"metadata_id": "pipeline-123"}}})
            )
            responses = [
                ({"ok": True}, 0.1),
                ({"access_token": "test-token"}, 0.1),
                ({"status": "complete"}, 0.1),
                ({"seq": 1}, 0.1),
            ]
            with (
                patch.dict(
                    os.environ,
                    {
                        "MATHMESH_URL": "http://127.0.0.1:1",
                        "MATHMESH_EMAIL": "test@example.invalid",
                        "MATHMESH_PASSWORD": "test-password",
                    },
                ),
                patch.object(integration_control, "request_json", side_effect=responses) as request,
            ):
                result = integration_control.call_mathmesh(RUN_ID, "complete", root)
            self.assertEqual(result["metadata_status"], "complete")
            self.assertEqual(request.call_args_list[2].kwargs["payload"]["id"], "pipeline-123")

    def test_each_support_failure_is_recorded_before_exit(self):
        cases = {
            "mathmesh": {"MATHMESH_URL": "http://127.0.0.1:1"},
            "rag": {
                "RAG_GOVERNOR_URL": "http://127.0.0.1:1",
                "RAG_GOVERNOR_OPERATOR_KEY": "test-key",
            },
            "stability": {"STABILITY_TOOLKIT_PATH": "/tmp/missing-stability-toolkit"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for service, overrides in cases.items():
                with self.subTest(service=service):
                    run_dir = root / service
                    run_dir.mkdir()
                    environment = os.environ.copy()
                    environment.update(overrides)
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(CONTROL),
                            "--phase",
                            "admit",
                            "--services",
                            service,
                            "--run-id",
                            f"{RUN_ID}-{service}",
                            "--run-dir",
                            str(run_dir),
                        ],
                        capture_output=True,
                        text=True,
                        env=environment,
                        timeout=5,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    failure = json.loads((run_dir / "integration-failure.json").read_text())
                    self.assertEqual(failure["service"], service)
                    self.assertEqual(failure["phase"], "admit")
                    self.assertEqual(failure["failure_count"], 1)
                    self.assertNotIn(
                        "integration-state.json", {path.name for path in run_dir.iterdir()}
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
