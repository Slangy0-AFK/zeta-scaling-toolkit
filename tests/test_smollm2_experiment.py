"""Resume behavior tests for the isolated SmolLM2 experiment runner."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/context-100k-experiment-v2"))
from run_experiment import has_completed_results  # noqa: E402


@pytest.mark.parametrize(
    ("status", "expected"),
    [("completed", True), ("incomplete", False), ("failed", False)],
)
def test_resume_only_short_circuits_completed_results(tmp_path, status, expected):
    results_path = tmp_path / "resource-results.json"
    results_path.write_text(json.dumps({"status": status}), encoding="utf-8")

    assert has_completed_results(results_path) is expected


def test_resume_rejects_invalid_existing_result_status(tmp_path):
    results_path = tmp_path / "resource-results.json"
    results_path.write_text(json.dumps({"status": "running"}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid status"):
        has_completed_results(results_path)
