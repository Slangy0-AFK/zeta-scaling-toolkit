"""Checks for the toy tuning grid and matched comparisons."""

import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "tune_copy_task.py"


def test_grid_reports_matched_seeds_and_all_configurations():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--steps",
            "4",
            "--batch-size",
            "2",
            "--seeds",
            "3",
            "4",
            "--attention-values",
            "0",
            "1",
            "--floor-ratios",
            "0.01",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["config"] for record in records] == list("AB BCCDDDD".replace(" ", ""))
    assert all(record["seeds"] == [3, 4] for record in records)
    assert all(len(record["evaluation_losses"]) == 2 for record in records)
    assert all(math.isfinite(record["mean_evaluation_loss"]) for record in records)
    baseline = records[0]
    zero_attention = next(
        record for record in records if record["config"] == "C" and record["A_attn"] == 0
    )
    for expected, actual in zip(baseline["evaluation_losses"], zero_attention["evaluation_losses"]):
        assert math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-6)
    low_floor = next(
        record for record in records if record["config"] == "B" and record["floor_ratio"] == 0.01
    )
    high_floor = next(
        record for record in records if record["config"] == "B" and record["floor_ratio"] == 1
    )
    assert low_floor["evaluation_losses"] != high_floor["evaluation_losses"]


def test_grid_rejects_invalid_floor():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--floor-ratios", "1.1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "floor ratio must be between 0 and 1" in result.stderr
