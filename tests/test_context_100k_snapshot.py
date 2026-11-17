import csv
import json
import sys
from unittest.mock import patch

import pytest

from tools import snapshot_context_100k_diagnostics as snapshot


def test_read_jsonl_ignores_only_incomplete_trailing_line(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"value": 1}\n{"value":')
    records, digest, partial = snapshot.read_jsonl(log)
    assert records == [{"value": 1}]
    assert len(digest) == 64
    assert partial
    log.write_text('{"value": 1}\ninvalid\n')
    with pytest.raises(json.JSONDecodeError):
        snapshot.read_jsonl(log)


def test_snapshot_records_checkpoints_and_refuses_overwrite(tmp_path):
    root = tmp_path / "root"
    experiment = root / "experiment"
    experiment.mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tools/monitor_context_100k.py").write_text("monitor")
    for name in ("run_experiment.py", "compat_adapter.py", "requirements-lock.txt"):
        (experiment / name).write_text(name)
    correctness = {
        "source_commit": "source",
        "model_id": "model",
        "model_revision": "revision",
        "weights_sha256": "weights",
        "tokenizer_sha256": "tokenizer",
        "stream_sha256": "stream",
        "corpus_sha256": "corpus",
    }
    (experiment / "CORRECTNESS_RESULTS.json").write_text(json.dumps(correctness))
    progress = [
        {"processed_tokens": 64, "interval_seconds": 32, "last_position": 63},
        {"processed_tokens": 128, "interval_seconds": 16, "last_position": 127},
    ]
    (experiment / "progress.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in progress)
    )
    health = root / "health.jsonl"
    health.write_text(
        json.dumps(
            {
                "pid": 123,
                "new_checkpoints": [
                    {"processed_tokens": 64, "checkpoint_sha256": "hash", "issues": []}
                ],
            }
        )
        + "\n"
    )
    output = root / "output"
    args = [
        "snapshot",
        "--experiment",
        str(experiment),
        "--health",
        str(health),
        "--output-dir",
        str(output),
    ]
    with (
        patch.object(snapshot, "ROOT", root),
        patch.object(sys, "argv", args),
        patch.object(snapshot.subprocess, "check_output", return_value="head\n"),
    ):
        snapshot.main()
        result = json.loads((output / "snapshot.json").read_text())
        assert result["progress"] == progress
        assert result["manifest"]["git_head_at_snapshot"] == "head"
        with (output / "checkpoints.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert [row["interval_tokens_per_second"] for row in rows] == ["2.0", "4.0"]
        assert [row["monitor_status"] for row in rows] == ["PASS", "NOT TESTED"]
        assert rows[0]["checkpoint_sha256"] == "hash"
        with pytest.raises(SystemExit):
            snapshot.main()
        assert json.loads((output / "snapshot.json").read_text()) == result
