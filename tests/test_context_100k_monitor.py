import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
from safetensors.numpy import save_file

from tools import monitor_context_100k as diagnostics


def test_digest_hashes_large_files_in_chunks(tmp_path):
    content = b"checkpoint-data" * 100_000
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(content)

    assert diagnostics.digest(path) == hashlib.sha256(content).hexdigest()


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    (root / "checkpoints").mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "num_attention_heads": 1,
                "hidden_size": 4,
            }
        )
    )
    monkeypatch.setattr(diagnostics, "CONFIG_SHA256", diagnostics.digest(config))
    (root / "CORRECTNESS_RESULTS.json").write_text(
        json.dumps(
            {"stream_sha256": "stream", "weights_sha256": "weights", "model_revision": "rev"}
        )
    )
    return root, config


def write_checkpoint(root, end, *, nan=False, changed=False, wrong_metadata=False):
    positions = diagnostics.expected_positions(end)
    shape = (1, 1, len(positions), 4)
    arrays = {"positions": np.array(positions, dtype=np.int64)}
    for layer in range(2):
        for kind in ("keys", "values"):
            data = np.broadcast_to(
                np.array(positions, dtype=np.float32)[None, None, :, None], shape
            ).copy()
            if nan and layer == 0 and kind == "keys":
                data[0, 0, 0, 0] = np.nan
            if changed and layer == 0 and kind == "values":
                data[0, 0, 0, 0] += 1
            arrays[f"layer_{layer}_{kind}"] = data
    path = root / "checkpoints" / f"position-{end:06d}.safetensors"
    save_file(
        arrays,
        str(path),
        metadata={
            "next_position": str(end),
            "stream_sha256": "wrong" if wrong_metadata else "stream",
            "model_sha256": "weights",
            "model_revision": "rev",
        },
    )
    entry = {
        "processed_tokens": end,
        "first_position": 0,
        "last_position": end - 1,
        "checkpoint": str(path.relative_to(root)),
        "retained_positions": positions,
        "cache_bytes": sum(x.nbytes for k, x in arrays.items() if k != "positions")
        + len(positions) * 8 * 2,
        "stream_sha256": "stream",
        "elapsed_seconds": float(end),
        "tokens_per_second": 1.0,
    }
    with (root / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")


def test_checks_new_checkpoints_and_shared_cache(experiment):
    root, config = experiment
    write_checkpoint(root, 2048)
    monitor = diagnostics.Monitor(root, config, os.getpid(), 1200)
    first = monitor.observe()
    assert first["latest_processed_tokens"] == 2048
    assert first["issues"] == []
    write_checkpoint(root, 4096)
    second = monitor.observe()
    assert second["issues"] == []
    assert second["new_checkpoints"][0]["shared_positions_checked"] == 64


def test_detects_numerical_provenance_and_cache_changes(experiment):
    root, config = experiment
    write_checkpoint(root, 2048)
    monitor = diagnostics.Monitor(root, config, os.getpid(), 1200)
    assert monitor.observe()["issues"] == []
    write_checkpoint(root, 4096, nan=True, changed=True, wrong_metadata=True)
    issues = monitor.observe()["issues"]
    assert any("nonfinite cache tensor" in issue for issue in issues)
    assert any("stream_sha256" in issue for issue in issues)
    assert any("retained cache state changed" in issue for issue in issues)


def test_detects_missing_checkpoint_and_progress_rewrite(experiment):
    root, config = experiment
    write_checkpoint(root, 4096)
    monitor = diagnostics.Monitor(root, config, os.getpid(), 1200)
    assert any("checkpoint sequence" in issue for issue in monitor.observe()["issues"])
    (root / "progress.jsonl").write_text("")
    assert any("modified or truncated" in issue for issue in monitor.observe()["issues"])


def test_ignores_partial_final_progress_line(experiment):
    root, config = experiment
    write_checkpoint(root, 2048)
    with (root / "progress.jsonl").open("a") as stream:
        stream.write('{"processed_tokens":')
    result = diagnostics.Monitor(root, config, os.getpid(), 1200).observe()
    assert result["latest_processed_tokens"] == 2048
    assert result["incomplete_progress_line"]
    assert result["issues"] == []


def test_reports_suspected_stall_and_process_exit(experiment, monkeypatch):
    root, config = experiment
    write_checkpoint(root, 2048)
    old_time = time.time() - 10
    os.utime(root / "progress.jsonl", (old_time, old_time))
    monitor = diagnostics.Monitor(root, config, os.getpid(), 1)
    assert monitor.observe()["suspected_stall"]
    monkeypatch.setattr(diagnostics, "process_identity", lambda pid: None)
    result = monitor.observe()
    assert not result["process_alive"]
    assert result["terminal_status"] == "missing_status"
    assert any("stopped without successful completion" in issue for issue in result["issues"])


def test_output_cannot_be_written_inside_experiment(experiment):
    root, config = experiment
    result = subprocess.run(
        [
            sys.executable,
            diagnostics.__file__,
            "--experiment-dir",
            str(root),
            "--model-config",
            str(config),
            "--pid",
            str(os.getpid()),
            "--output",
            str(root / "health.jsonl"),
            "--once",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must be outside the active experiment directory" in result.stderr
    assert not (root / "health.jsonl").exists()
