import fcntl
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "research/context-100k-experiment-v2/run_in_tmux.sh"


def executable(path: Path, content: str) -> Path:
    path.write_text(content)
    path.chmod(0o755)
    return path


def test_launcher_preflights_then_resumes_with_venv_python(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(
        bin_dir / "tmux",
        "#!/bin/bash\n"
        'if [[ "$1" == has-session ]]; then exit 1; fi\n'
        '[[ "$1" == new-session ]] || exit 2\n'
        'shift 4\nexec "$@"\n',
    )
    fake_python = executable(
        bin_dir / "fake-python",
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "with (output / 'invocations').open('a') as f:\n"
        "    f.write(sys.argv[0] + ' ' + sys.argv[-1] + '\\n')\n"
        "if '--preflight-only' in sys.argv:\n"
        "    (output / 'CORRECTNESS_RESULTS.json').write_text('{}')\n"
        "    (output / 'stream.safetensors').write_text('stream')\n",
    )
    venv_python = bin_dir / "venv-python"
    venv_python.symlink_to(fake_python)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").touch()
    output = tmp_path / "output"
    output.mkdir()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    subprocess.run(
        ["bash", str(LAUNCHER), str(output), str(snapshot), str(venv_python)],
        env=env,
        check=True,
    )
    invocations = (output / "invocations").read_text().splitlines()
    assert invocations == [
        f"{venv_python} --preflight-only",
        f"{venv_python} --resume",
    ]

    subprocess.run(
        ["bash", str(LAUNCHER), str(output), str(snapshot), str(venv_python)],
        env=env,
        check=True,
    )
    assert (output / "invocations").read_text().splitlines()[-1] == f"{venv_python} --resume"
    assert len((output / "invocations").read_text().splitlines()) == 3


def test_worker_rejects_locked_output(tmp_path):
    lock = (tmp_path / ".experiment.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--worker", str(tmp_path), str(tmp_path), "/bin/false"],
            check=False,
        )
        assert result.returncode != 0
        assert "holds the experiment lock" in (tmp_path / "experiment.log").read_text()
    finally:
        lock.close()


def test_worker_rejects_incomplete_preflight(tmp_path):
    (tmp_path / "CORRECTNESS_RESULTS.json").touch()
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--worker", str(tmp_path), str(tmp_path), "/bin/false"],
        check=False,
    )
    assert result.returncode != 0
    assert "Incomplete preflight artifacts" in (tmp_path / "experiment.log").read_text()


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_real_tmux_runs_worker_in_isolated_session(tmp_path):
    fake_python = executable(
        tmp_path / "fake-python",
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "if '--preflight-only' in sys.argv:\n"
        "    (output / 'CORRECTNESS_RESULTS.json').write_text('{}')\n"
        "    (output / 'stream.safetensors').write_text('stream')\n"
        "else:\n"
        "    (output / 'finished').touch()\n",
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").touch()
    output = tmp_path / "output"
    output.mkdir()
    env = {**os.environ, "TMUX_TMPDIR": str(tmp_path)}
    env.pop("TMUX", None)
    subprocess.run(
        ["bash", str(LAUNCHER), str(output), str(snapshot), str(fake_python)],
        env=env,
        check=True,
    )
    deadline = time.monotonic() + 10
    while not (output / "finished").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (output / "finished").exists(), (output / "experiment.log").read_text()
