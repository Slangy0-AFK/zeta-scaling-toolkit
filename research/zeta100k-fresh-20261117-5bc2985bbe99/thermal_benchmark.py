"""Run the existing Zeta attention microbenchmark with optional procfs sampling."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
THERMAL_ROOT = Path("/sys/class/thermal")
HWMON_ROOT = Path("/sys/class/hwmon")
CGROUP_ROOT = Path("/sys/fs/cgroup")


def cpu_stat() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def process_cpu_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return int(fields[11]) + int(fields[12])
    except (OSError, ValueError, IndexError):
        return None


def read_process_memory(pid: int) -> dict[str, float] | None:
    try:
        values = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, value, _unit = line.split()
                values[name[:-1].lower() + "_mib"] = int(value) / 1024
        return values
    except (OSError, ValueError):
        return None


def cpu_frequencies_mhz() -> list[float]:
    frequencies = []
    for match in re.finditer(r"^cpu MHz\s*:\s*([0-9.]+)", Path("/proc/cpuinfo").read_text(), re.M):
        frequencies.append(float(match.group(1)))
    return frequencies


def memory_available_mib() -> float | None:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return None


def read_integer_file(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def thermal_snapshot() -> dict[str, Any]:
    zones = []
    for path in sorted(THERMAL_ROOT.glob("thermal_zone*/temp")):
        raw = read_integer_file(path)
        if raw is not None:
            zones.append({"path": str(path), "temperature_c": raw / 1000})
    hwmon = []
    for path in sorted(HWMON_ROOT.glob("hwmon*/temp*_input")):
        raw = read_integer_file(path)
        if raw is not None:
            label_path = path.with_name(path.name.replace("_input", "_label"))
            try:
                label = label_path.read_text().strip()
            except OSError:
                label = None
            hwmon.append({"path": str(path), "label": label, "temperature_c": raw / 1000})
    cooling = []
    for directory in sorted(THERMAL_ROOT.glob("cooling_device*")):

        def read_text(name: str) -> str | None:
            try:
                return (directory / name).read_text().strip()
            except OSError:
                return None

        cooling.append(
            {
                "type": read_text("type"),
                "current_state": read_integer_file(directory / "cur_state"),
                "max_state": read_integer_file(directory / "max_state"),
            }
        )
    throttle = {}
    path = Path("/sys/devices/system/cpu/cpu0/thermal_throttle")
    if path.is_dir():
        for counter in sorted(path.glob("*")):
            value = read_integer_file(counter)
            if value is not None:
                throttle[counter.name] = value
    return {
        "thermal_zones": zones,
        "hwmon": hwmon,
        "cooling_devices": cooling,
        "thermal_throttle_counters": throttle,
    }


def cgroup_snapshot() -> dict[str, Any]:
    values: dict[str, Any] = {}
    try:
        values["cpu_max"] = (CGROUP_ROOT / "cpu.max").read_text().strip()
        for line in (CGROUP_ROOT / "cpu.stat").read_text().splitlines():
            name, number = line.split()
            values[name] = int(number)
        values["memory_current_bytes"] = int((CGROUP_ROOT / "memory.current").read_text())
        peak = (CGROUP_ROOT / "memory.peak").read_text().strip()
        values["memory_peak_bytes"] = int(peak) if peak != "max" else None
        values["memory_limit_bytes"] = (CGROUP_ROOT / "memory.max").read_text().strip()
    except (OSError, ValueError):
        pass
    return values


class Sampler:
    def __init__(self, pid: int, interval_seconds: float) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, name="thermal-observer", daemon=True)

    def sample(self) -> dict[str, Any]:
        total, idle = cpu_stat()
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host_cpu_ticks": [total, idle],
            "child_cpu_ticks": process_cpu_ticks(self.pid),
            "child_memory": read_process_memory(self.pid),
            "cpu_mhz": cpu_frequencies_mhz(),
            "available_memory_mib": memory_available_mib(),
            "thermal": thermal_snapshot(),
            "cgroup": cgroup_snapshot(),
        }

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.samples.append(self.sample())
            self.stop_event.wait(self.interval_seconds)
        self.samples.append(self.sample())

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)


def cpu_utilization_percent(samples: list[dict[str, Any]]) -> float | None:
    if len(samples) < 2:
        return None
    first = samples[0]["host_cpu_ticks"]
    last = samples[-1]["host_cpu_ticks"]
    total_delta = last[0] - first[0]
    idle_delta = last[1] - first[1]
    return None if total_delta <= 0 else 100 * (1 - idle_delta / total_delta)


def child_cpu_percent(samples: list[dict[str, Any]], elapsed: float) -> float | None:
    ticks = [
        sample["child_cpu_ticks"] for sample in samples if sample["child_cpu_ticks"] is not None
    ]
    if len(ticks) < 2 or elapsed <= 0:
        return None
    clock_ticks = os.sysconf("SC_CLK_TCK")
    return (ticks[-1] - ticks[0]) / clock_ticks / elapsed * 100


def run_one(mode: str, args: argparse.Namespace, output_dir: Path, sequence: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "benchmarks/window_speed.py"),
        "--head-dimension",
        str(args.head_dimension),
        "--lengths",
        *(str(length) for length in args.lengths),
        "--windows",
        *(str(window) for window in args.windows),
        "--repeats",
        str(args.repeats),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    with tempfile.TemporaryFile(mode="w+b") as capture:
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        before_resources = resource.getrusage(resource.RUSAGE_SELF)
        before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        host_ticks_before = cpu_stat()
        process = subprocess.Popen(
            command, stdout=capture, stderr=subprocess.STDOUT, env=environment
        )
        samples: list[dict[str, Any]] = []
        sampler = Sampler(process.pid, args.sample_interval) if mode == "telemetry_on" else None
        before = {
            "host_cpu_ticks": host_ticks_before,
            "cpu_mhz": cpu_frequencies_mhz(),
            "available_memory_mib": memory_available_mib(),
            "thermal": thermal_snapshot(),
            "cgroup": cgroup_snapshot(),
        }
        if sampler:
            sampler.start()
        while process.poll() is None:
            if sampler:
                time.sleep(args.sample_interval)
            else:
                time.sleep(0.1)
        return_code = process.wait()
        elapsed = time.perf_counter() - started_wall
        host_ticks_after = cpu_stat()
        if sampler:
            sampler.stop()
            samples = sampler.samples
        after = {
            "host_cpu_ticks": host_ticks_after,
            "cpu_mhz": cpu_frequencies_mhz(),
            "available_memory_mib": memory_available_mib(),
            "thermal": thermal_snapshot(),
            "cgroup": cgroup_snapshot(),
        }
        capture.seek(0)
        stdout = capture.read().decode("utf-8", errors="replace")
        if return_code != 0:
            raise RuntimeError(f"window_speed failed ({return_code}): {stdout[-4000:]}")
        benchmark = json.loads(stdout)
        after_resources = resource.getrusage(resource.RUSAGE_SELF)
        after_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    if not samples:
        samples = [before, after]
    child_rss_samples = [
        sample["child_memory"].get("vmhwm_mib")
        for sample in samples
        if sample.get("child_memory") and sample["child_memory"].get("vmhwm_mib") is not None
    ]
    frequencies = [
        value
        for sample in samples
        for value in sample.get("cpu_mhz", [])
        if isinstance(value, (int, float))
    ]
    child_cpu_seconds = (after_children.ru_utime + after_children.ru_stime) - (
        before_children.ru_utime + before_children.ru_stime
    )
    rows = []
    for result in benchmark["results"]:
        length = result["length"]
        for window_row in result["window_ms"]:
            mode_times = result["window_ms"][window_row]
            rows.append(
                {
                    "length": length,
                    "window": result["window"],
                    "scaling": result["scaling"],
                    "measurement": window_row,
                    "full_ms": result["full_causal_log_ms"][window_row],
                    "window_ms": mode_times,
                    "forward_ratio": result["full_causal_log_ms"][window_row] / mode_times,
                    "full_attention_tokens_per_second": length
                    / (result["full_causal_log_ms"][window_row] / 1000),
                    "window_attention_tokens_per_second": length / (mode_times / 1000),
                }
            )
    return {
        "sequence": sequence,
        "telemetry_mode": mode,
        "command": command,
        "exit_code": return_code,
        "elapsed_seconds": elapsed,
        "parent_cpu_seconds": time.process_time() - started_cpu,
        "parent_cpu_seconds_delta_ru": (after_resources.ru_utime + after_resources.ru_stime)
        - (before_resources.ru_utime + before_resources.ru_stime),
        "sample_interval_seconds": args.sample_interval if sampler else None,
        "sample_count": len(samples),
        "host_cpu_utilization_percent_sampled": cpu_utilization_percent(samples),
        "benchmark_child_cpu_percent_sampled": child_cpu_percent(samples, elapsed)
        if sampler
        else child_cpu_seconds / elapsed * 100,
        "benchmark_child_cpu_seconds": child_cpu_seconds,
        "benchmark_child_peak_rss_mib_sampled": max(
            child_rss_samples, default=after_children.ru_maxrss / 1024
        ),
        "cpu_mhz_sampled_min": min(frequencies, default=None),
        "cpu_mhz_sampled_median": median(frequencies) if frequencies else None,
        "cpu_mhz_sampled_max": max(frequencies, default=None),
        "available_memory_before_mib": before["available_memory_mib"],
        "available_memory_after_mib": after["available_memory_mib"],
        "thermal_before": before["thermal"],
        "thermal_after": after["thermal"],
        "cgroup_before": before["cgroup"],
        "cgroup_after": after["cgroup"],
        "samples": samples,
        "attention_results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=(4096, 8192))
    parser.add_argument("--windows", nargs="+", type=int, default=(64, 256))
    parser.add_argument("--head-dimension", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    args = parser.parse_args()
    if args.repeats < 1 or args.pairs < 1 or args.sample_interval <= 0:
        parser.error("repeats, pairs, and sample interval must be positive")
    if any(value < 1 for value in (*args.lengths, *args.windows, args.head_dimension)):
        parser.error("lengths, windows, and head dimension must be positive")
    output = args.output.resolve()
    records = []
    for pair in range(args.pairs):
        order = (
            ("telemetry_off", "telemetry_on")
            if pair % 2 == 0
            else ("telemetry_on", "telemetry_off")
        )
        for mode in order:
            records.append(run_one(mode, args, output.parent, len(records)))
    results = {
        "run_id": "zeta100k-fresh-20261117-5bc2985bbe99",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_scope": "NonCollapsingAttention microbenchmark only; no model, tokenizer, or 100K stream loaded",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0)),
            "head_dimension": args.head_dimension,
            "lengths": args.lengths,
            "windows": args.windows,
            "repeats": args.repeats,
            "pairs": args.pairs,
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
