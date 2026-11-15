"""Run the full test suite against targeted, temporary source mutations."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = (
    (
        "zeta_scaling_toolkit/core.py",
        "max(1.0, math.log(counts) / math.log(self.train_length))",
        "1.0",
    ),
    (
        "zeta_scaling_toolkit/core.py",
        "counts.clamp_min(1).to(dtype).log() / math.log(self.train_length)",
        "counts.clamp_min(1).to(dtype).log() * 0 + 1.0",
    ),
    ("zeta_scaling_toolkit/core.py", "max=self.tau_eff_max", "max=1e30"),
    (
        "zeta_scaling_toolkit/positions.py",
        "effective_base ** (-frequencies / dimension)",
        "effective_base ** (-frequencies / (dimension * 2))",
    ),
    (
        "zeta_scaling_toolkit/positions.py",
        "positions.to(torch.float64)",
        "positions.to(torch.float16)",
    ),
    (
        "zeta_scaling_toolkit/positions.py",
        "position_values / scale_factor",
        "position_values * scale_factor",
    ),
    (
        "zeta_scaling_toolkit/positions.py",
        "exponent = dimension / (dimension - 2) if dimension > 2 else 1.0",
        "exponent = 1.0",
    ),
    (
        "zeta_scaling_toolkit/positions.py",
        "bias[:, :, start:end, :] = -slopes[None, :, None, None] * distances[None, None, :, :]",
        "bias[:, :, start:end, :] = slopes[None, :, None, None] * distances[None, None, :, :]",
    ),
    (
        "zeta_scaling_toolkit/positions.py",
        "torch.exp2(-8.0 * head_ids / num_heads)",
        "torch.exp2(-4.0 * head_ids / num_heads)",
    ),
    (
        "zeta_scaling_toolkit/positions.py",
        "block[:, :sink_end] |= key_positions[:sink_end][None, :] <= query_positions[:, None]",
        "block[:, :sink_end] &= key_positions[:sink_end][None, :] <= query_positions[:, None]",
    ),
    (
        "zeta_scaling_toolkit/positions.py",
        "query_positions[:, None] - window + 1",
        "query_positions[:, None] - window",
    ),
    (
        "zeta_scaling_toolkit/positions.py",
        "positions >= self.next_position - self.window + 1",
        "positions >= self.next_position - self.window",
    ),
    (
        "zeta_scaling_toolkit/positions.py",
        "(positions < self.sink_tokens) | (positions >= self.next_position - self.window + 1)",
        "(positions >= self.next_position - self.window + 1)",
    ),
)


def run_mutant(index: int, relative_path: str, old: str, new: str) -> str:
    path = ROOT / relative_path
    original = path.read_bytes()
    old_bytes, new_bytes = old.encode(), new.encode()
    if original.count(old_bytes) < 1:
        print(f"M{index:02}: NOT FOUND: {relative_path}: {old}")
        return "NOT FOUND"

    result: subprocess.CompletedProcess[str] | None = None
    try:
        path.write_bytes(original.replace(old_bytes, new_bytes, 1))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-x",
                "-q",
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        path.write_bytes(original)

    if result is None:
        status = "ERROR"
    elif result.returncode == 1:
        status = "KILLED"
    elif result.returncode == 0:
        status = "SURVIVED"
    else:
        status = f"ERROR (pytest exit {result.returncode})"
    print(f"M{index:02}: {status}: {relative_path}: {old}")
    if status.startswith("ERROR") and result is not None:
        print(result.stdout[-2000:])
    return status


def main() -> int:
    results = [run_mutant(index, *mutant) for index, mutant in enumerate(MUTANTS, start=1)]
    killed = results.count("KILLED")
    survivors = results.count("SURVIVED")
    missing = results.count("NOT FOUND")
    errors = len(results) - killed - survivors - missing
    print(
        f"Mutation results: {killed} killed, {survivors} survived, "
        f"{missing} not found, {errors} errors."
    )
    return int(survivors > 0 or missing > 0 or errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
