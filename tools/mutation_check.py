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
        "torch.exp(-log_base * frequencies / dimension)",
        "torch.exp(-log_base * frequencies / (dimension * 2))",
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
    (
        "zeta_scaling_toolkit/core.py",
        "key_positions[None, :] < sink_tokens",
        "key_positions[None, :] < 0",
    ),
    (
        "zeta_scaling_toolkit/core.py",
        "query_positions[:, None] - window + 1",
        "query_positions[:, None] - window",
    ),
    (
        "zeta_scaling_toolkit/core.py",
        "query_offset = key_length - query_length",
        "query_offset = 0",
    ),
    (
        "zeta_scaling_toolkit/core.py",
        "bias = -slope_values * distances",
        "bias = slope_values * distances",
    ),
    (
        "zeta_scaling_toolkit/core.py",
        "counts = torch.tensor(window + sink_tokens, dtype=q.dtype, device=q.device)",
        "counts = torch.tensor(window, dtype=q.dtype, device=q.device)",
    ),
    (
        "benchmarks/length_generalization.py",
        "queries[:, None] - window + 1",
        "queries[:, None] - window",
    ),
    (
        "benchmarks/context_ability.py",
        "tokens[rows, positions + 1] = values",
        "tokens[rows, positions] = values",
    ),
    (
        "benchmarks/context_char.py",
        "selected = losses[:, start:end]",
        "selected = losses[:, start + 1 : end]",
    ),
    (
        "benchmarks/context_successor.py",
        "digits = selected != 10",
        "digits = selected == 10",
    ),
    (
        "benchmarks/context_successor.py",
        "roles[start:position] = torch.arange(1, len(number) + 1)",
        "roles[start:position] = torch.arange(len(number), 0, -1)",
    ),
    (
        "benchmarks/context_successor.py",
        "index < len(number) - 1 and digit != old_digit",
        "index < len(number) - 1 and digit == old_digit",
    ),
    (
        "benchmarks/context_successor.py",
        "targets[:, rule_start - 5 : rule_end - 5]",
        "targets[:, rule_start - 4 : rule_end - 4]",
    ),
    (
        "benchmarks/context_successor.py",
        "last_digits[:, rule_start:rule_end], (copied + 1) % 10, copied",
        "last_digits[:, rule_start:rule_end], copied, copied",
    ),
    (
        "benchmarks/context_successor.py",
        "train = encode_successors(100, 700)",
        "train = encode_successors(100, 701)",
    ),
    (
        "benchmarks/context_successor.py",
        "offsets = offsets[~held_out]",
        "offsets = offsets[held_out]",
    ),
    (
        "benchmarks/context_successor.py",
        "sum(value >= 0.95 for value in accuracies) >= 8",
        "sum(value >= 0.95 for value in accuracies) >= 7",
    ),
    (
        "benchmarks/context_extrapolation.py",
        "TUNING_SEEDS = range(100, 110)",
        "TUNING_SEEDS = range(42, 52)",
    ),
    (
        "benchmarks/context_extrapolation.py",
        "method: max(\n            candidates,",
        "method: min(\n            candidates,",
    ),
    (
        "benchmarks/context_extrapolation.py",
        'accuracy["ci_low"] > 0 and abs(entry["relative_training_perplexity"]) <= 0.01',
        'accuracy["ci_high"] > 0 and abs(entry["relative_training_perplexity"]) <= 0.01',
    ),
    (
        "benchmarks/context_needle_tuned.py",
        'return sum(record["correct"] / record["total"] >= 0.9 for record in records) >= 8',
        'return sum(record["correct"] / record["total"] >= 0.9 for record in records) >= 7',
    ),
    (
        "benchmarks/context_char_tuned.py",
        "selected[method] = min((0, 1), key=lambda index: losses[index])",
        "selected[method] = max((0, 1), key=lambda index: losses[index])",
    ),
    (
        "benchmarks/context_report.py",
        "values[seed] - baseline[seed] for seed in SEEDS",
        "values[seed] - baseline[42] for seed in SEEDS",
    ),
    (
        "benchmarks/context_reuse.py",
        'block.rope_scale = settings["rope_scale"]',
        "block.rope_scale = 1.0",
    ),
    (
        "benchmarks/length_generalization.py",
        "rotary_embedding(q[..., -1:, :], positions[-1:], base=self.rope_base)",
        "rotary_embedding(q[..., -1:, :], positions[:1], base=self.rope_base)",
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
