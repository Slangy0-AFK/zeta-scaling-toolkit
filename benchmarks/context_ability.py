"""Resumable training-length sanity check for associative retrieval."""

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zeta_scaling_toolkit import AsymptoticLRScheduler

if __package__:
    from .length_generalization import LEARNING_RATE, SequenceLM, positive_int
else:
    from length_generalization import LEARNING_RATE, SequenceLM, positive_int

NOISE_TOKENS = 64
KEY_TOKENS = 16
VALUE_TOKENS = 16
QUERY_TOKEN = NOISE_TOKENS + KEY_TOKENS + VALUE_TOKENS
VOCABULARY_SIZE = QUERY_TOKEN + 1


def planted_needle_batch(
    batch_size: int, length: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 1 or length < 4:
        raise ValueError("batch_size must be positive and length must be at least four")
    tokens = torch.randint(NOISE_TOKENS, (batch_size, length), generator=generator)
    keys = torch.randint(KEY_TOKENS, (batch_size,), generator=generator) + NOISE_TOKENS
    values = (
        torch.randint(VALUE_TOKENS, (batch_size,), generator=generator) + NOISE_TOKENS + KEY_TOKENS
    )
    positions = torch.randint(length - 3, (batch_size,), generator=generator)
    rows = torch.arange(batch_size)
    tokens[rows, positions] = keys
    tokens[rows, positions + 1] = values
    tokens[:, -2] = QUERY_TOKEN
    tokens[:, -1] = keys
    return tokens, values


def train_needle(
    seed: int, steps: int, length: int, batch_size: int, full_output: bool = False
) -> SequenceLM:
    torch.manual_seed(seed)
    model = SequenceLM(VOCABULARY_SIZE, "rope-full", False, length, 10.0, 10000.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=0.1, total_steps=steps)
    generator = torch.Generator().manual_seed(seed + 50000)
    model.train()
    for _ in range(steps):
        inputs, targets = planted_needle_batch(batch_size, length, generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)[:, -1] if full_output else model.forward_last(inputs)
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite needle loss for seed {seed}")
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model.eval()


def evaluate_needle_sanity(
    model: SequenceLM, seed: int, length: int, samples: int, batch_size: int
) -> tuple[int, int]:
    generator = torch.Generator().manual_seed(seed + length + 90000)
    correct = 0
    with torch.no_grad():
        for start in range(0, samples, batch_size):
            inputs, targets = planted_needle_batch(
                min(batch_size, samples - start), length, generator
            )
            logits = model(inputs)[:, -1]
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite needle evaluation logits")
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
    return correct, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=3000)
    parser.add_argument("--train-length", type=positive_int, default=256)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--eval-queries", type=positive_int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--max-runs", type=positive_int)
    parser.add_argument("--full-output", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/context-ability-needle-sanity.jsonl")
    )
    args = parser.parse_args()
    if args.train_length < 4 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error(
            "train-length must be at least four and seeds must be distinct and nonnegative"
        )
    torch.set_num_threads(1)
    records: list[dict[str, Any]] = []
    if args.output.exists():
        records = [json.loads(line) for line in args.output.read_text().splitlines() if line]
    configuration = {
        "variant": "rope-full",
        "steps": args.steps,
        "train_length": args.train_length,
        "batch_size": args.batch_size,
        "eval_queries": args.eval_queries,
    }
    if any(any(record[key] != value for key, value in configuration.items()) for record in records):
        raise ValueError("output contains a different experiment configuration")
    completed = {record["seed"] for record in records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_runs = 0
    with args.output.open("a", encoding="utf-8") as output:
        for seed in args.seeds:
            if seed in completed:
                continue
            model = train_needle(
                seed, args.steps, args.train_length, args.batch_size, args.full_output
            )
            correct, total = evaluate_needle_sanity(
                model, seed, args.train_length, args.eval_queries, args.batch_size
            )
            record = {
                **configuration,
                "seed": seed,
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
                "training_implementation": "full-output" if args.full_output else "last-query",
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "platform": platform.platform(),
                    "torch_threads": torch.get_num_threads(),
                },
            }
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            new_runs += 1
            if args.max_runs is not None and new_runs >= args.max_runs:
                break


if __name__ == "__main__":
    main()
