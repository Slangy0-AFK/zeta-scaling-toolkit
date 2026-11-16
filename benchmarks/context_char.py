"""Train matched full/window controls and evaluate character-context use."""

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zeta_scaling_toolkit import AsymptoticLRScheduler

if __package__:
    from .length_generalization import (
        LEARNING_RATE,
        SequenceLM,
        dataset_info,
        positive_int,
        sample_text_batch,
    )
else:
    from length_generalization import (
        LEARNING_RATE,
        SequenceLM,
        dataset_info,
        positive_int,
        sample_text_batch,
    )

VARIANTS = ("rope-full", "rope-window", "rope-log-full", "rope-log-window")
LENGTHS = (256, 512, 1024, 4096, 16384)
BUCKETS = ((0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, None))
PREFIX_LENGTHS = (16, 32, 64, 128, 256, 512, 1024)


def bucket_losses(losses: torch.Tensor) -> dict[str, float]:
    result = {}
    for start, end in BUCKETS:
        selected = losses[:, start:end]
        if selected.numel():
            name = f"{start}_{end - 1}" if end is not None else f"{start}_plus"
            result[name] = float(selected.mean())
    return result


def context_benefit(model: SequenceLM, data: torch.Tensor, seed: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed + 120000)
    inputs, targets = sample_text_batch(data, 2048, 1, generator)
    positions = [1024 + 128 * index for index in range(8)]
    target_values = targets[0, positions]
    metrics = {}
    with torch.no_grad():
        for context in PREFIX_LENGTHS:
            prefixes = torch.stack(
                [inputs[0, position - context + 1 : position + 1] for position in positions]
            )
            logits = model(prefixes)[:, -1]
            loss = F.cross_entropy(logits, target_values)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite character context loss at prefix {context}")
            metrics[str(context)] = float(loss)
    return metrics


def evaluate_char_contexts(
    model: SequenceLM, data: torch.Tensor, seed: int, lengths: list[int], batch_size: int
) -> dict[str, Any]:
    perplexity = {}
    buckets = {}
    with torch.no_grad():
        for length in lengths:
            if model.positions is not None and length > model.train_length:
                continue
            generator = torch.Generator().manual_seed(seed + length + 70000)
            inputs, targets = sample_text_batch(data, length, batch_size, generator)
            logits = model(inputs)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
            ).reshape(batch_size, length)
            if not torch.isfinite(losses).all():
                raise RuntimeError(f"non-finite character evaluation loss at context {length}")
            perplexity[str(length)] = math.exp(float(losses.mean()))
            buckets[str(length)] = bucket_losses(losses)
    return {
        "perplexity": perplexity,
        "position_nll": buckets,
        "context_nll": {} if model.positions is not None else context_benefit(model, data, seed),
    }


def train_char(
    data: torch.Tensor,
    vocabulary_size: int,
    variant: str,
    seed: int,
    steps: int,
    train_length: int,
    batch_size: int,
    window: int,
    sinks: int,
    learning_rate: float = LEARNING_RATE,
    tau0: float = 10.0,
) -> SequenceLM:
    torch.manual_seed(seed)
    model = SequenceLM(vocabulary_size, variant, False, train_length, tau0, 10000.0, window, sinks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=0.1, total_steps=steps)
    generator = torch.Generator().manual_seed(seed + 50000)
    model.train()
    for _ in range(steps):
        inputs, targets = sample_text_batch(data, train_length, batch_size, generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocabulary_size), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite character loss for {variant}/{seed}")
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=500)
    parser.add_argument("--train-length", type=positive_int, default=256)
    parser.add_argument("--window", type=positive_int, default=64)
    parser.add_argument("--sinks", type=int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--eval-batch-size", type=positive_int, default=2)
    parser.add_argument("--lengths", nargs="+", type=positive_int, default=list(LENGTHS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--max-runs", type=positive_int)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path.home() / ".cache" / "zeta-scaling-toolkit" / "tinyshakespeare.txt",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/context-ability-char500.jsonl")
    )
    args = parser.parse_args()
    if (
        args.train_length < 2
        or args.sinks < 0
        or len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
        or len(set(args.lengths)) != len(args.lengths)
        or len(set(args.variants)) != len(args.variants)
    ):
        parser.error("invalid train length, sink count, seeds, or repeated contexts")
    torch.set_num_threads(1)
    text, digest = dataset_info(args.data_path)
    vocabulary = sorted(set(text))
    ids = {character: index for index, character in enumerate(vocabulary)}
    data = torch.tensor([ids[character] for character in text], dtype=torch.long)
    split = int(len(data) * 0.9)
    training, evaluation = data[:split], data[split:]
    if max(*args.lengths, 2048) + 1 >= len(evaluation):
        parser.error("evaluation context exceeds the held-out dataset")

    configuration = {
        "task": "char",
        "steps": args.steps,
        "train_length": args.train_length,
        "window": args.window,
        "sinks": args.sinks,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lengths": args.lengths,
        "dataset_sha256": digest,
    }
    records: list[dict[str, Any]] = []
    if args.output.exists():
        records = [json.loads(line) for line in args.output.read_text().splitlines() if line]
    if any(any(record[key] != value for key, value in configuration.items()) for record in records):
        raise ValueError("output contains a different experiment configuration")
    completed = {(record["variant"], record["seed"]) for record in records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_runs = 0
    with args.output.open("a", encoding="utf-8") as output:
        for variant in args.variants:
            for seed in args.seeds:
                if (variant, seed) in completed:
                    continue
                model = train_char(
                    training,
                    len(vocabulary),
                    variant,
                    seed,
                    args.steps,
                    args.train_length,
                    args.batch_size,
                    args.window,
                    args.sinks,
                )
                metrics = {}
                for mode in ("full", "window"):
                    model.set_attention_evaluation(mode)
                    metrics[mode] = evaluate_char_contexts(
                        model, evaluation, seed, args.lengths, args.eval_batch_size
                    )
                record = {
                    **configuration,
                    "variant": variant,
                    "seed": seed,
                    "metrics": metrics,
                    "environment": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform(),
                        "torch_threads": torch.get_num_threads(),
                    },
                }
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                completed.add((variant, seed))
                print(json.dumps({"variant": variant, "seed": seed, "completed": True}), flush=True)
                new_runs += 1
                if args.max_runs is not None and new_runs >= args.max_runs:
                    return


if __name__ == "__main__":
    main()
