"""Matched-seed char-level language-model comparison on tiny Shakespeare."""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    make_corrected_linear,
)

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
CONTEXT = 64
MODEL_WIDTH = 32
HEADS = 4
LAYERS = 2
BASE_LR = 1e-3
COMPONENTS = ("baseline", "attention", "scheduler", "init", "all")


class TransformerBlock(nn.Module):
    def __init__(self, corrected_attention: bool, corrected_init: bool, head_size: int):
        super().__init__()
        self.norm_attention = nn.LayerNorm(MODEL_WIDTH)
        self.qkv = (
            make_corrected_linear(MODEL_WIDTH, MODEL_WIDTH * 3, nonlinearity="linear")
            if corrected_init
            else nn.Linear(MODEL_WIDTH, MODEL_WIDTH * 3)
        )
        self.projection = (
            make_corrected_linear(MODEL_WIDTH, MODEL_WIDTH, nonlinearity="linear")
            if corrected_init
            else nn.Linear(MODEL_WIDTH, MODEL_WIDTH)
        )
        self.norm_feedforward = nn.LayerNorm(MODEL_WIDTH)
        feedforward_input = (
            make_corrected_linear(MODEL_WIDTH, MODEL_WIDTH * 2, nonlinearity="relu")
            if corrected_init
            else nn.Linear(MODEL_WIDTH, MODEL_WIDTH * 2)
        )
        feedforward_output = (
            make_corrected_linear(MODEL_WIDTH * 2, MODEL_WIDTH, nonlinearity="linear")
            if corrected_init
            else nn.Linear(MODEL_WIDTH * 2, MODEL_WIDTH)
        )
        self.feedforward = nn.Sequential(
            feedforward_input,
            nn.ReLU(),
            feedforward_output,
        )
        self.attention = NonCollapsingAttention(head_size) if corrected_attention else None
        if corrected_init:
            residual_scale = 1.0 / math.sqrt(2 * LAYERS)
            with torch.no_grad():
                self.projection.weight.mul_(residual_scale)
                feedforward_output.weight.mul_(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        q, k, v = self.qkv(self.norm_attention(x)).chunk(3, dim=-1)
        head_size = MODEL_WIDTH // HEADS
        q, k, v = (
            tensor.reshape(batch, sequence, HEADS, head_size).transpose(1, 2)
            for tensor in (q, k, v)
        )
        causal = torch.ones(sequence, sequence, dtype=torch.bool, device=x.device).tril()
        if self.attention is not None:
            attended = self.attention(q, k, v, attn_mask=causal)
        else:
            logits = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
            weights = logits.masked_fill(~causal, float("-inf")).softmax(dim=-1)
            attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, MODEL_WIDTH)
        x = x + self.projection(attended)
        return x + self.feedforward(self.norm_feedforward(x))


class TinyCharLM(nn.Module):
    def __init__(self, vocabulary_size: int, corrected_attention: bool, corrected_init: bool):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, MODEL_WIDTH)
        self.position_embedding = nn.Embedding(CONTEXT, MODEL_WIDTH)
        self.blocks = nn.Sequential(
            *(
                TransformerBlock(corrected_attention, corrected_init, MODEL_WIDTH // HEADS)
                for _ in range(LAYERS)
            )
        )
        self.norm = nn.LayerNorm(MODEL_WIDTH)
        self.output = (
            make_corrected_linear(MODEL_WIDTH, vocabulary_size, nonlinearity="linear")
            if corrected_init
            else nn.Linear(MODEL_WIDTH, vocabulary_size)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.output(self.norm(self.blocks(x)))


def load_text(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8")


def cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.uname().processor or "unknown"


def sample_batches(
    data: torch.Tensor, seed: int, count: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(data) - CONTEXT - 1, (count, batch_size), generator=generator)
    offsets = torch.arange(CONTEXT + 1)
    chunks = data[starts[..., None] + offsets]
    return chunks[..., :-1].contiguous(), chunks[..., 1:].contiguous()


def train_trial(
    component: str,
    seed: int,
    train_batches: tuple[torch.Tensor, torch.Tensor],
    evaluation_batches: tuple[torch.Tensor, torch.Tensor],
    vocabulary_size: int,
    steps: int,
    floor_ratio: float,
    device: torch.device,
) -> dict[str, float]:
    corrected_attention = component in ("attention", "all")
    corrected_init = component in ("init", "all")
    torch.manual_seed(seed)
    model = TinyCharLM(vocabulary_size, corrected_attention, corrected_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    scheduler = (
        AsymptoticLRScheduler(optimizer, floor_ratio=floor_ratio, total_steps=steps)
        if component in ("scheduler", "all")
        else CosineAnnealingLR(optimizer, T_max=steps)
    )

    model.train()
    inputs, targets = (tensor.to(device) for tensor in train_batches)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs[step])
        loss = F.cross_entropy(logits.reshape(-1, vocabulary_size), targets[step].reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss for component={component}, seed={seed}")
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    evaluation_inputs, evaluation_targets = evaluation_batches
    with torch.no_grad():
        for batch_inputs, batch_targets in zip(evaluation_inputs, evaluation_targets):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_inputs)
            losses.append(
                F.cross_entropy(
                    logits.reshape(-1, vocabulary_size), batch_targets.reshape(-1)
                ).item()
            )
            correct += (logits.argmax(dim=-1) == batch_targets).sum().item()
            total += batch_targets.numel()
    return {
        "evaluation_loss": statistics.mean(losses),
        "evaluation_accuracy": correct / total,
    }


def bootstrap_interval(values: Sequence[float], seed: int = 0) -> tuple[float, float]:
    generator = random.Random(seed)
    means = [statistics.mean(generator.choices(values, k=len(values))) for _ in range(10000)]
    means.sort()
    return means[249], means[9749]


def paired_summary(
    baseline: Sequence[dict[str, float]],
    candidate: Sequence[dict[str, float]],
    metric: str,
) -> dict[str, float]:
    differences = [
        candidate_result[metric] - baseline_result[metric]
        for baseline_result, candidate_result in zip(baseline, candidate)
    ]
    low, high = bootstrap_interval(differences)
    return {
        "mean_difference_candidate_minus_baseline": statistics.mean(differences),
        "std_difference": statistics.stdev(differences),
        "paired_95pct_bootstrap_ci_low": low,
        "paired_95pct_bootstrap_ci_high": high,
    }


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(
    default_steps: int = 100,
    default_seeds: Optional[Sequence[int]] = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=default_steps)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--eval-batches", type=positive_int, default=16)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(default_seeds or range(42, 52))
    )
    parser.add_argument("--floor-ratio", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path.home() / ".cache" / "zeta-scaling-toolkit" / "tinyshakespeare.txt",
    )
    args = parser.parse_args()
    if len(args.seeds) < 10:
        parser.error("at least 10 seeds are required")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if not 0.0 <= args.floor_ratio <= 1.0 or not math.isfinite(args.floor_ratio):
        parser.error("floor ratio must be finite and in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested but CUDA is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(1)

    text = load_text(args.data_path)
    vocabulary = sorted(set(text))
    character_ids = {character: index for index, character in enumerate(vocabulary)}
    encoded = torch.tensor([character_ids[character] for character in text], dtype=torch.long)
    split = int(len(encoded) * 0.9)
    train_data, evaluation_data = encoded[:split], encoded[split:]
    results: dict[str, list[dict[str, float]]] = {name: [] for name in COMPONENTS}

    for seed in args.seeds:
        train_batches = sample_batches(train_data, seed, args.steps, args.batch_size)
        evaluation_batches = sample_batches(
            evaluation_data, seed + 100000, args.eval_batches, args.batch_size
        )
        for component in COMPONENTS:
            result = train_trial(
                component,
                seed,
                train_batches,
                evaluation_batches,
                len(vocabulary),
                args.steps,
                args.floor_ratio,
                device,
            )
            results[component].append(result)
            print(
                json.dumps(
                    {"component": component, "seed": seed, **result},
                    allow_nan=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {
        "metadata": {
            "dataset_url": DATA_URL,
            "dataset_path": str(args.data_path),
            "dataset_sha256": hashlib.sha256(args.data_path.read_bytes()).hexdigest(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_model": cpu_model_name(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "seeds": args.seeds,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "context": CONTEXT,
            "eval_batches": args.eval_batches,
            "base_lr": BASE_LR,
            "floor_ratio": args.floor_ratio,
        },
        "results": {},
    }
    baseline = results["baseline"]
    for component, observations in results.items():
        summary["results"][component] = {}
        for metric in ("evaluation_loss", "evaluation_accuracy"):
            values = [result[metric] for result in observations]
            low, high = bootstrap_interval(values)
            summary["results"][component][metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
                "bootstrap_95pct_ci_low": low,
                "bootstrap_95pct_ci_high": high,
            }
        if component != "baseline":
            summary["results"][component]["paired_difference_vs_baseline"] = {
                metric: paired_summary(baseline, observations, metric)
                for metric in ("evaluation_loss", "evaluation_accuracy")
            }
    print(json.dumps(summary, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
