"""Matched-seed synthetic-image classification benchmark for corrected initialization."""

import argparse
import json
import math
import platform
import random
import statistics
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from zeta_scaling_toolkit import get_scale_invariant_vit_coords, make_corrected_linear

IMAGE_SIZE = 16
PATCH_SIZE = 4
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
WIDTH = 32
HEADS = 4
NUM_CLASSES = 10


def linear_layer(in_features: int, out_features: int, corrected: bool, gain: str) -> nn.Linear:
    if corrected:
        return make_corrected_linear(in_features, out_features, nonlinearity=gain)
    return nn.Linear(in_features, out_features)


class TinyVisionTransformer(nn.Module):
    def __init__(self, corrected_init: bool) -> None:
        super().__init__()
        self.patch_embedding = linear_layer(
            PATCH_SIZE * PATCH_SIZE, WIDTH, corrected_init, "linear"
        )
        self.position_projection = linear_layer(3, WIDTH, corrected_init, "linear")
        self.norm_attention = nn.LayerNorm(WIDTH)
        self.qkv = linear_layer(WIDTH, WIDTH * 3, corrected_init, "linear")
        self.attention_projection = linear_layer(WIDTH, WIDTH, corrected_init, "linear")
        self.norm_feedforward = nn.LayerNorm(WIDTH)
        self.feedforward_input = linear_layer(WIDTH, WIDTH * 2, corrected_init, "relu")
        self.feedforward_output = linear_layer(WIDTH * 2, WIDTH, corrected_init, "linear")
        self.classifier = linear_layer(WIDTH, NUM_CLASSES, corrected_init, "linear")
        coordinates = get_scale_invariant_vit_coords(
            GRID_SIZE, GRID_SIZE, smooth_angle=True
        ).reshape(GRID_SIZE * GRID_SIZE, 3)
        self.register_buffer("coordinates", coordinates)
        if corrected_init:
            residual_scale = 1.0 / math.sqrt(2.0)
            with torch.no_grad():
                self.attention_projection.weight.mul_(residual_scale)
                self.feedforward_output.weight.mul_(residual_scale)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        patches = images.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch, GRID_SIZE**2, -1)
        x = self.patch_embedding(patches)
        x = x + self.position_projection(self.coordinates).unsqueeze(0)

        batch, sequence, _ = x.shape
        head_size = WIDTH // HEADS
        q, k, v = self.qkv(self.norm_attention(x)).chunk(3, dim=-1)
        q, k, v = (
            tensor.reshape(batch, sequence, HEADS, head_size).transpose(1, 2)
            for tensor in (q, k, v)
        )
        weights = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(head_size), dim=-1)
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, sequence, WIDTH)
        x = x + self.attention_projection(attended)
        hidden = self.feedforward_input(self.norm_feedforward(x)).relu()
        x = x + self.feedforward_output(hidden)
        return self.classifier(x.mean(dim=1))


def make_dataset(seed: int, samples_per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(NUM_CLASSES).repeat_interleave(samples_per_class)
    images = torch.randn(len(labels), 1, IMAGE_SIZE, IMAGE_SIZE, generator=generator) * 0.05
    for index, label in enumerate(labels.tolist()):
        row, column = divmod(label, GRID_SIZE)
        top, left = row * PATCH_SIZE, column * PATCH_SIZE
        images[index, 0, top : top + PATCH_SIZE, left : left + PATCH_SIZE] += 1.0
    permutation = torch.randperm(len(labels), generator=generator)
    return images[permutation], labels[permutation]


def evaluate(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), 128):
            batch_images = images[start : start + 128].to(device)
            batch_labels = labels[start : start + 128].to(device)
            logits = model(batch_images)
            losses.append(F.cross_entropy(logits, batch_labels).item())
            correct += int((logits.argmax(dim=-1) == batch_labels).sum())
    return {"loss": statistics.mean(losses), "accuracy": correct / len(labels)}


def train_trial(
    corrected_init: bool,
    seed: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    eval_images: torch.Tensor,
    eval_labels: torch.Tensor,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyVisionTransformer(corrected_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed + 50000)
    model.train()
    for _ in range(steps):
        indices = torch.randint(len(train_labels), (batch_size,), generator=generator)
        images = train_images[indices].to(device)
        labels = train_labels[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for corrected_init={corrected_init}, seed={seed}")
        loss.backward()
        optimizer.step()
    return evaluate(model, eval_images, eval_labels, device)


def bootstrap_interval(values: Sequence[float], seed: int = 0) -> tuple[float, float]:
    generator = random.Random(seed)
    means = [statistics.mean(generator.choices(values, k=len(values))) for _ in range(10000)]
    means.sort()
    return means[249], means[9749]


def summarize(
    baseline: Sequence[dict[str, float]], corrected: Sequence[dict[str, float]], metric: str
) -> dict[str, float | int]:
    differences = [candidate[metric] - base[metric] for base, candidate in zip(baseline, corrected)]
    low, high = bootstrap_interval(differences)
    return {
        "baseline_mean": statistics.mean(result[metric] for result in baseline),
        "baseline_sd": statistics.stdev(result[metric] for result in baseline),
        "corrected_mean": statistics.mean(result[metric] for result in corrected),
        "corrected_sd": statistics.stdev(result[metric] for result in corrected),
        "paired_difference_corrected_minus_standard": statistics.mean(differences),
        "paired_ci_low": low,
        "paired_ci_high": high,
        "seeds_better": sum(value < 0 if metric == "loss" else value > 0 for value in differences),
        "seed_count": len(differences),
    }


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--train-per-class", type=positive_int, default=128)
    parser.add_argument("--eval-per-class", type=positive_int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if len(args.seeds) < 10 or len(set(args.seeds)) != len(args.seeds):
        parser.error("provide at least 10 distinct seeds")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(1)

    baseline_results: list[dict[str, float]] = []
    corrected_results: list[dict[str, float]] = []
    for seed in args.seeds:
        train_images, train_labels = make_dataset(seed, args.train_per_class)
        eval_images, eval_labels = make_dataset(seed + 100000, args.eval_per_class)
        baseline = train_trial(
            False,
            seed,
            train_images,
            train_labels,
            eval_images,
            eval_labels,
            args.steps,
            args.batch_size,
            device,
        )
        corrected = train_trial(
            True,
            seed,
            train_images,
            train_labels,
            eval_images,
            eval_labels,
            args.steps,
            args.batch_size,
            device,
        )
        baseline_results.append(baseline)
        corrected_results.append(corrected)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "standard_init": baseline,
                    "corrected_init": corrected,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "metadata": {
            "task": "synthetic 16x16 grayscale image classification by bright 4x4 tile location",
            "steps": args.steps,
            "seeds": args.seeds,
            "batch_size": args.batch_size,
            "train_samples_per_class": args.train_per_class,
            "eval_samples_per_class": args.eval_per_class,
            "device": str(device),
            "python_platform": platform.platform(),
            "torch": torch.__version__,
        },
        "loss": summarize(baseline_results, corrected_results, "loss"),
        "accuracy": summarize(baseline_results, corrected_results, "accuracy"),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
