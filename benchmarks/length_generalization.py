"""Resumable matched-seed evaluation for length extrapolation experiments."""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    make_corrected_linear,
    rotary_embedding,
)

if __package__:
    from .window_speed import full_causal
else:
    from window_speed import full_causal

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
WINDOW_VARIANTS = {
    "rope-window64": (64, "none"),
    "rope-log-window64": (64, "log"),
    "rope-window256": (256, "none"),
    "rope-log-window256": (256, "log"),
}
DEFAULT_VARIANTS = ("abs", "rope", "rope-log", "alibi", "rope-log-stream")
ABILITY_VARIANTS = (
    "rope-full",
    "rope-window",
    "rope-log-full",
    "rope-log-window",
    "rope-linear",
    "rope-ntk",
)
VARIANTS = (*DEFAULT_VARIANTS, *WINDOW_VARIANTS, *ABILITY_VARIANTS)
CHAR_LENGTHS = (64, 128, 256, 512, 1024, 4096)
NEEDLE_LENGTHS = (64, 128, 256, 512, 1024, 4096, 16384)
SUCCESSOR_LENGTHS = (64, 128, 256, 512, 1024, 4096)
WIDTH = 32
HEADS = 4
LAYERS = 2
LEARNING_RATE = 1e-3


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def load_text(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8")


def dataset_info(path: Path) -> tuple[str, str]:
    text = load_text(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"unexpected tiny Shakespeare SHA-256: {digest}")
    return text, digest


def encode_successors(
    start: int, stop: int, *, least_significant_digit_first: bool = False
) -> torch.Tensor:
    tokens: list[int] = []
    for number in range(start, stop):
        digits = [int(digit) for digit in str(number)]
        if least_significant_digit_first:
            digits.reverse()
        tokens.extend(digits)
        tokens.append(10)
    return torch.tensor(tokens, dtype=torch.long)


def needle_batch(
    batch_size: int, length: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_count = (length - 2) // 2
    keys = torch.randint(0, 1023, (batch_size, pair_count), generator=generator)
    values = torch.randint(1024, 2048, (batch_size, pair_count), generator=generator)
    insertion = torch.randint(0, pair_count, (batch_size,), generator=generator)
    query_keys = torch.randint(0, 1023, (batch_size,), generator=generator)
    targets = torch.randint(1024, 2048, (batch_size,), generator=generator)
    for row in range(batch_size):
        keys[row] = torch.where(keys[row] == query_keys[row], 1023, keys[row])
        keys[row, insertion[row]] = query_keys[row]
        values[row, insertion[row]] = targets[row]
    pairs = torch.stack((keys, values), dim=-1).flatten(1)
    query = torch.stack((torch.full_like(query_keys, 2048), query_keys), dim=-1)
    return torch.cat((pairs, query), dim=-1), targets


def windowed_dot_product(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int, sinks: int
) -> torch.Tensor:
    length = q.shape[-2]
    sink_end = min(sinks, length)
    outputs = []
    for start in range(0, length, 128):
        end = min(start + 128, length)
        recent_start = max(sink_end, start - window + 1)
        recent_end = max(recent_start, end)
        positions = torch.cat(
            (
                torch.arange(sink_end, device=q.device),
                torch.arange(recent_start, recent_end, device=q.device),
            )
        )
        queries = torch.arange(start, end, device=q.device)
        keep = (positions[None, :] <= queries[:, None]) & (
            (positions[None, :] < sinks) | (positions[None, :] >= queries[:, None] - window + 1)
        )
        keys = torch.cat((k[..., :sink_end, :], k[..., recent_start:recent_end, :]), dim=-2)
        values = torch.cat((v[..., :sink_end, :], v[..., recent_start:recent_end, :]), dim=-2)
        outputs.append(
            F.scaled_dot_product_attention(
                q[..., start:end, :], keys, values, attn_mask=keep, dropout_p=0.0
            )
        )
    return torch.cat(outputs, dim=-2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        variant: str,
        corrected_init: bool,
        train_length: int,
        tau0: float,
        rope_base: float,
        window: int = 64,
        sinks: int = 4,
        rope_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.window = window
        self.sinks = sinks
        self.rope_scale = rope_scale
        self.norm_attention = nn.LayerNorm(WIDTH)
        self.qkv = self._linear(WIDTH, WIDTH * 3, corrected_init, "linear")
        self.projection = self._linear(WIDTH, WIDTH, corrected_init, "linear")
        self.norm_feedforward = nn.LayerNorm(WIDTH)
        self.feedforward_input = self._linear(WIDTH, WIDTH * 2, corrected_init, "relu")
        self.feedforward_output = self._linear(WIDTH * 2, WIDTH, corrected_init, "linear")
        self.attention = None
        if (
            variant
            in (
                "rope-log",
                "rope-log-stream",
                "rope-log-full",
                "rope-log-window",
            )
            or variant in WINDOW_VARIANTS
        ):
            scaling = WINDOW_VARIANTS[variant][1] if variant in WINDOW_VARIANTS else "log"
            self.attention = NonCollapsingAttention(
                WIDTH // HEADS,
                tau0=tau0,
                length_scaling=scaling,
                train_length=train_length,
            )
        self.rope_base = rope_base
        if corrected_init:
            residual_scale = 1.0 / math.sqrt(2 * LAYERS)
            with torch.no_grad():
                self.projection.weight.mul_(residual_scale)
                self.feedforward_output.weight.mul_(residual_scale)

    @staticmethod
    def _linear(in_features: int, out_features: int, corrected: bool, gain: str) -> nn.Linear:
        if corrected:
            return make_corrected_linear(in_features, out_features, nonlinearity=gain)
        return nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        q, k, v = self.qkv(self.norm_attention(x)).chunk(3, dim=-1)
        head_size = WIDTH // HEADS
        q, k, v = (
            tensor.reshape(batch, sequence, HEADS, head_size).transpose(1, 2)
            for tensor in (q, k, v)
        )
        if self.variant in ("rope", "rope-log", "rope-log-stream", *ABILITY_VARIANTS) or (
            self.variant in WINDOW_VARIANTS
        ):
            positions = torch.arange(sequence, dtype=torch.int64, device=x.device)
            scaling = {"rope-linear": "linear", "rope-ntk": "ntk"}.get(self.variant)
            q = rotary_embedding(
                q, positions, base=self.rope_base, scaling=scaling, scale_factor=self.rope_scale
            )
            k = rotary_embedding(
                k, positions, base=self.rope_base, scaling=scaling, scale_factor=self.rope_scale
            )

        if self.variant in ("rope-full", "rope-linear", "rope-ntk"):
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        elif self.variant == "rope-window":
            attended = windowed_dot_product(q, k, v, self.window, self.sinks)
        elif self.variant == "rope-log-window":
            assert self.attention is not None
            attended = self.attention.forward_streaming(q, k, v, self.window, self.sinks)
        elif self.variant == "rope-log-full":
            assert self.attention is not None
            attended = full_causal(self.attention, q, k, v)
        elif self.variant in WINDOW_VARIANTS:
            assert self.attention is not None
            attended = self.attention.forward_streaming(
                q, k, v, window=WINDOW_VARIANTS[self.variant][0], sink_tokens=4
            )
        elif self.variant == "rope-log" and not self.training:
            assert self.attention is not None
            attended = full_causal(self.attention, q, k, v)
        else:
            outputs = []
            key_positions = torch.arange(sequence, device=x.device)
            for start in range(0, sequence, 128):
                end = min(start + 128, sequence)
                query_positions = torch.arange(start, end, device=x.device)
                mask = key_positions[None, :] <= query_positions[:, None]
                if self.variant == "rope-log-stream":
                    mask &= (key_positions[None, :] < 4) | (
                        key_positions[None, :] >= query_positions[:, None] - 63
                    )
                query = q[..., start:end, :]
                if self.attention is not None:
                    output = self.attention(query, k, v, attn_mask=mask)
                else:
                    logits = (query @ k.transpose(-2, -1)) / math.sqrt(head_size)
                    if self.variant == "alibi":
                        heads = torch.arange(1, HEADS + 1, device=x.device, dtype=logits.dtype)
                        slopes = torch.exp2(-8.0 * heads / HEADS)
                        distances = (query_positions[:, None] - key_positions[None, :]).abs()
                        logits = logits - slopes[None, :, None, None] * distances
                    output = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1) @ v
                outputs.append(output)
            attended = torch.cat(outputs, dim=-2)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, WIDTH)
        x = x + self.projection(attended)
        hidden = self.feedforward_input(self.norm_feedforward(x)).relu()
        return x + self.feedforward_output(hidden)

    def forward_last_rope_full(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        q, k, v = self.qkv(self.norm_attention(x)).chunk(3, dim=-1)
        head_size = WIDTH // HEADS
        q, k, v = (
            tensor.reshape(batch, sequence, HEADS, head_size).transpose(1, 2)
            for tensor in (q, k, v)
        )
        positions = torch.arange(sequence, dtype=torch.int64, device=x.device)
        q = rotary_embedding(q[..., -1:, :], positions[-1:], base=self.rope_base)
        k = rotary_embedding(k, positions, base=self.rope_base)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attended = attended.transpose(1, 2).contiguous().view(batch, 1, WIDTH)
        x = x[:, -1:] + self.projection(attended)
        hidden = self.feedforward_input(self.norm_feedforward(x)).relu()
        return x + self.feedforward_output(hidden)


class SequenceLM(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        variant: str,
        corrected_init: bool,
        train_length: int,
        tau0: float,
        rope_base: float,
        window: int = 64,
        sinks: int = 4,
        rope_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.train_length = train_length
        self.tokens = nn.Embedding(vocabulary_size, WIDTH)
        self.positions = nn.Embedding(train_length, WIDTH) if variant == "abs" else None
        self.blocks = nn.Sequential(
            *(
                TransformerBlock(
                    variant,
                    corrected_init,
                    train_length,
                    tau0,
                    rope_base,
                    window,
                    sinks,
                    rope_scale,
                )
                for _ in range(LAYERS)
            )
        )
        self.norm = nn.LayerNorm(WIDTH)
        self.output = (
            make_corrected_linear(WIDTH, vocabulary_size, nonlinearity="linear")
            if corrected_init
            else nn.Linear(WIDTH, vocabulary_size)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.positions is not None and tokens.shape[1] > self.train_length:
            raise ValueError("absolute-position model cannot exceed its trained position table")
        x = self.tokens(tokens)
        if self.positions is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            x = x + self.positions(positions)
        return self.output(self.norm(self.blocks(x)))

    def forward_last(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.variant != "rope-full":
            return self(tokens)[:, -1]
        x = self.tokens(tokens)
        for block in list(self.blocks)[:-1]:
            x = block(x)
        x = self.blocks[-1].forward_last_rope_full(x)
        return self.output(self.norm(x))[:, 0]

    def set_window_evaluation_variant(self, variant: str) -> None:
        if self.training or variant not in ("rope-log", *WINDOW_VARIANTS):
            raise ValueError("window evaluation requires an eval-mode RoPE + log model")
        for block in self.blocks:
            if block.attention is None:
                raise ValueError("window evaluation requires QK-normalized attention")
            block.variant = variant
            block.attention.length_scaling = (
                WINDOW_VARIANTS[variant][1] if variant in WINDOW_VARIANTS else "log"
            )
        self.variant = variant

    def set_attention_evaluation(self, mode: str) -> None:
        if self.training or mode not in ("full", "window"):
            raise ValueError("attention intervention requires evaluation mode and full or window")
        for block in self.blocks:
            if block.variant not in (
                "rope-full",
                "rope-window",
                "rope-log-full",
                "rope-log-window",
            ):
                raise ValueError("attention intervention requires a Stage 1 RoPE model")
            block.variant = ("rope-log-" if block.attention is not None else "rope-") + mode
        self.variant = self.blocks[0].variant


def sample_text_batch(
    data: torch.Tensor, context: int, batch_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - context - 1, (batch_size,), generator=generator)
    offsets = torch.arange(context + 1)
    samples = data[starts[:, None] + offsets]
    return samples[:, :-1], samples[:, 1:]


def train_trial(
    task: str,
    train_data: torch.Tensor,
    vocabulary_size: int,
    variant: str,
    init_mode: str,
    scheduler_kind: str,
    steps: int,
    seed: int,
    tau0: float,
    rope_base: float,
) -> SequenceLM:
    train_length = 64 if task in ("char", "needle") else 256
    torch.manual_seed(seed)
    model = SequenceLM(
        vocabulary_size,
        variant,
        corrected_init=init_mode == "corrected",
        train_length=train_length,
        tau0=tau0,
        rope_base=rope_base,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    if scheduler_kind == "toolkit":
        scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=0.1, total_steps=steps)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=0.1 * LEARNING_RATE
        )
    generator = torch.Generator().manual_seed(seed + 50000)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if task == "char":
            inputs, targets = sample_text_batch(train_data, 64, 32, generator)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, vocabulary_size), targets.reshape(-1))
        elif task == "needle":
            inputs, targets = needle_batch(16, 64, generator)
            loss = F.cross_entropy(model(inputs)[:, -1], targets)
        else:
            inputs, targets = sample_text_batch(train_data, 256, 16, generator)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, vocabulary_size), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for {task}/{variant}/{seed}")
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model.eval()


def evaluate_char(
    model: SequenceLM, data: torch.Tensor, seed: int, lengths: Sequence[int] = CHAR_LENGTHS
) -> dict[str, Optional[float]]:
    metrics: dict[str, Optional[float]] = {}
    for context in lengths:
        if model.variant == "abs" and context > model.train_length:
            metrics[str(context)] = None
            continue
        generator = torch.Generator().manual_seed(seed + context + 70000)
        inputs, targets = sample_text_batch(data, context, 2, generator)
        with torch.no_grad():
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        metrics[str(context)] = math.exp(float(loss))
    return metrics


def evaluate_needle(model: SequenceLM, seed: int, samples: int = 16) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for length in NEEDLE_LENGTHS:
        if model.variant == "abs" and length > model.train_length:
            continue
        generator = torch.Generator().manual_seed(seed + length + 90000)
        correct = 0
        for _ in range(samples):
            inputs, target = needle_batch(1, length, generator)
            with torch.no_grad():
                prediction = model(inputs)[:, -1].argmax(dim=-1)
            correct += int(prediction.item() == target.item())
        metrics[str(length)] = correct / samples
    return metrics


def evaluate_successor(model: SequenceLM, seed: int) -> dict[str, float]:
    stream = encode_successors(1000, 3000)
    metrics: dict[str, float] = {}
    for length in SUCCESSOR_LENGTHS:
        if model.variant == "abs" and length > model.train_length:
            continue
        generator = torch.Generator().manual_seed(seed + length + 110000)
        starts = torch.randint(0, len(stream) - length - 1, (8,), generator=generator)
        correct_by_bucket = [0, 0, 0, 0]
        total_by_bucket = [0, 0, 0, 0]
        for start in starts.tolist():
            inputs = stream[start : start + length].unsqueeze(0)
            targets = stream[start + 1 : start + length + 1]
            with torch.no_grad():
                predictions = model(inputs)[0].argmax(dim=-1)
            for position in range(length):
                bucket = min(position // 256, 3)
                correct_by_bucket[bucket] += int(predictions[position] == targets[position])
                total_by_bucket[bucket] += 1
        for bucket, name in enumerate(("0_255", "256_511", "512_767", "768_plus")):
            if total_by_bucket[bucket]:
                metrics[f"{length}:{name}"] = correct_by_bucket[bucket] / total_by_bucket[bucket]
    return metrics


def bootstrap_interval(values: Sequence[float], seed: int = 0) -> tuple[float, float]:
    generator = random.Random(seed)
    means = [statistics.mean(generator.choices(values, k=len(values))) for _ in range(10000)]
    means.sort()
    return means[249], means[9749]


def make_summaries(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            record["task"],
            record["variant"],
            record["init_mode"],
            record["scheduler"],
            record["steps"],
            record["seed"],
            record["tau0"],
            record["rope_base"],
        ): record
        for record in records
    }
    summaries: list[dict[str, Any]] = []
    for task, variant, init_mode, scheduler, steps, seed, tau0, rope_base in by_key:
        if variant != "abs":
            baseline_key = (task, "abs", init_mode, scheduler, steps, seed, tau0, rope_base)
            comparisons = [("position", baseline_key)]
        else:
            comparisons = []
        if variant == "rope-log":
            comparisons.append(
                (
                    "qk_normalization_and_log_scaling",
                    (task, "rope", init_mode, scheduler, steps, seed, tau0, rope_base),
                )
            )
        if variant == "rope-log-stream":
            comparisons.append(
                (
                    "streaming_mask",
                    (task, "rope-log", init_mode, scheduler, steps, seed, tau0, rope_base),
                )
            )
        if variant in WINDOW_VARIANTS:
            comparisons.append(
                (
                    "window_vs_full_rope_log",
                    (task, "rope-log", init_mode, scheduler, steps, seed, tau0, rope_base),
                )
            )
        if init_mode == "corrected":
            comparisons.append(
                ("init", (task, variant, "baseline", scheduler, steps, seed, tau0, rope_base))
            )
        if scheduler == "toolkit":
            comparisons.append(
                ("scheduler", (task, variant, init_mode, "cosine", steps, seed, tau0, rope_base))
            )
        current = by_key[(task, variant, init_mode, scheduler, steps, seed, tau0, rope_base)]
        for comparison, baseline_key in comparisons:
            baseline = by_key.get(baseline_key)
            if baseline is None:
                continue
            metrics = set(current["metrics"]) & set(baseline["metrics"])
            for metric in metrics:
                current_value = current["metrics"][metric]
                baseline_value = baseline["metrics"][metric]
                if current_value is None or baseline_value is None:
                    continue
                summaries.append(
                    {
                        "task": task,
                        "comparison": comparison,
                        "variant": variant,
                        "init_mode": init_mode,
                        "scheduler": scheduler,
                        "steps": steps,
                        "tau0": tau0,
                        "rope_base": rope_base,
                        "seed": seed,
                        "metric": metric,
                        "difference": current_value - baseline_value,
                    }
                )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for summary in summaries:
        key = tuple(
            summary[field]
            for field in (
                "task",
                "comparison",
                "variant",
                "init_mode",
                "scheduler",
                "steps",
                "tau0",
                "rope_base",
                "metric",
            )
        )
        grouped.setdefault(key, []).append(summary)
    results = []
    for key, rows in grouped.items():
        if len(rows) < 10:
            continue
        differences = [row["difference"] for row in rows]
        low, high = bootstrap_interval(differences)
        task, comparison, variant, init_mode, scheduler, steps, tau0, rope_base, metric = key
        results.append(
            {
                "task": task,
                "comparison": comparison,
                "variant": variant,
                "init_mode": init_mode,
                "scheduler": scheduler,
                "steps": steps,
                "tau0": tau0,
                "rope_base": rope_base,
                "metric": metric,
                "n": len(rows),
                "mean_difference": statistics.mean(differences),
                "std_difference": statistics.stdev(differences) if len(differences) > 1 else 0.0,
                "paired_95pct_bootstrap_ci": [low, high],
                "seeds_better": sum(
                    value < 0 if task == "char" else value > 0 for value in differences
                ),
            }
        )
    return results


def make_measurements(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[tuple[int, float]]] = {}
    for record in records:
        for metric, value in record["metrics"].items():
            if value is None:
                continue
            key = (
                record["task"],
                record["variant"],
                record["init_mode"],
                record["scheduler"],
                record["steps"],
                record["tau0"],
                record["rope_base"],
                metric,
            )
            grouped.setdefault(key, []).append((record["seed"], value))

    measurements = []
    for key, observations in grouped.items():
        if len(observations) < 10:
            continue
        values = [value for _, value in observations]
        low, high = bootstrap_interval(values)
        task, variant, init_mode, scheduler, steps, tau0, rope_base, metric = key
        measurements.append(
            {
                "task": task,
                "variant": variant,
                "init_mode": init_mode,
                "scheduler": scheduler,
                "steps": steps,
                "tau0": tau0,
                "rope_base": rope_base,
                "metric": metric,
                "n": len(values),
                "seeds": sorted(seed for seed, _ in observations),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "bootstrap_95pct_ci": [low, high],
            }
        )
    return measurements


def run_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record[field]
        for field in (
            "task",
            "variant",
            "init_mode",
            "scheduler",
            "steps",
            "seed",
            "tau0",
            "rope_base",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("char", "needle", "successor"), default="char")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument("--steps", nargs="+", type=positive_int, default=[500, 2000])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument(
        "--init-modes",
        nargs="+",
        choices=("baseline", "corrected"),
        default=["baseline", "corrected"],
    )
    parser.add_argument(
        "--schedulers", nargs="+", choices=("toolkit", "cosine"), default=["toolkit", "cosine"]
    )
    parser.add_argument("--tau0", type=float, default=10.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument(
        "--sweep", action="store_true", help="run the separate 100-109 hyperparameter grid"
    )
    parser.add_argument(
        "--max-runs", type=positive_int, help="limit newly completed runs for chunked execution"
    )
    parser.add_argument("--eval-samples", type=positive_int, default=16)
    parser.add_argument(
        "--paired-window-eval",
        action="store_true",
        help="train one full RoPE + log model per seed and evaluate each window on those weights",
    )
    parser.add_argument(
        "--char-lengths",
        nargs="+",
        type=positive_int,
        default=list(CHAR_LENGTHS),
        help="character evaluation contexts; use a fresh output file for a different set",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path.home() / ".cache" / "zeta-scaling-toolkit" / "tinyshakespeare.txt",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/length-generalization-results.jsonl")
    )
    args = parser.parse_args()
    if len(args.seeds) < 10 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("provide at least 10 distinct nonnegative seeds")
    if (
        not math.isfinite(args.tau0)
        or args.tau0 <= 0
        or not math.isfinite(args.rope_base)
        or args.rope_base <= 1
    ):
        parser.error("tau0 must be positive finite and rope-base must be finite > 1")
    if args.paired_window_eval and (
        args.task != "char"
        or args.sweep
        or any(variant not in ("rope-log", *WINDOW_VARIANTS) for variant in args.variants)
    ):
        parser.error("--paired-window-eval requires char and only RoPE + log/window variants")
    torch.set_num_threads(1)

    if args.sweep:
        if args.seeds == list(range(42, 52)):
            args.seeds = list(range(100, 110))
        args.variants = ["rope-log", "rope-log-stream"]
        args.init_modes = ["baseline", "corrected"]
        args.schedulers = ["toolkit"]
        sweep_values = [
            (tau, base)
            for tau in (1.0, 2.5, 5.0, 10.0, 20.0)
            for base in (10000.0, 500000.0, 1000000.0)
        ]
    else:
        sweep_values = [(args.tau0, args.rope_base)]

    records: list[dict[str, Any]] = []
    if args.output.exists():
        records = [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line
        ]
    for record in records:
        paired = record.get("training_variant") == "rope-log"
        if paired != args.paired_window_eval:
            raise ValueError(
                "cannot mix paired-window and independently trained runs in one output"
            )
        if record["task"] == "char" and set(record["metrics"]) != {
            str(length) for length in args.char_lengths
        }:
            raise ValueError("existing character records have different evaluation contexts")
    completed = {run_key(record) for record in records}

    text: Optional[str] = None
    digest: Optional[str] = None
    if args.task == "char":
        text, digest = dataset_info(args.data_path)
        vocabulary = sorted(set(text))
        character_ids = {character: index for index, character in enumerate(vocabulary)}
        encoded = torch.tensor([character_ids[character] for character in text], dtype=torch.long)
        split = int(len(encoded) * 0.9)
        train_data, evaluation_data = encoded[:split], encoded[split:]
        vocabulary_size = len(vocabulary)
    elif args.task == "needle":
        train_data = torch.empty(0, dtype=torch.long)
        evaluation_data = train_data
        vocabulary_size = 2049
    else:
        train_data = encode_successors(0, 1000)
        evaluation_data = encode_successors(0, 3000)
        vocabulary_size = 11

    cpu_name = platform.processor() or platform.uname().processor or "unknown"
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": cpu_name,
        "torch_threads": torch.get_num_threads(),
        "dataset_sha256": digest,
    }
    pending = 0
    paired_models: dict[tuple[Any, ...], SequenceLM] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output_file:
        for variant in args.variants:
            for init_mode in args.init_modes:
                for scheduler in args.schedulers:
                    for steps in args.steps:
                        for seed in args.seeds:
                            for tau0, rope_base in sweep_values:
                                key = (
                                    args.task,
                                    variant,
                                    init_mode,
                                    scheduler,
                                    steps,
                                    seed,
                                    tau0,
                                    rope_base,
                                )
                                if key in completed:
                                    continue
                                if args.task == "char":
                                    assert text is not None
                                    model_key = (init_mode, scheduler, steps, seed, tau0, rope_base)
                                    if args.paired_window_eval and model_key in paired_models:
                                        model = paired_models[model_key]
                                    else:
                                        model = train_trial(
                                            args.task,
                                            train_data,
                                            vocabulary_size,
                                            "rope-log" if args.paired_window_eval else variant,
                                            init_mode,
                                            scheduler,
                                            steps,
                                            seed,
                                            tau0,
                                            rope_base,
                                        )
                                        if args.paired_window_eval:
                                            paired_models[model_key] = model
                                    if args.paired_window_eval:
                                        model.set_window_evaluation_variant(variant)
                                    metrics = evaluate_char(
                                        model, evaluation_data, seed, args.char_lengths
                                    )
                                elif args.task == "needle":
                                    model = train_trial(
                                        args.task,
                                        train_data,
                                        vocabulary_size,
                                        variant,
                                        init_mode,
                                        scheduler,
                                        steps,
                                        seed,
                                        tau0,
                                        rope_base,
                                    )
                                    metrics = evaluate_needle(model, seed, args.eval_samples)
                                else:
                                    model = train_trial(
                                        args.task,
                                        train_data,
                                        vocabulary_size,
                                        variant,
                                        init_mode,
                                        scheduler,
                                        steps,
                                        seed,
                                        tau0,
                                        rope_base,
                                    )
                                    metrics = evaluate_successor(model, seed)
                                record = {
                                    "task": args.task,
                                    "variant": variant,
                                    "init_mode": init_mode,
                                    "scheduler": scheduler,
                                    "steps": steps,
                                    "seed": seed,
                                    "tau0": tau0,
                                    "rope_base": rope_base,
                                    "metrics": metrics,
                                    "environment": environment,
                                }
                                if args.paired_window_eval:
                                    record["training_variant"] = "rope-log"
                                output_file.write(json.dumps(record, sort_keys=True) + "\n")
                                output_file.flush()
                                records.append(record)
                                completed.add(key)
                                pending += 1
                                if args.max_runs is not None and pending >= args.max_runs:
                                    break
                            if args.max_runs is not None and pending >= args.max_runs:
                                break
                        if args.max_runs is not None and pending >= args.max_runs:
                            break
                    if args.max_runs is not None and pending >= args.max_runs:
                        break
                if args.max_runs is not None and pending >= args.max_runs:
                    break
            if args.max_runs is not None and pending >= args.max_runs:
                break

    summaries = make_summaries(records)
    measurements = make_measurements(records)
    print(
        json.dumps(
            {
                "new_runs": pending,
                "total_records": len(records),
                "measurements": measurements,
                "summaries": summaries,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
