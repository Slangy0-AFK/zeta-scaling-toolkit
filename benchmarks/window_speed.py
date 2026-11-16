"""Time causal windowed attention against exact full causal attention on CPU."""

import argparse
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from zeta_scaling_toolkit import NonCollapsingAttention
from zeta_scaling_toolkit.core import _unit


def full_causal(
    attention: NonCollapsingAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Full causal QK-normalized attention, with per-row log scaling."""
    assert attention.train_length is not None
    dtype = (
        torch.float64
        if any(tensor.dtype == torch.float64 for tensor in (q, k, v))
        else torch.float32
    )
    counts = torch.arange(1, q.shape[-2] + 1, device=q.device, dtype=dtype)
    factors = (counts.log() / math.log(attention.train_length)).clamp_min(1.0)
    scale = (attention.tau.to(dtype) * factors).clamp(max=attention.tau_eff_max)
    scaled_q = _unit(q.to(dtype)) * (scale[:, None] * math.sqrt(attention.d_k))
    result = F.scaled_dot_product_attention(
        scaled_q, _unit(k.to(dtype)), v.to(dtype), is_causal=True, dropout_p=0.0
    )
    return result.to(v.dtype)


def time_operation(
    operation: Callable[[], torch.Tensor],
    attention: NonCollapsingAttention,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    repeats: int,
    backward: bool,
) -> list[float]:
    times = []
    for iteration in range(repeats + 1):
        attention.zero_grad(set_to_none=True)
        for tensor in inputs:
            tensor.grad = None
        start = time.perf_counter()
        if backward:
            operation().square().mean().backward()
        else:
            with torch.no_grad():
                operation()
        elapsed = (time.perf_counter() - start) * 1000
        if iteration:
            times.append(elapsed)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 4096])
    parser.add_argument("--windows", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--head-dimension", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in [*args.lengths, *args.windows, args.head_dimension]):
        parser.error("lengths, windows, head dimension, and repeats must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "threads": 1,
        "batch": 1,
        "heads": 4,
        "head_dimension": args.head_dimension,
        "dtype": "float32",
        "sinks": 4,
        "train_length": 64,
        "repeats": args.repeats,
    }
    results: list[dict[str, Any]] = []
    for length in args.lengths:
        q = torch.randn(1, 4, length, args.head_dimension, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        inputs = (q, k, v)
        full = NonCollapsingAttention(args.head_dimension, length_scaling="log", train_length=64)
        full_samples = {
            mode: time_operation(lambda: full_causal(full, *inputs), full, inputs, args.repeats, bw)
            for mode, bw in (("forward_ms", False), ("forward_backward_ms", True))
        }
        full_times = {mode: statistics.median(samples) for mode, samples in full_samples.items()}
        for window in args.windows:
            for scaling in ("none", "log"):
                attention = NonCollapsingAttention(
                    args.head_dimension, length_scaling=scaling, train_length=64
                )

                def operation() -> torch.Tensor:
                    return attention.forward_streaming(*inputs, window, sink_tokens=4)

                samples = {
                    mode: time_operation(operation, attention, inputs, args.repeats, bw)
                    for mode, bw in (("forward_ms", False), ("forward_backward_ms", True))
                }
                times = {mode: statistics.median(values) for mode, values in samples.items()}
                results.append(
                    {
                        "length": length,
                        "window": window,
                        "scaling": scaling,
                        "full_causal_log_ms": full_times,
                        "full_causal_log_samples_ms": full_samples,
                        "window_ms": times,
                        "window_samples_ms": samples,
                    }
                )
    print(json.dumps({"environment": environment, "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
