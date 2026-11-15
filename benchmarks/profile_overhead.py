"""Profile a standard MultiheadAttention block against NonCollapsingAttention."""

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from zeta_scaling_toolkit import NonCollapsingAttention


class StandardAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(inputs, inputs, inputs, need_weights=False)
        return output


class ToolkitAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_size = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.attention = NonCollapsingAttention(self.head_size)
        self.projection = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = inputs.shape
        q, k, v = self.qkv(inputs).chunk(3, dim=-1)
        q, k, v = (
            tensor.reshape(batch, sequence, self.heads, self.head_size).transpose(1, 2)
            for tensor in (q, k, v)
        )
        attended = self.attention(q, k, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, self.width)
        return self.projection(attended)


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    block: nn.Module
    if args.backend == "standard":
        block = StandardAttention(args.width, args.heads)
    else:
        block = ToolkitAttention(args.width, args.heads)
    block = block.to(device).train()
    inputs = torch.randn(args.batch_size, args.seq_len, args.width, device=device)

    for _ in range(2):
        block.zero_grad(set_to_none=True)
        block(inputs).square().mean().backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    forward_seconds = 0.0
    backward_seconds = 0.0
    for _ in range(args.iterations):
        block.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        outputs = block(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - start

        loss = outputs.square().mean()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_seconds += time.perf_counter() - start

    print(
        json.dumps(
            {
                "backend": args.backend,
                "seq_len": args.seq_len,
                "device": str(device),
                "forward_ms": 1000 * forward_seconds / args.iterations,
                "backward_ms": 1000 * backward_seconds / args.iterations,
                "peak_memory_mb": peak_memory_mb(device),
            },
            sort_keys=True,
        )
    )


def run_case(
    backend: str,
    seq_len: int,
    device: str,
    batch_size: int,
    width: int,
    heads: int,
    iterations: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--seq-len",
        str(seq_len),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--width",
        str(width),
        "--heads",
        str(heads),
        "--iterations",
        str(iterations),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("standard", "toolkit"), default="standard")
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return
    if any(length < 1 for length in args.seq_lengths):
        parser.error("sequence lengths must be positive")
    if args.batch_size < 1 or args.width < 1 or args.heads < 1:
        parser.error("batch size, width, and heads must be positive")
    if args.width % args.heads:
        parser.error("width must be divisible by heads")
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    results = [
        run_case(backend, seq_len, device, args.batch_size, args.width, args.heads, args.iterations)
        for seq_len in args.seq_lengths
        for backend in ("standard", "toolkit")
    ]
    print(
        f"Environment: PyTorch {torch.__version__}, {platform.platform()}, "
        f"device={device}, batch={args.batch_size}, width={args.width}, "
        f"heads={args.heads}, iterations={args.iterations}"
    )
    print("\nPaste-ready Markdown table:")
    print("| Sequence length | Backend | Forward (ms) | Backward (ms) | Peak memory (MB) |")
    print("|---:|---|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['seq_len']} | {result['backend']} | "
            f"{result['forward_ms']:.3f} | {result['backward_ms']:.3f} | "
            f"{result['peak_memory_mb']:.1f} |"
        )


if __name__ == "__main__":
    main()
