"""Profile dense mask/bias construction against chunked and sparse alternatives."""

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

from zeta_scaling_toolkit import NonCollapsingAttention, alibi_bias, streaming_mask

DEFAULT_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
WINDOW = 64
SINK_TOKENS = 4
HEADS = 4
HEAD_SIZE = 16
CHUNK_SIZE = 128


def dense_mask_reference(
    q_len: int, k_len: int, window: int, sinks: int, device: torch.device
) -> torch.Tensor:
    query_positions = torch.arange(k_len - q_len, k_len, device=device)
    key_positions = torch.arange(k_len, device=device)
    causal = key_positions[None, :] <= query_positions[:, None]
    recent = key_positions[None, :] >= query_positions[:, None] - window + 1
    sink = key_positions[None, :] < sinks
    return causal & (recent | sink)


def dense_alibi_reference(
    heads: int, q_len: int, k_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    query_positions = torch.arange(k_len - q_len, k_len, device=device)
    key_positions = torch.arange(k_len, device=device)
    distances = (query_positions[:, None] - key_positions[None, :]).abs().to(dtype)
    head_ids = torch.arange(1, heads + 1, dtype=dtype, device=device)
    slopes = torch.exp2(-8.0 * head_ids / heads)
    return -slopes[None, :, None, None] * distances[None, None, :, :]


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(20261115)
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    attention: Optional[NonCollapsingAttention] = None
    query = key = value = None
    if args.operation in ("attention", "compare"):
        query = torch.randn(1, HEADS, args.seq_len, HEAD_SIZE, dtype=dtype, device=device)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        attention = NonCollapsingAttention(HEAD_SIZE, length_scaling="log", train_length=64).to(
            device
        )

    def operation() -> torch.Tensor:
        if args.operation == "mask":
            if args.implementation == "reference":
                return dense_mask_reference(args.seq_len, args.seq_len, WINDOW, SINK_TOKENS, device)
            return streaming_mask(
                args.seq_len,
                args.seq_len,
                WINDOW,
                SINK_TOKENS,
                device=device,
                chunk_size=CHUNK_SIZE,
            )
        if args.operation == "alibi":
            if args.implementation == "reference":
                return dense_alibi_reference(HEADS, args.seq_len, args.seq_len, dtype, device)
            return alibi_bias(
                HEADS,
                args.seq_len,
                args.seq_len,
                dtype=dtype,
                device=device,
                chunk_size=CHUNK_SIZE,
            )
        if args.operation == "attention":
            assert (
                attention is not None
                and query is not None
                and key is not None
                and value is not None
            )
            if args.implementation == "dense":
                mask = dense_mask_reference(args.seq_len, args.seq_len, WINDOW, SINK_TOKENS, device)
                bias = dense_alibi_reference(
                    HEADS, args.seq_len, args.seq_len, torch.float32, device
                )
                combined = bias.masked_fill(~mask[None, None], float("-inf"))
                return attention(query, key, value, attn_mask=combined)
            return attention.forward_streaming(
                query,
                key,
                value,
                window=WINDOW,
                sink_tokens=SINK_TOKENS,
                alibi=True,
                chunk_size=CHUNK_SIZE,
            )
        raise ValueError(f"unsupported operation: {args.operation}")

    if args.operation == "compare_mask":
        reference = dense_mask_reference(args.seq_len, args.seq_len, WINDOW, SINK_TOKENS, device)
        chunked = streaming_mask(
            args.seq_len,
            args.seq_len,
            WINDOW,
            SINK_TOKENS,
            device=device,
            chunk_size=CHUNK_SIZE,
        )
        difference = (reference.float() - chunked.float()).abs()
        print(
            json.dumps(
                {
                    "operation": "compare_mask",
                    "seq_len": args.seq_len,
                    "max_abs_error": difference.max().item() if difference.numel() else 0.0,
                    "rmse": difference.square().mean().sqrt().item() if difference.numel() else 0.0,
                }
            )
        )
        return
    if args.operation == "compare_alibi":
        reference = dense_alibi_reference(HEADS, args.seq_len, args.seq_len, dtype, device)
        chunked = alibi_bias(
            HEADS,
            args.seq_len,
            args.seq_len,
            dtype=dtype,
            device=device,
            chunk_size=CHUNK_SIZE,
        )
        difference = (reference.float() - chunked.float()).abs()
        print(
            json.dumps(
                {
                    "operation": "compare_alibi",
                    "seq_len": args.seq_len,
                    "max_abs_error": difference.max().item() if difference.numel() else 0.0,
                    "rmse": difference.square().mean().sqrt().item() if difference.numel() else 0.0,
                }
            )
        )
        return
    if args.operation == "compare":
        assert attention is not None and query is not None and key is not None and value is not None
        mask = dense_mask_reference(args.seq_len, args.seq_len, WINDOW, SINK_TOKENS, device)
        bias = dense_alibi_reference(HEADS, args.seq_len, args.seq_len, torch.float32, device)
        dense = attention(
            query, key, value, attn_mask=bias.masked_fill(~mask[None, None], float("-inf"))
        )
        sparse = attention.forward_streaming(
            query,
            key,
            value,
            window=WINDOW,
            sink_tokens=SINK_TOKENS,
            alibi=True,
            chunk_size=CHUNK_SIZE,
        )
        difference = (dense.float() - sparse.float()).abs()
        print(
            json.dumps(
                {
                    "operation": "compare",
                    "seq_len": args.seq_len,
                    "max_abs_error": difference.max().item(),
                    "rmse": difference.square().mean().sqrt().item(),
                }
            )
        )
        return

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(2):
        result = operation()
        if args.operation == "attention":
            assert attention is not None
            attention.zero_grad(set_to_none=True)
            result.square().mean().backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    elapsed = 0.0
    forward_elapsed = 0.0
    backward_elapsed = 0.0
    for _ in range(args.iterations):
        if args.operation == "attention":
            assert attention is not None
            attention.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = operation()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_elapsed += time.perf_counter() - start
            start = time.perf_counter()
            output.square().mean().backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            backward_elapsed += time.perf_counter() - start
        else:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = operation()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
    result_json: dict[str, Any] = {
        "operation": args.operation,
        "implementation": args.implementation,
        "seq_len": args.seq_len,
        "device": str(device),
        "dtype": args.dtype,
        "peak_memory_mb": peak_memory_mb(device),
    }
    if args.operation == "attention":
        result_json["forward_ms"] = 1000 * forward_elapsed / args.iterations
        result_json["backward_ms"] = 1000 * backward_elapsed / args.iterations
    else:
        result_json["time_ms"] = 1000 * elapsed / args.iterations
    print(json.dumps(result_json, sort_keys=True))


def run_case(
    operation: str,
    implementation: str,
    seq_len: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--operation",
        operation,
        "--implementation",
        implementation,
        "--seq-len",
        str(seq_len),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--iterations",
        str(args.iterations),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--dense-attention-max", type=int, default=2048)
    parser.add_argument("--dense-alibi-max", type=int, default=4096)
    parser.add_argument("--dense-mask-compare-max", type=int, default=8192)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--operation",
        choices=("mask", "alibi", "attention", "compare", "compare_mask", "compare_alibi"),
        default="mask",
    )
    parser.add_argument(
        "--implementation", choices=("reference", "chunked", "dense", "sparse"), default="reference"
    )
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return
    if any(length < 1 for length in args.seq_lengths):
        parser.error("sequence lengths must be positive")
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    args.device = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    results: list[dict[str, Any]] = []
    for seq_len in args.seq_lengths:
        for implementation in ("reference", "chunked"):
            results.append(run_case("mask", implementation, seq_len, args))
        if seq_len <= args.dense_mask_compare_max:
            results.append(run_case("compare_mask", "reference", seq_len, args))
        if seq_len <= args.dense_alibi_max:
            for implementation in ("reference", "chunked"):
                results.append(run_case("alibi", implementation, seq_len, args))
            results.append(run_case("compare_alibi", "reference", seq_len, args))
        for implementation in ("dense", "sparse"):
            if implementation == "dense" and seq_len > args.dense_attention_max:
                continue
            results.append(run_case("attention", implementation, seq_len, args))
        if seq_len <= args.dense_attention_max:
            results.append(run_case("compare", "dense", seq_len, args))

    print(
        f"Environment: Python {__import__('platform').python_version()}, PyTorch {torch.__version__}, "
        f"{platform.platform()}, device={args.device}, dtype={args.dtype}, "
        f"window={WINDOW}, sinks={SINK_TOKENS}, chunk={CHUNK_SIZE}, iterations={args.iterations}"
    )
    print("\nRaw per-case JSON:")
    for result in results:
        print(json.dumps(result, sort_keys=True))
    print(
        "\nDense builders: reference -> chunked (time ms / peak MB; speedup; max abs error / RMSE)"
    )
    for operation in ("mask", "alibi"):
        for seq_len in args.seq_lengths:
            reference = next(
                (
                    r
                    for r in results
                    if r["operation"] == operation
                    and r["implementation"] == "reference"
                    and r["seq_len"] == seq_len
                ),
                None,
            )
            chunked = next(
                (
                    r
                    for r in results
                    if r["operation"] == operation
                    and r["implementation"] == "chunked"
                    and r["seq_len"] == seq_len
                ),
                None,
            )
            if reference is None or chunked is None:
                print(
                    f"| {operation} | {seq_len} | N/A (dense output over configured memory cap) |"
                )
                continue
            time_key = "time_ms"
            comparison = next(
                (
                    row
                    for row in results
                    if row["operation"] == f"compare_{operation}" and row["seq_len"] == seq_len
                ),
                None,
            )
            error = (
                "N/A"
                if comparison is None
                else (f"{comparison['max_abs_error']:.3g} / {comparison['rmse']:.3g}")
            )
            speedup = reference[time_key] / chunked[time_key]
            print(
                f"| {operation} | {seq_len} | "
                f"{reference[time_key]:.3f}/{reference['peak_memory_mb']:.1f} -> "
                f"{chunked[time_key]:.3f}/{chunked['peak_memory_mb']:.1f} | "
                f"{speedup:.2f}x | {error} |"
            )
    print("\nAttention: dense vs sparse (forward+backward ms / peak MB; numerical error)")
    for seq_len in args.seq_lengths:
        dense = next(
            (
                r
                for r in results
                if r["operation"] == "attention"
                and r["implementation"] == "dense"
                and r["seq_len"] == seq_len
            ),
            None,
        )
        sparse = next(
            (
                r
                for r in results
                if r["operation"] == "attention"
                and r["implementation"] == "sparse"
                and r["seq_len"] == seq_len
            ),
            None,
        )
        compare = next(
            (r for r in results if r["operation"] == "compare" and r["seq_len"] == seq_len), None
        )
        if sparse is None:
            continue
        dense_text = (
            "N/A"
            if dense is None
            else f"{dense['forward_ms'] + dense['backward_ms']:.3f}/{dense['peak_memory_mb']:.1f}"
        )
        error_text = (
            "N/A" if compare is None else f"{compare['max_abs_error']:.3g}/{compare['rmse']:.3g}"
        )
        speed = (
            "N/A"
            if dense is None
            else f"{(dense['forward_ms'] + dense['backward_ms']) / (sparse['forward_ms'] + sparse['backward_ms']):.2f}x"
        )
        print(
            f"| {seq_len} | {dense_text} | {sparse['forward_ms'] + sparse['backward_ms']:.3f}/{sparse['peak_memory_mb']:.1f} | {speed} | {error_text} |"
        )


if __name__ == "__main__":
    main()
