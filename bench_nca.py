#!/usr/bin/env python3
"""Benchmark NonCollapsingAttention vs standard attention."""

import time
import torch
from zeta_scaling_toolkit import NonCollapsingAttention


def bench(fn, warmup=10, trials=5, device='cpu'):
    """Run a function multiple times and return median time."""
    for _ in range(warmup):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]  # median


def main():
    seq_len, d_k, B = 2048, 64, 1  # Smaller for CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    Q = torch.randn(B, seq_len, d_k, device=device, dtype=dtype)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    attn = NonCollapsingAttention(d_k=d_k).to(device=device, dtype=dtype)

    # Baseline: standard scaled dot-product attention
    def baseline():
        s = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        return torch.matmul(torch.softmax(s, -1), V)

    # Zeta: NonCollapsingAttention
    def zeta():
        return attn(Q, K, V)

    print("Running benchmarks...")
    print(f"Device: {device}, Dtype: {dtype}")
    print()

    # Baseline benchmark
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_base = bench(baseline, device=device)
    if device == 'cuda':
        base_mem = torch.cuda.max_memory_allocated() / 1e9
    else:
        base_mem = None

    # Zeta benchmark
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_zeta = bench(zeta, device=device)
    if device == 'cuda':
        zeta_mem = torch.cuda.max_memory_allocated() / 1e9
    else:
        zeta_mem = None

    # Results
    if base_mem is not None:
        print(f"Baseline: {t_base * 1000:.2f} ms, {base_mem:.2f} GB")
        print(f"Zeta:     {t_zeta * 1000:.2f} ms, {zeta_mem:.2f} GB")
        print()
        print(f"Speedup:  {t_base / t_zeta:.2f}x")
        print(f"Memory:   {(1 - zeta_mem / base_mem) * 100:.1f}% less")
    else:
        print(f"Baseline: {t_base * 1000:.2f} ms")
        print(f"Zeta:     {t_zeta * 1000:.2f} ms")
        print()
        print(f"Speedup:  {t_base / t_zeta:.2f}x")


if __name__ == "__main__":
    main()
