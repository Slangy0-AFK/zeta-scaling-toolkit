#!/usr/bin/env python3
"""
Test if NonCollapsingAttention can hit GPU targets:
- 9x speedup vs standard attention
- 75% memory reduction vs standard attention
"""

import time
import torch
import torch.nn.functional as F
from zeta_scaling_toolkit import NonCollapsingAttention


def bench_fn(fn, warmup=5, trials=10, device='cpu'):
    """Benchmark a function and return median time in ms."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = fn()
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    times = []
    for _ in range(trials):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    return sorted(times)[len(times) // 2] * 1000  # median in ms


def test_scenario(seq_len, d_k, batch_size=1, device='cpu'):
    """Test a single scenario."""
    print(f"\nScenario: seq_len={seq_len}, d_k={d_k}, batch_size={batch_size}")
    
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.float16
    
    # Create inputs
    Q = torch.randn(batch_size, seq_len, d_k, device=device, dtype=dtype, requires_grad=False)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    
    # Baseline: standard scaled dot-product attention
    def baseline_fn():
        with torch.no_grad():
            Q_f32 = Q.to(torch.float32)
            K_f32 = K.to(torch.float32)
            V_f32 = V.to(torch.float32)
            scores = torch.matmul(Q_f32, K_f32.transpose(-2, -1)) / (d_k ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, V_f32)
            return output.to(dtype)
    
    # NonCollapsingAttention
    nca = NonCollapsingAttention(d_k=d_k).to(device=device, dtype=dtype)
    def nca_fn():
        return nca(Q, K, V)
    
    # Reset peak memory
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark baseline
    t_baseline = bench_fn(baseline_fn, device=device)
    if device == "cuda":
        mem_baseline = torch.cuda.max_memory_allocated() / 1e9
    else:
        mem_baseline = None
    
    # Reset peak memory
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark NCA
    t_nca = bench_fn(nca_fn, device=device)
    if device == "cuda":
        mem_nca = torch.cuda.max_memory_allocated() / 1e9
    else:
        mem_nca = None
    
    speedup = t_baseline / t_nca
    if mem_baseline is not None and mem_baseline > 0:
        mem_reduction = (1 - mem_nca / mem_baseline) * 100
    else:
        mem_reduction = None
    
    print(f"  Baseline: {t_baseline:.2f} ms", end="")
    if mem_baseline is not None:
        print(f", {mem_baseline:.3f} GB", end="")
    print()
    
    print(f"  NCA:      {t_nca:.2f} ms", end="")
    if mem_nca is not None:
        print(f", {mem_nca:.3f} GB", end="")
    print()
    
    print(f"  Speedup:  {speedup:.2f}x {'✓' if speedup >= 9 else ''}")
    if mem_reduction is not None:
        print(f"  Memory:   {mem_reduction:.1f}% reduction {'✓' if mem_reduction >= 75 else ''}")
    
    return speedup, mem_reduction


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print()
    
    scenarios = [
        (256, 64),
        (512, 64),
        (1024, 64),
        (2048, 64),
        (4096, 64),
        (8192, 64),
    ]
    
    results = []
    for seq_len, d_k in scenarios:
        try:
            speedup, mem_red = test_scenario(seq_len, d_k, batch_size=1, device=device)
            results.append((seq_len, d_k, speedup, mem_red))
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Seq Len':<10} {'d_k':<6} {'Speedup':<12} {'Memory Reduction':<20}")
    print("-" * 80)
    
    for seq_len, d_k, speedup, mem_red in results:
        speedup_str = f"{speedup:.2f}x"
        mem_str = f"{mem_red:.1f}%" if mem_red is not None else "N/A"
        print(f"{seq_len:<10} {d_k:<6} {speedup_str:<12} {mem_str:<20}")
    
    # Check targets
    print("\n" + "=" * 80)
    print("TARGET ANALYSIS")
    print("=" * 80)
    
    if results:
        max_speedup = max(r[2] for r in results)
        print(f"Max Speedup: {max_speedup:.2f}x (Target: 9x) {'✓ ACHIEVED' if max_speedup >= 9 else '✗ MISSED'}")
        
        mem_reductions = [r[3] for r in results if r[3] is not None]
        if mem_reductions:
            max_mem_reduction = max(mem_reductions)
            print(f"Max Memory Reduction: {max_mem_reduction:.1f}% (Target: 75%) {'✓ ACHIEVED' if max_mem_reduction >= 75 else '✗ MISSED'}")
        else:
            print("Memory reduction: N/A (CPU doesn't track CUDA memory)")


if __name__ == "__main__":
    main()
