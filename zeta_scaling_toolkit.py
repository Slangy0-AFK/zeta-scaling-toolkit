"""Smoke tests for the corrected toolkit. Not validation on real training runs."""

import math

import torch
import torch.nn.functional as F
from torch.optim import SGD

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)

if __name__ == "__main__":
    torch.manual_seed(0)
    results = []

    Q, K, V = (torch.randn(1, 8, 64, 32) for _ in range(3))
    attn = NonCollapsingAttention(d_k=32)
    out = attn(Q, K, V)
    assert out.shape == (1, 8, 64, 32), out.shape
    N = Q.shape[-2]
    W = 1 + 1.0 / math.log(N)
    logits = Q @ K.transpose(-2, -1) / (math.sqrt(32) * W)
    weight_std = F.softmax(logits, dim=-1).std().item()
    results.append(("attention weights not uniform (std > 1e-4)", weight_std > 1e-4))

    opt = SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = AsymptoticLRScheduler(opt)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    results.append(("LR starts at exactly base_lr", abs(lrs[0] - 1e-3) < 1e-12))
    results.append(
        ("LR finite and positive for 10 steps", all(math.isfinite(x) and x > 0 for x in lrs))
    )
    results.append(
        ("LR monotonically non-increasing", all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1)))
    )

    coords = get_scale_invariant_vit_coords(32, 32, T=1024.0)
    max_abs = coords.abs().max().item()
    results.append((f"coords max_abs = {max_abs:.2f} < 3.0", max_abs < 3.0))

    x = torch.randn(64, 128)
    for _ in range(30):
        x = F.relu(make_corrected_linear(128, 128)(x))
    std = x.std().item()
    results.append((f"30-layer activation std = {std:.3f} in [0.3, 3.0]", 0.3 <= std <= 3.0))

    print("Smoke tests:")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(ok for _, ok in results):
        raise SystemExit(1)
    print("\nAll smoke tests passed. This does not verify improvements in training.")
