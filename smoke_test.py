"""Smoke tests (replaces root zeta_scaling_toolkit.py, which shadowed the package name)."""
import math

import torch
from torch.optim import SGD

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)

if __name__ == "__main__":
    torch.manual_seed(0)
    res = []
    attn = NonCollapsingAttention(32)
    k = torch.randn(1, 8, 64, 32)
    res.append(("attention output shape", attn(k, k, k).shape == k.shape))
    res.append(("aligned q=k is sharply peaked (>0.5 self weight)", torch.softmax((torch.nn.functional.normalize(k, dim=-1) @ torch.nn.functional.normalize(k, dim=-1).transpose(-2, -1)) * attn.tau, -1).diagonal(dim1=-2, dim2=-1).mean() > 0.5))
    opt = SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = AsymptoticLRScheduler(opt, warmup_steps=5)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sched.step()
    res.append(("LR finite, positive, peaks at base", all(math.isfinite(x) and x > 0 for x in lrs) and max(lrs) == 1e-3))
    c = get_scale_invariant_vit_coords(224, 448)
    res.append(("coords radius <= sqrt(2)/2", c[..., 0].max().item() <= math.sqrt(0.5) + 1e-6))
    x = torch.randn(512, 256); n0 = x.norm(dim=1).mean()
    for _ in range(30):
        x = torch.relu(make_corrected_linear(256, 256)(x))
    ratio = (x.norm(dim=1).mean() / n0).item()
    res.append((f"30-layer ReLU norm ratio = {ratio:.3f} in (0.2, 5)", 0.2 < ratio < 5))
    for name, ok in res:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    raise SystemExit(0 if all(ok for _, ok in res) else 1)
