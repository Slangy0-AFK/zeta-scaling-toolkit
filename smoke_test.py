"""Smoke tests (replaces root zeta_scaling_toolkit.py, which shadowed the package name)."""

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


def main() -> int:
    torch.manual_seed(0)
    results = []

    attn = NonCollapsingAttention(32)
    k = torch.randn(1, 8, 64, 32)
    results.append(("attention output shape", attn(k, k, k).shape == k.shape))
    unit = F.normalize(k, dim=-1)
    weights = torch.softmax((unit @ unit.transpose(-2, -1)) * attn.tau, -1)
    self_weight = weights.diagonal(dim1=-2, dim2=-1).mean()
    results.append(("aligned q=k is sharply peaked (>0.5 self weight)", bool(self_weight > 0.5)))

    opt = SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = AsymptoticLRScheduler(opt, warmup_steps=5)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    ok = all(math.isfinite(v) and v > 0 for v in lrs) and max(lrs) == 1e-3
    results.append(("LR finite, positive, peaks at base", ok))

    coords = get_scale_invariant_vit_coords(224, 448)
    results.append(
        ("coords radius <= sqrt(2)/2", coords[..., 0].max().item() <= math.sqrt(0.5) + 1e-6)
    )

    x = torch.randn(512, 256)
    n0 = x.norm(dim=1).mean()
    for _ in range(30):
        x = torch.relu(make_corrected_linear(256, 256)(x))
    ratio = (x.norm(dim=1).mean() / n0).item()
    results.append((f"30-layer ReLU norm ratio = {ratio:.3f} in (0.2, 5)", 0.2 < ratio < 5))

    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
