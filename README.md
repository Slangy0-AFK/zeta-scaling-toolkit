# Zeta Scaling Toolkit

Experimental PyTorch scaling utilities. This is an AI-assisted build, originally
created in one day and revised to remove known mathematical errors. It has not
been validated on real training runs or externally validated. The smoke tests
and self-run toy experiment below do not establish training benefits.

## Components

- `NonCollapsingAttention` divides attention logits by
  `sqrt(d_k) * (1 + A_attn / ln(N))`, with `A_attn=1.0` by default.
- `AsymptoticLRScheduler` starts at the base learning rate and scales it by
  `max(ln(2) / ln(T), floor_ratio)`, with a default floor of 1% of the base.
- `get_scale_invariant_vit_coords` produces polar coordinates with radius
  `sqrt((ln(T) + A_coords) / pi)`; `A_coords` defaults to zero.
- `make_corrected_linear` creates a linear layer with Kaiming normal weights
  and zero bias.

These are formulas to evaluate, not guarantees against attention collapse,
training stagnation, or image distortion.

## Installation

```bash
pip install -e .
```

## Quick start

```python
import torch
from zeta_scaling_toolkit import (
    NonCollapsingAttention,
    AsymptoticLRScheduler,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)

attn = NonCollapsingAttention(d_k=64)
Q, K, V = (torch.randn(2, 128, 64) for _ in range(3))
output = attn(Q, K, V)

linear = make_corrected_linear(in_features=1024, out_features=4096)
optimizer = torch.optim.Adam(linear.parameters(), lr=0.001)
scheduler = AsymptoticLRScheduler(optimizer)
optimizer.step()
scheduler.step()

coords = get_scale_invariant_vit_coords(height=224, width=224, T=256)
```

The corrected API replaces the old global `A0`: use `A_attn` for attention,
`floor_ratio` for the scheduler, and `A_coords` for coordinates. Attention
infers `N` from `Q`, and `make_corrected_linear` replaces
`CorrectedVarianceLinear`.

## Testing

```bash
python zeta_scaling_toolkit.py
pytest tests/
```

The first command runs self-verification smoke tests only.

## Toy benchmark (self-run, single seed, not peer-reviewed)

```bash
pip install -e ".[benchmark]"
python benchmarks/toy_copy_task.py
```

The CPU benchmark runs 300 steps of sequence-copy training per transformer,
using identical initial parameters and batches. It compares standard attention
with cosine decay to corrected attention with the asymptotic scheduler, prints
the final batch losses, and writes `loss_curves.png` in the current directory.
It also trains two 30-layer MLPs for 300 steps on an identity regression task
and prints their last hidden-layer activation standard deviations on the same
held-out inputs. Both MLPs use the same seed and training data.

One run used seed 42 and AdamW with a base learning rate of 0.001:

| Variant | Final batch loss | 30-layer MLP activation std |
|---|---:|---:|
| Standard attention + cosine LR / standard linear | 0.0208 | 0.081 |
| Corrected attention + asymptotic LR / corrected linear | 1.167 | 0.300 |

The corrected transformer performed substantially worse on this task. The
comparison changes both attention and scheduler, so it cannot attribute the
difference to either individually. After 300 scheduler steps, the asymptotic
rate is about 0.12 times the base rate; cosine is 0.5 at step 150 and zero
after step 300. Both attention variants use four heads and the same input and
output projection architecture. This is one task and one run, not external
validation. A 2×2 attention/scheduler ablation is a next step.

## Requirements and license

Python >= 3.9 and PyTorch >= 2.0. MIT License; see [LICENSE](LICENSE).
