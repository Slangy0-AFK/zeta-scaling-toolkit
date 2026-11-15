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
validation. The tuning script below also runs a 2×2 attention/scheduler
ablation.

### Tuning and ablation

```bash
python benchmarks/tune_copy_task.py > tuning-results.jsonl
```

This runs all four attention/scheduler combinations and a standard-attention
constant-LR control (E) over five matched seeds,
using the same initial weights and training batches for each configuration.
It prints one JSON record per grid point with per-seed final training-batch
and held-out toy-batch losses and their means. Compare D against C with the
same `learning_rate` and `A_attn`, and A against the other configurations
with the same learning rate. E uses a constant LR of 0.75 times each base LR
by default; use `--constant-lr-ratio` to change it. Compare E with B at the
same `comparison_base_lr` to assess the effect of the schedule. Use `--help`
to change the grid, steps, or seeds.
To use your own task, replace `train_trial` with your model, data, and training
loop, preserving the attention/scheduler switches and seed matching.

`A_attn` changes the attention divisor `1 + A_attn / ln(N)`; zero recovers
ordinary scaled dot-product attention in this benchmark. The scheduler's
`floor_ratio` is a fraction of the optimizer's base learning rate, not an
absolute rate. Its decay depends on scheduler steps, so call `step()` after
each optimizer update. A higher floor becomes effective only once the
logarithmic factor falls below it. With 300 steps, the default 0.01 floor
never activates: the logarithmic factor is still about 0.12. Both the decay
and choice of floor can affect finite-run results and should be tuned for the
task and training duration. In this toy benchmark, `A_attn=1` slightly
increased the mean loss compared with `A_attn=0`; it may degrade performance
on other small tasks and needs task-specific evaluation. The script also
searches base learning rates if supplied via `--learning-rates`. There is no
global `A0` in the current API, and the attention and scheduler do not require
corrected linear initialization; this transformer's linear layers use PyTorch defaults.
These toy results cannot guarantee a particular ranking or real-task benefit.

### Empirical benchmark and status

The following self-run toy copy-task results use 300 steps, batch size 32,
seeds 42–46 and base LR 0.001. The evaluation loss is averaged over one
held-out batch per seed; this is not a real-world benchmark.

| Config | Attention and LR schedule | Settings | Mean evaluation loss |
|---|---|---|---:|
| A | Standard attention + cosine | Base LR 0.001 | 0.01996 |
| B | Standard attention + asymptotic | `floor_ratio=0.75` | 0.00729 |
| C | Corrected attention + cosine | `A_attn=1.0` | 0.02048 |
| D | Corrected attention + asymptotic | `A_attn=1.0`, `floor_ratio=0.75` | 0.00730 |
| E | Standard attention + constant | LR 0.00075 | 0.00734 |

The high-floor scheduler and constant-LR control perform similarly in this
experiment, but this does not establish a general scheduler advantage or
remove the need to tune the learning rate. With the default 0.01 floor,
standard attention plus the asymptotic schedule yielded a mean evaluation
loss of 1.08061; rapid decay may contribute, but the result does not establish
the cause. Corrected attention had slightly higher loss than standard
attention with cosine decay on this task.

Development on this line of research is paused at these results. Standard
attention and established LR schedules are recommended unless the experimental
components have been evaluated for the intended task. To reproduce the
high-floor run, use `floor_ratio=0.75`; it is not a universally recommended
minimum, and higher values require their own evaluation.

## Requirements and license

Python >= 3.9 and PyTorch >= 2.0. MIT License; see [LICENSE](LICENSE).
