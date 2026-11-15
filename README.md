# Zeta Scaling Toolkit

Experimental PyTorch utilities for attention, learning-rate schedules,
coordinates, and linear initialization. Numerical behavior is covered by the
linked tests; none of these components is validated as a training improvement
on large-scale tasks.

## Components

- **`NonCollapsingAttention`** computes `softmax(tau * cos(q, k)) @ v`, where
  `tau` is learnable, starts at 10, and is clamped to `[tau_min, tau_max]`
  (defaults 1 and 50). Its normalization and softmax calculations use at least
  float32, and the result is cast to `v.dtype`. Boolean masks use `True` to
  retain a position; floating masks are added to the logits. Fully masked query
  rows return zeros. Shapes, dtypes, devices, and optional finite-value checks
  are validated; an empty key sequence raises `ValueError`, while empty query
  sequences and batches are supported. See [attention property tests](./tests/test_properties.py),
  [edge-case tests](./tests/test_edge_cases.py), and
  [regression tests](./tests/test_hardening_regressions.py).
- **`AsymptoticLRScheduler`** is the retained public class name for a schedule
  that linearly warms up, follows cosine decay to `floor_ratio * base_lr`, then
  stays at that floor. `floor_ratio` must be in `[0, 1]`, `total_steps >= 1`,
  and `0 <= warmup_steps < total_steps`. See
  [scheduler property and resume tests](./tests/test_properties.py).
- **`get_isotropic_coords`** returns pixel-centred `(x, y)` coordinates
  normalized by `max(height, width)`. **`get_scale_invariant_vit_coords`**
  returns `(radius, angle)` or, with `smooth_angle=True`,
  `(radius, cos(angle), sin(angle))`. See the
  [coordinate tests](./tests/test_properties.py) and
  [compatibility tests](./tests/test_zeta_toolkit.py).
- **`make_corrected_linear`** applies orthogonal initialization scaled by a
  gain supported by `torch.nn.init.calculate_gain`; its default is ReLU gain.
  GELU and SiLU are not supported as fixed gains. Use **`calibrate_gain`** with
  representative sample data for sequential GELU/SiLU stacks. See
  [initialization properties](./tests/test_properties.py) and
  [calibration regressions](./tests/test_hardening_regressions.py).

Deprecated `A_attn`, `T`, and `A_coords` arguments remain accepted, warn when
non-default values are supplied, and have no effect. Legacy attention
checkpoints containing `q_scale` and `k_scale` load with a warning and retain
the new module's initialized temperature; see
[compatibility tests](./tests/test_zeta_toolkit.py).

## Installation

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import torch
from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    get_isotropic_coords,
    make_corrected_linear,
)

attention = NonCollapsingAttention(d_k=64)
q, k, v = (torch.randn(2, 128, 64) for _ in range(3))
output = attention(q, k, v)

linear = make_corrected_linear(1024, 4096, nonlinearity="relu")
optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
scheduler = AsymptoticLRScheduler(optimizer, total_steps=300, warmup_steps=10)

coords = get_isotropic_coords(height=224, width=448)
```

## Testing and quality checks

`pytest` runs the full deterministic property and regression suite. Coverage can
be checked with:

```bash
pytest --cov=zeta_scaling_toolkit --cov-report=term-missing --cov-fail-under=95
ruff check zeta_scaling_toolkit tests
mypy --strict zeta_scaling_toolkit
```

CI runs the CPU test matrix and a nightly Hypothesis profile. CUDA coverage is
skipped when no CUDA device is available. See [TESTING.md](./TESTING.md).

## Real-task evaluation

The matched-seed char-level language-model benchmark and its results are
documented in [benchmarks/RESULTS.md](./benchmarks/RESULTS.md). Results are
reported as paired differences with bootstrap confidence intervals; no
component is described as beneficial unless the interval excludes zero.

## Known limitations

- The epsilon in normalization makes scale invariance approximate near zero;
  its relative effect becomes small for vector norms around `1e-2` and larger.
- Attention temperature is bounded by `tau_min` and `tau_max`; a clamp can
  produce zero temperature gradients while the parameter is outside that
  interval.
- GELU/SiLU require data-dependent `calibrate_gain`; fixed `calculate_gain`
  does not support them.
- An all-masked attention row returns zeros, not a probability distribution.
- The `(cos(angle), sin(angle))` representation removes the branch cut away
  from the origin; angle remains undefined at the origin.
- For rectangular orthogonal matrices with fewer output than input features,
  norm shrinkage of roughly `sqrt(out_features / in_features)` is structurally
  expected. See [initialization properties](./tests/test_properties.py).
- `calibrate_gain` equalises the whole-batch norm, not each sample's norm. For
  GELU/SiLU in deep stacks without normalisation layers the batch norm can be
  preserved while the median sample's signal collapses (the fixed-seed
  30-layer SiLU characterization test checks for a median per-row ratio below
  0.5). See `test_known_limitation_silu_calibration_*`.
- `calibrate_gain` emits a `RuntimeWarning` when its target ratio is not reached
  within 2% (for example, bounded activations such as tanh cannot reach 1.0).
- Nothing here has been validated on large-scale training runs.

## License and support

Python >= 3.9 and PyTorch >= 2.0. MIT License; see [LICENSE](./LICENSE).
Experimental results are not a substitute for evaluating these components on
the intended model, data, and training budget.
