# Zeta Scaling Toolkit

Experimental PyTorch utilities for attention, learning-rate schedules,
coordinates, and linear initialization. Numerical behavior is covered by the
linked tests; none of these components is validated as a training improvement
on large-scale tasks.

## Components

- **`NonCollapsingAttention`** computes `softmax(tau * cos(q, k)) @ v`, where
  `tau` is learnable, starts at `tau0` (default 10), and is bounded to `[tau_min, tau_max]`
  (defaults 1 and 50) using a logit-sigmoid parameterization. This parameterization
  ensures non-zero gradients throughout the bounded range, allowing the temperature
  parameter to recover if perturbed beyond the bounds. Its normalization and softmax
  calculations use at least float32, and the result is cast to `v.dtype`. Boolean
  masks use `True` to retain a position; floating masks are added to the logits.
  Fully masked query rows return zeros. Shapes, dtypes, devices, and optional
  finite-value checks are validated; an empty key sequence raises `ValueError`,
  while empty query sequences and batches are supported. See [attention property tests](./tests/test_properties.py),
  [edge-case tests](./tests/test_edge_cases.py), and
  [regression tests](./tests/test_hardening_regressions.py).
- **`AsymptoticLRScheduler`** is the retained public class name for a schedule
  that linearly warms up, follows cosine decay to `floor_ratio * base_lr`, then
  stays at that floor. `floor_ratio` must be in `[0, 1]`, `total_steps >= 1`,
  and `0 <= warmup_steps < total_steps`. Without warmup it is mathematically
  equivalent to `CosineAnnealingLR` with the same floor inside the configured
  horizon, up to floating-point rounding; unlike stock cosine it stays clamped
  to the floor beyond that horizon. See
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

The repository's 0.5.1 package version is an unpublished candidate; PyPI
currently provides 0.5.0. See the [0.5.1 release notes](./RELEASE_NOTES_v0.5.1.md).

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
# On another device, move the module too: attention = attention.to(q.device)
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

Run the optional mutation check with `python tools/mutation_check.py`. It
temporarily injects targeted source changes and runs the full test suite for
each one, restoring the source afterward. `KILLED` means a test detected the
injected bug; `SURVIVED` means the suite missed it. This expensive check is not
part of the default CI matrix.

CI runs the CPU test matrix and a nightly Hypothesis profile. CUDA coverage is
skipped when no CUDA device is available. See [TESTING.md](./TESTING.md).
The toolkit includes CUDA-compatible operations and an optional CUDA smoke
test, but CUDA behavior is only covered when that test runs on a CUDA-enabled
host. The Stage 1 context-ability benchmark in
[benchmarks/RESULTS.md](./benchmarks/RESULTS.md) was run on CPU; its GPU
execution and performance have not been validated.
Reviewers can see the passing full GitHub Actions matrix at commit
[719bfb7](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/commit/719bfb73f97e350087cfefa9617f94571a9d02d1).

## Real-task evaluation

The matched-seed char-level language-model benchmark and its results are
documented in [benchmarks/RESULTS.md](./benchmarks/RESULTS.md). The 100-step
table is early-optimisation only. At 500 steps, only the scheduler-only variant
improved loss on all ten matched seeds; attention-only, initialization-only,
and the combined variant had higher loss. At 1,000 steps over 20 seeds, the
scheduler-only variant improved loss on all 20 matched seeds; attention-only
was worse, while initialization-only and all-components paired intervals
included zero. These results apply to the measured model and dataset only.

## Long-stream engineering run

A separate CPU FP32 experiment processed 100,000 contiguous tokens with
SmolLM2-360M-Instruct, using a bounded 64-sink plus 64-recent KV cache. The
runner exited successfully at position 99,999; 56/56 checkpoints and their
provenance/cache state were verified. Throughput was 10.7199 tokens/second,
total runtime was 9,636.7 seconds, and peak process RSS was 3,400.3 MiB.
MathMesh delivered 56/56 telemetry records with no recorded failures or drops;
the Stability observer recorded 56 advisory recommendations. These results
demonstrate operation for this configuration, not improved model quality or
unbounded information retention. There is no matched 100K end-to-end speed
baseline or evidence for larger-model scalability. A physical service outage
during the full run and restoration of the final checkpoint were not tested.

See the [completed run report](./research/zeta100k-fresh-20261117-5bc2985bbe99/FINAL_REPORT.md),
[provenance](./research/zeta100k-fresh-20261117-5bc2985bbe99/PROVENANCE.md), and
[research evidence checklist](./ZETA_RESEARCH_EVIDENCE_CHECKLIST.md) for
measurements and limitations. This is separate from the char-level training
benchmark above.

## Retrieval diagnostics

The original 100K retrieval result was 0/8 and has been audited from saved
evidence; it was not rerun. Two separate short diagnostics returned 0/3 at
8,192 tokens and 0/4 at 16,384 tokens. Their direct positive controls passed
(3/3 and 4/4), as did both negative controls. Each queried fact was absent
from the bounded cache after its question was prefetched. Eviction correlates
with the misses, but these tests do not establish causation or rule out model
and prompt limitations. These retrieval results are distinct from contiguous
stream processing, checkpoint/numerical integrity, and observer delivery;
they are not failures of MathMesh or evidence that the processing pipeline
failed.

See the [full short-retrieval report](./ZETA_FINAL_SHORT_RETRIEVAL_DIAGNOSTICS.md)
and [machine-readable results](./ZETA_FINAL_SHORT_RETRIEVAL_DIAGNOSTICS.json).

## Experiment integrity and reproducibility

The completed 100K experiment is preserved as an original evidence run and
was not rerun during subsequent retrieval testing. The original 100K results
were audited against saved experiment evidence; the 8,192-token and
16,384-token retrieval diagnostics were separate experiments and did not
modify or replace those results. The original run's checkpoints and
provenance remain preserved.

| Experiment | Results | Evidence status |
|---|---|---|
| 100K contiguous stream processing | 100,000 tokens processed through position 99,999; 56/56 checkpoints verified; 10.7199 tokens/second; 9,636.7 seconds total runtime; 3,400.3 MiB peak process RSS. | Completed original run; not rerun during later retrieval testing. |
| Original 100K retrieval | 0/8 exact matches; 8/8 positive controls. | Audited from saved evidence; not rerun or independently reproduced. |
| 8,192-token retrieval diagnostic | 0/3 exact matches; 3/3 positive controls; negative control passed. | Separate diagnostic. |
| 16,384-token retrieval diagnostic | 0/4 exact matches; 4/4 positive controls; negative control passed. | Separate diagnostic. |

These results measure different aspects of the system: the 100K processing
results demonstrate stream processing and checkpoint integrity, while the
retrieval results measure exact-match recall in their respective tests.
There is no matched full-history 100K speed baseline, and the retrieval
diagnostics do not replace the original run.

## What this can and cannot do

- Opt-in log-length temperature scaling compensates for softmax dilution as
  unmasked key count grows. The synthetic measurement and approximation are
  recorded in [benchmarks/RESULTS.md](./benchmarks/RESULTS.md) and tested in
  [test_properties.py](./tests/test_properties.py). This does not establish a
  language-model training benefit. For floating masks, every finite bias counts
  as visible, including large negative padding sentinels. Pass a boolean
  `visibility_mask` alongside the additive `attn_mask` to exclude padding while
  retaining finite biases; the two masks are intersected.
- RoPE and ALiBi do not require a learned absolute-position table. Their
  relative-position and numerical properties are covered in
  [test_properties.py](./tests/test_properties.py); extrapolation quality is
  task-dependent and must be measured.
- Context is not infinite. The optional streaming mask exposes a fixed window
  plus sink tokens and cannot recall information outside that window. Its
  visibility set is tested in [test_properties.py](./tests/test_properties.py).
  The mask alone is dense. `StreamingKVCache` supplies a bounded inference-only
  KV cache when used one token at a time; it detaches cached values and retains
  only sinks plus a recent window. Cache retention and agreement with the dense
  mask are tested in [test_properties.py](./tests/test_properties.py).
- Attention uses SDPA when one scale applies to every row and chunks query rows
  for row-specific scales or fully masked rows. Dense masks and ALiBi biases
  still use quadratic memory when constructed for a full sequence. None of
  these mechanisms provide unbounded recall. `forward_streaming` bounds
  per-chunk forward work, not total autograd memory: saved activations grow
  with query length. Some CPU inputs process up to 16 chunks at once. For
  ALiBi, its default assumes the last broadcast batch
  dimension is heads when there are at least two batch dimensions; set
  `head_axis` explicitly for other layouts, e.g. `head_axis=0` for `(H, Q, D)`.
  The matched-context attention timings and perplexity measurements are in
  [benchmarks/RESULTS.md](./benchmarks/RESULTS.md#windowed-attention-versus-full-causal-attention).
  The context-use experiments and their negative result are summarized in
  [docs/REPORT.md](./docs/REPORT.md).
- RoPE angles are computed in float64 from int64 positions before casting back
  to the input type. Large positions should not be converted directly to
  float16 or bfloat16; the precision behavior is tested in
  [test_properties.py](./tests/test_properties.py). Create position tensors as
  int64 from the start; casting already-rounded floats back cannot restore them.
- Language-model measurements use one 32-wide model and one small dataset; the
  synthetic vision task used 500 steps and is deliberately simple. Validated
  on a 1,000-step character-level LM and a separate 500-step synthetic vision
  task, but not on large-scale foundation models. These results do not
  generalize to large-scale training; see
  [benchmarks/RESULTS.md](./benchmarks/RESULTS.md).

## Known limitations

- The epsilon in normalization makes scale invariance approximate near zero;
  its relative effect becomes small for vector norms around `1e-2` and larger.
  Without additive biases, zero queries produce uniform weights over visible
  keys, not necessarily zero output; small query/key norms can make attention
  much less selective.
- Move attention modules and inputs to the same device with `.to(device)`;
  the learnable temperature is a module parameter and is not moved automatically.
- Attention temperature is bounded to `[tau_min, tau_max]` via a logit-sigmoid
  parameterization; at extreme raw_tau values, sigmoid saturation may produce zero
  gradients, but the parameter can recover for finite perturbations.
- Fixed-temperature configurations (where tau_min == tau_max) are not supported;
  tau bounds must have tau_min < tau_max.
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
  Check per-sample output-to-input norm ratios on representative held-out data,
  and consider normalization layers if individual signals collapse.
- `calibrate_gain` emits a `RuntimeWarning` when its target ratio is not reached
  within 2% (for example, bounded activations such as tanh may not reach 1.0);
  it retains the best-effort parameters when emitting this warning.
- `calibrate_gain` is now transactional: if validation fails on any layer, all
  parameters revert to their original values.
- To resume the scheduler, restore optimizer and scheduler state dicts. If using
  `last_epoch >= 0` directly, each optimizer group must contain its original
  `initial_lr`; the current (possibly decayed) LR is not a safe substitute.
- Nothing here has been validated on large-scale training runs.

## License and support

Python >= 3.9 and PyTorch >= 2.0. MIT License; see [LICENSE](./LICENSE).
Experimental results are not a substitute for evaluating these components on
the intended model, data, and training budget.
