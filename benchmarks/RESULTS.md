# Matched-seed tiny Shakespeare evaluation

This is a small, CPU-only char-level language-model comparison. It is an
experiment on one dataset, architecture, and training budget—not evidence that
any component improves other tasks or large-scale training.

## Contents

- [Exact run](#exact-run)
- [Results](#results)
  - [100-step early-optimisation only](#100-step-early-optimisation-only)
  - [500-step matched-seed run](#500-step-matched-seed-run)
  - [1000-step, 20-seed matched run](#1000-step-20-seed-matched-run)
  - [Attention runtime and memory profile](#attention-runtime-and-memory-profile)
  - [Synthetic vision task](#synthetic-vision-task)
- [Sparse attention optimization](#sparse-attention-optimization)
  - [Supporting numerical checks](#supporting-numerical-checks)
- [Exploratory length evaluation](#exploratory-length-evaluation-incomplete-gate)
  - [Exploratory 2000-step paired comparison](#exploratory-2000-step-paired-comparison)
  - [16k-token numerical smoke checks](#16k-token-numerical-smoke-checks)
  - [Exploratory needle retrieval](#exploratory-needle-retrieval-500-steps)
- [Mutation sanity](#mutation-sanity)
- [Extended validation](#extended-validation)
  - [Attention profiling](#attention-profiling)
  - [Synthetic vision](#synthetic-vision)
  - [Tiny Shakespeare at 1000 steps](#tiny-shakespeare-at-1000-steps)

## Exact run

The historical 100- and 500-step results in this section were run at commit
`bc9d44e`, before the benchmark's optional initialization was changed. To
reproduce the historical initialization, run the commands at that revision.

```bash
python benchmarks/real_task.py --steps 100 --seeds 42 43 44 45 46 47 48 49 50 51
```

The script uses a 32-wide, two-block transformer with four attention heads,
context length 64, batch size 32, AdamW at `1e-3`, 100 updates, and 16 held-out
batches per seed. The text is split into the first 90% for training and final
10% for evaluation. All five configurations use the same per-seed training and
evaluation batches. The variants are baseline; attention only; cosine-with-floor
schedule only (`floor_ratio=0.1`); gain-aware initialization only; and all
three changes. No hyperparameter search was performed.

On the run recorded below, the script downloaded
[tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
Its SHA-256 was
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

Environment: Python 3.14.2; PyTorch 2.15.0+cu132; Linux
6.8.0-1070-azure-x86_64; AMD EPYC 7763 64-Core Processor; two CPUs available,
one PyTorch thread; CUDA was unavailable and not used.

## Results

### 100-step early-optimisation only

This retained result describes early optimisation only. It is not evidence of
improved converged training, and should not be used to select recommended
defaults. Means, sample standard deviations, and percentile 95% bootstrap
confidence intervals are across ten seeds. Paired differences are candidate
minus baseline; negative loss and positive accuracy differences favor the
candidate. Bootstrap intervals use 10,000 resamples with seed 0.

| Variant | Evaluation loss: mean ± SD [95% CI] | Paired loss difference [95% CI] | Evaluation accuracy: mean ± SD [95% CI] | Paired accuracy difference [95% CI] |
|---|---:|---:|---:|---:|
| Baseline | 3.316960 ± 0.016550 [3.307213, 3.326262] | — | 0.155560 ± 0.004749 [0.153000, 0.158585] | — |
| Attention only | 3.297334 ± 0.019872 [3.285348, 3.308567] | -0.019626 [-0.022289, -0.017156] | 0.161313 ± 0.006508 [0.157492, 0.165149] | +0.005753 [+0.003778, +0.007718] |
| Schedule only | 3.295493 ± 0.017318 [3.285119, 3.305255] | -0.021467 [-0.022869, -0.019659] | 0.158344 ± 0.005904 [0.155127, 0.162015] | +0.002783 [+0.001904, +0.003641] |
| Initialization only | 3.297055 ± 0.020868 [3.284797, 3.309173] | -0.019905 [-0.034063, -0.004583] | 0.170090 ± 0.007276 [0.165765, 0.174347] | +0.014529 [+0.010599, +0.018985] |
| All components | 3.287375 ± 0.020377 [3.275629, 3.299360] | -0.029586 [-0.042960, -0.015037] | 0.169904 ± 0.007197 [0.165662, 0.174225] | +0.014343 [+0.010147, +0.019150] |

### 500-step matched-seed run

Exact command:

```bash
python benchmarks/real_task.py --steps 500 --seeds 42 43 44 45 46 47 48 49 50 51
```

This was re-run on the same tiny Shakespeare split and hash above. Environment:
Python 3.14.2; PyTorch 2.15.0+cu132; Linux
6.8.0-1070-azure-x86_64; AMD EPYC 7763 64-Core Processor; two CPUs available,
one PyTorch thread; CPU only. Bootstrap intervals use 10,000 resamples with
seed 0. The table gives mean and sample SD of evaluation loss, then the paired
candidate-minus-baseline loss difference and percentile 95% bootstrap CI.

| Variant | Evaluation loss: mean ± SD | Paired loss difference [95% CI] | Seeds with lower loss |
|---|---:|---:|---:|
| Baseline | 2.669571 ± 0.019377 | — | — |
| Attention only | 2.675200 ± 0.016485 | +0.005629 [+0.003551, +0.008006] | 0/10 |
| Schedule only | 2.640919 ± 0.016540 | -0.028653 [-0.030431, -0.026831] | 10/10 |
| Initialization only | 2.708639 ± 0.026956 | +0.039068 [+0.028469, +0.051430] | 0/10 |
| All components | 2.727151 ± 0.029671 | +0.057579 [+0.044530, +0.072111] | 0/10 |

At this budget, attention-only and initialization-only have worse loss, and
combining all three is also worse. Only the scheduler-only variant improves
this measured loss; this is a result for this small model, dataset, and budget,
not evidence that the schedule generalizes to other training settings.

### 1000-step, 20-seed matched run

Exact command:

```bash
python benchmarks/tiny_shakespeare.py --steps 1000 --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 --device auto
```

The shortcut `--seeds 20` selects seeds 42 through 61. This run used the same
90/10 tiny Shakespeare split and SHA-256 recorded above. Environment: Python
3.14.2; PyTorch 2.15.0+cu132; AMD EPYC 7763 CPU; one PyTorch thread; CUDA was
unavailable. The 100 per-variant/seed records and final summary are preserved
in [tiny-shakespeare-1000.jsonl](./tiny-shakespeare-1000.jsonl). Paired loss
differences are candidate minus baseline, with 10,000 bootstrap resamples.

| Variant | Evaluation loss: mean ± SD [95% CI] | Paired loss difference [95% CI] | Seeds with lower loss |
|---|---:|---:|---:|
| Baseline | 2.509060 ± 0.008279 [2.505421, 2.512512] | — | — |
| Attention only | 2.523690 ± 0.008614 [2.519839, 2.527316] | +0.014630 [+0.012299, +0.016914] | 0/20 |
| Schedule only | 2.491991 ± 0.008270 [2.488274, 2.495369] | -0.017070 [-0.017894, -0.016205] | 20/20 |
| Initialization only | 2.510488 ± 0.012162 [2.505358, 2.515830] | +0.001427 [-0.003189, +0.006408] | 11/20 |
| All components | 2.510019 ± 0.011445 [2.505258, 2.514954] | +0.000959 [-0.003359, +0.005490] | 11/20 |

The scheduler-only variant improves this measured 1,000-step loss. Attention
only is worse; initialization-only and all-components intervals include zero.
This is one small model and dataset, not evidence about large-scale models.

### Attention runtime and memory profile

Exact command: `python benchmarks/profile_overhead.py --iterations 5`. CPU,
PyTorch 2.15.0+cu132, batch size 2, width 64, four heads, five measured
forward/backward iterations after two warmups. Each case ran in a fresh process.
CPU peak memory is process maximum RSS, so it includes Python/PyTorch startup
and is not an incremental attention-only allocation measurement. CUDA runs use
`torch.cuda.max_memory_allocated` when available.

| Sequence length | Backend | Forward (ms) | Backward (ms) | Peak memory (MB) |
|---:|---|---:|---:|---:|
| 64 | Standard `nn.MultiheadAttention` | 0.461 | 0.954 | 545.6 |
| 64 | `NonCollapsingAttention` block | 1.265 | 1.761 | 583.8 |
| 256 | Standard `nn.MultiheadAttention` | 2.422 | 4.603 | 547.6 |
| 256 | `NonCollapsingAttention` block | 2.818 | 6.047 | 586.3 |
| 1024 | Standard `nn.MultiheadAttention` | 22.307 | 59.363 | 556.7 |
| 1024 | `NonCollapsingAttention` block | 23.631 | 63.555 | 595.1 |

This CPU profile shows higher latency and process RSS for the toolkit block at
these measured points; it is not a GPU or full-transformer benchmark.

### Synthetic vision task

Exact command: `python benchmarks/tiny_vit.py --steps 500 --seeds 42 43 44 45 46 47 48 49 50 51 --device auto`.
This is a deterministic synthetic 16×16 grayscale task: class labels identify
which of ten 4×4 tiles contains a bright patch, with seeded Gaussian noise.
The tiny ViT embeds 4×4 patches and uses `get_scale_invariant_vit_coords` for
positions. It is not CIFAR-10. Environment: CPU, Python 3.14.2, PyTorch
2.15.0+cu132; 128 train and 32 evaluation examples per class, batch size 64.
Paired differences are corrected initialization minus standard initialization.

| Metric | Standard init: mean ± SD | Corrected init: mean ± SD | Paired difference [95% CI] | Seeds favoring corrected |
|---|---:|---:|---:|---:|
| Cross-entropy loss | 0.005452 ± 0.008682 | 0.006570 ± 0.002834 | +0.001117 [-0.004393, +0.004719] | 1/10 |
| Accuracy | 0.999688 ± 0.000988 | 1.000000 ± 0.000000 | +0.000313 [+0.000000, +0.000938] | 1/10 |

## Sparse attention optimization

Command:

```bash
python benchmarks/profile_sparse_attention.py --iterations 3 --dense-attention-max 2048 --dense-alibi-max 4096 --dense-mask-compare-max 8192 --seq-lengths 128 256 512 1024 2048 4096 8192 16384 --device cpu
```

Environment: Python 3.14.2, PyTorch 2.15.0+cu132, Linux 6.8.0-1070-azure,
AMD EPYC 7763 CPU, one thread, float32, window 64, four sink tokens, and query
chunk size 128. Each measured case ran in a fresh process with two warmups and
three timed iterations. CPU memory is process peak RSS, including Python and
PyTorch startup; it is not incremental tensor allocation. CUDA was unavailable.
The dense attention and ALiBi limits prevent large quadratic references from
exceeding the configured caps. The mask equality check is capped separately.

Dense builders return the same dense tensors as before; chunking bounds
intermediate construction allocations but does not make the returned tensor
sparse. Timings and peak RSS are reference -> chunked; speedup is reference time
divided by chunked time.

| Builder | Length | Time (ms) | Peak RSS (MB) | Speedup | Max error / RMSE |
|---|---:|---:|---:|---:|---:|
| Mask | 128 | 0.084 -> 0.111 | 536.9 -> 538.1 | 0.76x | 0 / 0 |
| Mask | 256 | 0.171 -> 0.255 | 537.1 -> 538.0 | 0.67x | 0 / 0 |
| Mask | 512 | 0.779 -> 0.617 | 538.3 -> 538.6 | 1.26x | 0 / 0 |
| Mask | 1,024 | 2.638 -> 2.450 | 542.7 -> 541.1 | 1.08x | 0 / 0 |
| Mask | 2,048 | 9.031 -> 11.411 | 560.8 -> 550.0 | 0.79x | 0 / 0 |
| Mask | 4,096 | 36.989 -> 31.401 | 633.0 -> 571.6 | 1.18x | 0 / 0 |
| Mask | 8,192 | 275.419 -> 168.009 | 857.2 -> 668.1 | 1.64x | 0 / 0 |
| Mask | 16,384 | 1,118.670 -> 707.985 | 1,817.4 -> 1,056.0 | 1.58x | Not compared |
| ALiBi | 128 | 0.131 -> 0.241 | 538.6 -> 539.7 | 0.54x | 0 / 0 |
| ALiBi | 256 | 0.651 -> 0.641 | 541.6 -> 543.0 | 1.01x | 0 / 0 |
| ALiBi | 512 | 3.523 -> 3.816 | 550.4 -> 551.0 | 0.92x | 0 / 0 |
| ALiBi | 1,024 | 11.718 -> 10.051 | 586.4 -> 574.6 | 1.17x | 0 / 0 |
| ALiBi | 2,048 | 100.947 -> 61.394 | 682.3 -> 675.0 | 1.64x | 0 / 0 |
| ALiBi | 4,096 | 451.093 -> 220.674 | 1,114.5 -> 1,065.1 | 2.04x | 0 / 0 |
| ALiBi | 8,192, 16,384 | Not run | Not run | — | Over configured dense-reference cap |

Builder timings vary: chunking is slower at some short lengths and does not
reduce total output storage. Its benefits appear at larger lengths, especially
for ALiBi; the measured peak RSS reduction at 16,384 mask length was about 42%.

Attention timings include forward and backward; memory is process peak RSS.
Dense versus sparse output max absolute error / RMSE is shown where both paths
ran. Dense reference attention was capped at length 2,048 in the primary sweep.

| Length | Dense time / RSS | Sparse time / RSS | Speedup | Max error / RMSE |
|---:|---:|---:|---:|---:|
| 128 | 2.267 ms / 587.1 MB | 1.458 ms / 585.8 MB | 1.55x | 0 / 0 |
| 256 | 12.075 ms / 594.1 MB | 2.749 ms / 588.3 MB | 4.39x | 4.17e-7 / 3.03e-8 |
| 512 | 32.277 ms / 615.9 MB | 6.148 ms / 591.2 MB | 5.25x | 7.15e-7 / 3.66e-8 |
| 1,024 | 130.357 ms / 791.9 MB | 20.081 ms / 602.6 MB | 6.49x | 4.77e-7 / 3.82e-8 |
| 2,048 | 650.625 ms / 1,363.6 MB | 35.244 ms / 620.4 MB | 18.46x | 8.34e-7 / 3.76e-8 |
| 4,096 | Not in primary sweep | 64.243 ms / 661.4 MB | — | — |
| 8,192 | Not run | 106.905 ms / 740.1 MB | — | — |
| 16,384 | Not run | 235.062 ms / 885.9 MB | — | — |

A separate exploratory 4,096 run with one measured iteration completed: dense
attention took 2,522.016 ms at 2,695.1 MB peak RSS; sparse took 52.633 ms at
653.8 MB, with max absolute error 7.15e-7 and RMSE 3.74e-8. A three-iteration
repeat was terminated, so this single-iteration result is not included in the
primary table or treated as a stable benchmark. No dense attention comparison
was attempted at 8,192 or 16,384. Sparse-only timings at those lengths are not
speedup claims.

These are CPU microbenchmarks for batch 2, width 64, and four heads. They show
that the additive sparse attention path avoids the full attention matrix and
matches the dense reference closely in measured comparisons. They do not
establish GPU performance or end-to-end transformer throughput. Dense builder
APIs remain dense by design; callers seeking asymptotically reduced attention
memory must use the sparse attention API.

Both variants solve this deliberately simple task. Corrected initialization
does not improve paired loss, and the accuracy ceiling makes this a weak test
of vision generalization.

### Supporting numerical checks

For `d=64`, `tau=10`, five seeded simulations gave these mean single-needle
weights. The approximation includes the expected exponential mass of random
cosine distractors, `exp(tau^2 / (2*d))`; the simpler expression that treats
each distractor's exponential mass as one is not accurate here. The test is
`test_random_cosine_distractor_mass_matches_gaussian_approximation` in
`tests/test_properties.py`.

| Keys | Simulated mean | Approximation | Relative error |
|---:|---:|---:|---:|
| 128 | 0.988106 | 0.987563 | 0.055% |
| 1,024 | 0.908802 | 0.907900 | 0.099% |
| 8,192 | 0.554470 | 0.551803 | 0.483% |
| 65,536 | 0.134497 | 0.133358 | 0.854% |

`AsymptoticLRScheduler` without warmup implements the same cosine schedule as
`CosineAnnealingLR(T_max=total_steps, eta_min=floor_ratio*base_lr)` inside the
configured horizon, up to floating-point rounding (maximum observed absolute
LR difference `2.17e-18` on a 35-configuration grid). The toolkit scheduler
clamps to its floor after the horizon, while stock cosine can rise again; the
measured maximum difference after 200 extra updates was `0.001` on that grid.
It also supports warmup. Regression checks are in
`tests/test_properties.py`. The benchmark result above is not a comparison of
the two schedulers with an identical floor configuration.

## Exploratory length evaluation (incomplete gate)

The following command ran after the implementation changes, using the same
dataset digest and environment as the 500-step run above:

```bash
python benchmarks/length_generalization.py --task char --variants abs rope-log --steps 500 --seeds 42 43 44 45 46 47 48 49 50 51 --init-modes baseline --schedulers toolkit --max-runs 20 --output /tmp/length-generalization-char500.jsonl
```

The 20 individual trials completed and were persisted; the first summary pass
failed with `KeyError: 'tau0'`. After correcting the summary code, rerunning
the same command printed `new_runs: 0` and produced the paired summary below.
Each length uses two held-out sequences per seed; evaluation sequences differ
across lengths, so comparisons across lengths include sampling variation.
Values are mean perplexity ± sample SD and paired difference against the same
seed at length 64, with a 10,000-resample percentile bootstrap CI.

| Variant | Context | Perplexity mean ± SD | Paired difference vs context 64 [95% CI] |
|---|---:|---:|---:|
| Absolute baseline | 64 | 13.7467 ± 1.3604 | — |
| Absolute baseline | 128–4096 | N/A | — |
| RoPE + log | 64 | 10.5350 ± 0.9704 | — |
| RoPE + log | 128 | 10.8532 ± 1.0430 | +0.3182 [-0.7884, +1.3538] |
| RoPE + log | 256 | 11.7881 ± 0.9548 | +1.2530 [+0.5655, +1.9954] |
| RoPE + log | 512 | 13.0723 ± 0.7572 | +2.5373 [+1.6601, +3.3199] |
| RoPE + log | 1024 | 13.8927 ± 0.9808 | +3.3577 [+2.4131, +4.3867] |
| RoPE + log | 4096 | 16.0059 ± 0.8907 | +5.4708 [+4.6121, +6.2183] |

At length 64 the RoPE + log minus absolute-baseline paired difference is
`-3.2116`, CI `[-3.7032, -2.7610]`, with lower perplexity for 10/10 seeds.
These are different architectures and their batches differ from the historical
`real_task.py` run. This single comparison does not isolate RoPE or log scaling.
Perplexity rises substantially at longer lengths, so this slice does not show
stable performance at 64 times the training length.

The following command added matched RoPE-only runs with the same seeds and
settings. Its summary compares RoPE + log against RoPE; this changes both QK
normalization and the temperature rule, so it does not isolate log scaling.
The resulting 30 per-seed records are preserved in
[length-generalization-char500.jsonl](./length-generalization-char500.jsonl).

```bash
python benchmarks/length_generalization.py --task char --variants rope --steps 500 --seeds 42 43 44 45 46 47 48 49 50 51 --init-modes baseline --schedulers toolkit --output /tmp/length-generalization-char500.jsonl
```

| Context | RoPE perplexity mean ± SD | RoPE + log minus RoPE [95% CI] | Seeds with lower loss |
|---:|---:|---:|---:|
| 64 | 10.4035 ± 0.8867 | +0.1315 [-0.0267, +0.2815] | 3/10 |
| 128 | 10.5420 ± 0.9574 | +0.3112 [+0.1339, +0.4885] | 1/10 |
| 256 | 11.2669 ± 1.0690 | +0.5212 [+0.3011, +0.7314] | 1/10 |
| 512 | 13.6718 ± 1.2954 | -0.5995 [-1.3365, +0.0677] | 6/10 |
| 1024 | 16.3468 ± 2.5690 | -2.4541 [-3.9310, -1.1031] | 9/10 |
| 4096 | 20.7109 ± 3.8651 | -4.7050 [-6.8017, -2.6631] | 9/10 |

### Exploratory 2000-step paired comparison

```bash
python benchmarks/length_generalization.py --task char --variants rope rope-log --steps 2000 --seeds 42 43 44 45 46 47 48 49 50 51 --init-modes baseline --schedulers toolkit --output /tmp/length-generalization-char2000.jsonl
```

The paired runs use the same dataset hash, held-out split, seeds and evaluation
sampling as the exploratory 500-step run. The 20 trial records are preserved in
[length-generalization-char2000.jsonl](./length-generalization-char2000.jsonl).
Perplexity differences are RoPE + log minus RoPE. The variant changes both QK
normalization and temperature scaling, so their effects cannot be separated.

| Context | RoPE perplexity mean ± SD | RoPE + log perplexity mean ± SD | Paired difference [95% CI] | Seeds with lower loss |
|---:|---:|---:|---:|---:|
| 64 | 7.4986 ± 1.0199 | 7.7400 ± 0.7692 | +0.2414 [-0.0608, +0.5274] | 2/10 |
| 128 | 7.7718 ± 1.0069 | 8.6551 ± 1.1508 | +0.8833 [+0.6049, +1.1747] | 0/10 |
| 256 | 7.8326 ± 0.7167 | 9.5773 ± 1.2513 | +1.7448 [+1.3569, +2.1295] | 0/10 |
| 512 | 15.8098 ± 2.9635 | 12.4564 ± 0.6807 | -3.3534 [-5.1971, -1.9888] | 10/10 |
| 1024 | 27.4523 ± 9.1786 | 16.1943 ± 1.9355 | -11.2580 [-17.0692, -6.6306] | 10/10 |
| 4096 | 44.9728 ± 18.6546 | 24.3149 ± 2.9740 | -20.6579 [-32.5389, -10.5431] | 10/10 |

The longer-context comparison favors the combined variant here, but the
64-token confidence interval includes zero at both budgets. Perplexity also
increases with context length for both variants. These results do not meet the
recommendation rule and do not establish stable 64x length generalization.

### 16k-token numerical smoke checks

Two CPU single-thread forward/backward checks used `NonCollapsingAttention(64)`
with batch size 1 and sequence length 16,384. With standard-normal Q/K/V,
outputs and query gradients were finite and output standard deviation was
`0.0162`. In a separate stress check, Q and K were multiplied by `1e37` while
V remained standard-normal; outputs and Q/K/V gradients were finite, and output
standard deviation was `0.0160`. Neither run raised OOM. Peak memory was not
measured.

These are single-layer random-input numerical checks, not optimizer updates or
training. They do not prove stable training, length generalization, or
overflow safety for every finite input. The attention computation still has
quadratic work in sequence length; practical full training at this length was
not measured here.

The full 500/2000-step variant and initialization/scheduler matrix, separate
hyperparameter sweep and matched-seed successor evaluation have not been
completed. No component meets the stated recommendation rule on this evidence;
new options remain opt-in with no measured general training benefit.
The benchmark uses a dense streaming mask, not the separate bounded inference
KV cache, so it does not measure constant-memory streaming.

### Exploratory needle retrieval (500 steps)

```bash
python benchmarks/length_generalization.py --task needle --variants rope rope-log --steps 500 --seeds 42 43 44 45 46 47 48 49 50 51 --init-modes baseline --schedulers toolkit --eval-samples 1 --output /tmp/length-generalization-needle500.jsonl
```

All 20 matched trial records are preserved in
[length-generalization-needle500.jsonl](./length-generalization-needle500.jsonl).
Exact retrieval accuracy was zero for both variants at every tested length.
Each seed/length contributes one evaluation query, so these accuracies are
coarse and do not estimate small nonzero success rates reliably. Paired
candidate-minus-RoPE differences are all zero (10 matched seeds); this is a
negative result, not evidence that the variants are equivalent in general.

| Length | RoPE accuracy | RoPE + log accuracy | Paired accuracy difference [95% CI] | Seeds with higher candidate accuracy |
|---:|---:|---:|---:|---:|
| 64 | 0/10 | 0/10 | 0.00 [0.00, 0.00] | 0/10 |
| 128 | 0/10 | 0/10 | 0.00 [0.00, 0.00] | 0/10 |
| 256 | 0/10 | 0/10 | 0.00 [0.00, 0.00] | 0/10 |
| 512 | 0/10 | 0/10 | 0.00 [0.00, 0.00] | 0/10 |
| 1024 | 0/10 | 0/10 | 0.00 [0.00, 0.00] | 0/10 |
| 4096 | 0/10 | 0/10 | 0.00 [0.00, 0.00] | 0/10 |
| 16384 | 0/10 | 0/10 | 0.00 [0.00, 0.00] | 0/10 |

## Mutation sanity

Each mutation was injected into an isolated copy, then the relevant focused
pytest selection was run (not the whole suite for each mutation). Every
selection failed on an assertion, with no surviving mutation:

| Injected mutation | Focused test result |
|---|---|
| Remove log scaling | Failed |
| Use padded instead of unmasked length | Failed |
| Wrong RoPE frequency | Failed |
| Flip RoPE angle sign | Failed |
| Compute position angles from float16 positions | Failed |
| Drop sink tokens | Failed |
| Remove effective-temperature clamp | Failed |
| Flip ALiBi sign | Failed |
| Use naive dilution formula | Failed |

## Extended validation

### Attention profiling

Command: `python benchmarks/profile_overhead.py --iterations 5`.
Environment: Python 3.14.2, PyTorch 2.15.0+cu132, Linux 6.8.0-1070-azure,
AMD EPYC 7763, CPU, one thread. Batch size 2, width 64, four heads; two warmup
passes followed by five timed forward/backward passes. Each case ran in a fresh
process. CPU memory is process peak RSS and includes interpreter/PyTorch
startup; it is not incremental attention allocation. CUDA runs use
`torch.cuda.max_memory_allocated` when available.

| Sequence length | Backend | Forward (ms) | Backward (ms) | Peak memory (MB) |
|---:|---|---:|---:|---:|
| 64 | Standard `nn.MultiheadAttention` | 0.461 | 0.954 | 545.6 |
| 64 | Toolkit attention block | 1.265 | 1.761 | 583.8 |
| 256 | Standard `nn.MultiheadAttention` | 2.422 | 4.603 | 547.6 |
| 256 | Toolkit attention block | 2.818 | 6.047 | 586.3 |
| 1024 | Standard `nn.MultiheadAttention` | 22.307 | 59.363 | 556.7 |
| 1024 | Toolkit attention block | 23.631 | 63.555 | 595.1 |

On this CPU profile the toolkit block has higher process RSS and is slower at
all three lengths, including 1,024. This microbenchmark is not a full
transformer throughput comparison.

### Synthetic vision

Command: `python benchmarks/tiny_vit.py --steps 500 --seeds 42 43 44 45 46 47 48 49 50 51 --device auto`.
The procedural task classifies 16x16 grayscale images by the location of a
bright 4x4 tile, with seeded noise. The ViT uses patch embeddings and
`get_scale_invariant_vit_coords`. It is not CIFAR-10. Environment: CPU, Python
3.14.2, PyTorch 2.15.0+cu132; batch size 64, 128 training and 32 evaluation
examples per class. Paired differences are corrected init minus standard init
with 10,000 bootstrap resamples.

| Metric | Standard mean ± SD | Corrected mean ± SD | Paired difference [95% CI] | Seeds favoring corrected |
|---|---:|---:|---:|---:|
| Cross-entropy loss | 0.005452 ± 0.008682 | 0.006570 ± 0.002834 | +0.001117 [-0.004393, +0.004719] | 1/10 |
| Accuracy | 0.999688 ± 0.000988 | 1.000000 ± 0.000000 | +0.000313 [+0.000000, +0.000938] | 1/10 |

Both initializations nearly solve this simple task. Corrected initialization
does not improve paired loss; the accuracy ceiling makes this a weak test of
vision generalization. Per-seed output is preserved in
[tiny-vit-500.jsonl](./tiny-vit-500.jsonl).

### Tiny Shakespeare at 1000 steps

Command executed:

```bash
python benchmarks/tiny_shakespeare.py --steps 1000 --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 --device auto
```

The wrapper also accepts `--seeds 20` as shorthand for seeds 42–61. This run
uses the same 90/10 tiny Shakespeare split, 64-token context, batch size 32,
16 evaluation batches per seed, and dataset SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. Environment:
Python 3.14.2, PyTorch 2.15.0+cu132, AMD EPYC 7763, CPU, one thread; CUDA was
unavailable. The 100 per-variant/seed records and summary are preserved in
[tiny-shakespeare-1000.jsonl](./tiny-shakespeare-1000.jsonl). Means and SDs are
over 20 seeds; paired candidate-minus-baseline intervals use 10,000 bootstrap
resamples with seed 0. Negative loss differences favor the candidate.

| Variant | Evaluation loss: mean ± SD [95% CI] | Paired loss difference [95% CI] | Seeds with lower loss |
|---|---:|---:|---:|
| Baseline | 2.509060 ± 0.008279 [2.505421, 2.512512] | — | — |
| Attention only | 2.523690 ± 0.008614 [2.519839, 2.527316] | +0.014630 [+0.012299, +0.016914] | 0/20 |
| Schedule only | 2.491991 ± 0.008270 [2.488274, 2.495369] | -0.017070 [-0.017894, -0.016205] | 20/20 |
| Initialization only | 2.510488 ± 0.012162 [2.505358, 2.515830] | +0.001427 [-0.003189, +0.006408] | 11/20 |
| All components | 2.510019 ± 0.011445 [2.505258, 2.514954] | +0.000959 [-0.003359, +0.005490] | 11/20 |

Only the scheduler-only variant has a paired interval excluding zero in the
beneficial direction. Attention-only is worse, while initialization-only and
all-components are inconclusive. This supports a narrow 1,000-step
character-level LM validation, not a general training claim.
