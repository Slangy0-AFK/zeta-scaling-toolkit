# Matched-seed tiny Shakespeare evaluation

This is a small, CPU-only char-level language-model comparison. It is an
experiment on one dataset, architecture, and training budget—not evidence that
any component improves other tasks or large-scale training.

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

The full 500/2000-step variant and initialization/scheduler matrix, separate
hyperparameter sweep, and matched-seed needle and successor evaluations have
not been completed. No component meets the stated recommendation rule on this
evidence; new options remain opt-in with no measured general training benefit.
The benchmark uses a dense streaming mask, not the separate bounded inference
KV cache, so it does not measure constant-memory streaming.

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
