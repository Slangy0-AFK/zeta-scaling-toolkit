# Matched-seed tiny Shakespeare evaluation

This is a small, CPU-only char-level language-model comparison. It is an
experiment on one dataset, architecture, and training budget—not evidence that
any component improves other tasks or large-scale training.

## Exact run

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

Means, sample standard deviations, and percentile 95% bootstrap confidence
intervals are across the ten seeds. Paired differences are candidate minus
baseline; therefore negative loss and positive accuracy differences favor the
candidate. Bootstrap intervals use 10,000 resamples with seed 0.

| Variant | Evaluation loss: mean ± SD [95% CI] | Paired loss difference [95% CI] | Evaluation accuracy: mean ± SD [95% CI] | Paired accuracy difference [95% CI] |
|---|---:|---:|---:|---:|
| Baseline | 3.316960 ± 0.016550 [3.307213, 3.326262] | — | 0.155560 ± 0.004749 [0.153000, 0.158585] | — |
| Attention only | 3.297334 ± 0.019872 [3.285348, 3.308567] | -0.019626 [-0.022289, -0.017156] | 0.161313 ± 0.006508 [0.157492, 0.165149] | +0.005753 [+0.003778, +0.007718] |
| Schedule only | 3.295493 ± 0.017318 [3.285119, 3.305255] | -0.021467 [-0.022869, -0.019659] | 0.158344 ± 0.005904 [0.155127, 0.162015] | +0.002783 [+0.001904, +0.003641] |
| Initialization only | 3.297055 ± 0.020868 [3.284797, 3.309173] | -0.019905 [-0.034063, -0.004583] | 0.170090 ± 0.007276 [0.165765, 0.174347] | +0.014529 [+0.010599, +0.018985] |
| All components | 3.287375 ± 0.020377 [3.275629, 3.299360] | -0.029586 [-0.042960, -0.015037] | 0.169904 ± 0.007197 [0.165662, 0.174225] | +0.014343 [+0.010147, +0.019150] |

For this specific run, each candidate's paired intervals exclude zero in the
favorable direction. This does not establish a general training benefit: the
study has only ten seeds, one small dataset/model, one fixed learning rate, and
100 updates. Results can change with task, architecture, tuning, and budget.
