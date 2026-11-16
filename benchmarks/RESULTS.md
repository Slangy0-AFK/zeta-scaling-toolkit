# Matched-seed tiny Shakespeare evaluation

This is a small, CPU-only char-level language-model comparison. It is an
experiment on one dataset, architecture, and training budget—not evidence that
any component improves other tasks or large-scale training.

## Stage 1 context-ability gate

**Decision: NO-GO.** On the specified held-out four-digit successor task at
context 4096 and positions 1024–4095, the tuned `rope-log-window` accuracy was
0.4418 versus 0.5214 for the tuning-selected ALiBi baseline: paired difference
−0.0796, 95% CI [−0.1053, −0.0540]. The tuned `rope-log-full` accuracy was
0.2285 versus the same ALiBi baseline, with paired difference −0.2930,
95% CI [−0.3210, −0.2624]. At the 256-token training length, four-digit
perplexity was 37.23% higher for `rope-log-window` and 26.43% higher for
`rope-log-full` than for the ALiBi baseline. The planted-needle sanity gate
also failed. On character data at length 256, perplexity was 4.78% higher
for tuned `rope-log-window` and 9.21% higher for tuned `rope-log-full` than
for the tuning-selected ALiBi baseline. These 500-step results are a clear
NO-GO; the 2,000-step run and Stage 2 are not authorized.

On the supplementary three-digit *training-range* cyclic stream at context
4096, `rope-log-window` accuracy at positions 1024–4095 was 0.9530 versus
0.7806 for the ALiBi baseline, paired difference +0.1724, 95% CI
[+0.1063, +0.2298]. This stream repeats numbers seen in training rather than
the specified held-out range 1000–2999, so it is not counted as a gate pass
or evidence of held-out long-context ability. Character perplexity alone is
not evidence that a model uses distant context.

The base checked was `f924538e7aef263b6dec9ea24850ab31bf6fa34e` in an
isolated worktree. On that base, `pytest --cov-fail-under=95` passed with
457 passed and 1 skipped; `pytest --cov=zeta_scaling_toolkit
--cov-fail-under=95` passed with 457 passed, 1 skipped and 99.11% coverage.
`ruff check .`, `ruff format --check .`, and `mypy --strict
zeta_scaling_toolkit` passed. `python tools/mutation_check.py` killed 17 of 17
mutants, with none surviving, missing or errored. These are base checks, not
final validation of the uncommitted changes.

After the final-query retrieval optimization, `ruff check .`, `ruff format
--check .`, and `mypy --strict zeta_scaling_toolkit` passed. `python -m pytest
--cov=zeta_scaling_toolkit --cov-fail-under=95` passed with 549 passed, one
skipped, and 99.17% package coverage. In an isolated worktree containing the
same code, `python tools/mutation_check.py` killed 27 of 27 mutants, including
the final-query RoPE-position mutant; none survived, were missing, or errored.
The same full coverage command also passed on Python 3.9.25 with PyTorch
2.0.1+cpu: 549 passed, one skipped, and 99.17% package coverage. A separate
attempt to dispatch the hosted CI matrix was denied with HTTP 403; no hosted
CI-matrix result is claimed.

The Stage 1 runners persist each completed training trial as JSONL and reject
incompatible resume configurations. The experiments use batch size 16 and
one CPU thread. The character control run trains at length 256 and evaluates
each trained model both with full attention and with a 64-token window plus
four sinks. These runs differ from the 64-token training control below.

The completed local retrieval sanity records for seeds 42–44 used the original
full-output training path. The local run was resumed with the final-query
path and then stopped before another seed completed. Focused tests compare the
paths' outputs, gradients, and short training trajectories, but the
floating-point trajectories are not bit-identical. The
`training_implementation` field identifies the path in newer records. The
local character and successor-tuning files contain only 19 and 11 completed
records, respectively. They are partial and are not used in the decision.
An independent GitHub Actions run completed the Stage 1 trials on hosted CPU
runners using Python 3.14 and CPU PyTorch 2.15.0; its artifacts include
runner CPU details. Hosted and local results are not combined.
The [hosted run](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions/runs/43780419668)
uses the [workflow at commit `d586858`](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/blob/d586858/.github/workflows/context-ability-500.yml),
which specifies the individual per-seed commands.

### Planted-needle sanity gate

The hosted full-attention RoPE baseline passed the 0.9 accuracy threshold on
6 of 10 seeds, below the required 8 of 10. The task is **not learnable at this
scale** under the specified sanity gate; retrieval comparisons are skipped.
Each seed was trained for 3,000 steps at length 256 and evaluated on 200
queries. The recorded correct counts for seeds 42–51 were 187, 133, 200,
69, 108, 198, 180, 150, 200, and 191 respectively. The raw records are in
`context-ability-needle-sanity-hosted.jsonl`.

Each example has only one planted key-value pair and one value token, so a
model can find the value without matching the query key. Even a passed sanity
gate would not establish key-dependent associative retrieval.

The hosted jobs used one PyTorch thread, Python 3.14.10, and PyTorch
2.15.0+cpu. The virtual machines reported four logical CPUs and included AMD
EPYC 7763, AMD EPYC 9V74, Intel Xeon Platinum 8573C, and Intel Xeon 6975P-C
processors; CPU models were not identical across seeds. The per-job hardware
is recorded in `context-ability-hosted-hardware.json`. The tiny Shakespeare
dataset SHA-256 was
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

```bash
python benchmarks/context_char.py --steps 500 --train-length 256 --window 64 --sinks 4 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output benchmarks/context-ability-char500.jsonl
python benchmarks/context_ability.py --steps 3000 --train-length 256 --batch-size 16 \
  --eval-queries 200 --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output benchmarks/context-ability-needle-sanity.jsonl
python benchmarks/context_successor.py --tuning --steps 500 --batch-size 16 --samples 8 \
  --seeds 100 101 102 103 104 105 106 107 108 109 --candidates 0 1 \
  --output benchmarks/context-ability-successor-tuning500.jsonl
python benchmarks/context_successor.py --steps 500 --batch-size 16 --samples 8 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --tuning-records benchmarks/context-ability-successor-tuning500.jsonl \
  --output benchmarks/context-ability-successor500.jsonl
python benchmarks/context_char_tuned.py --tuning --steps 500 \
  --seeds 100 101 102 103 104 105 106 107 108 109 --candidates 0 1 \
  --output benchmarks/context-ability-char-tuning500.jsonl
python benchmarks/context_char_tuned.py --steps 500 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --tuning-records benchmarks/context-ability-char-tuning500.jsonl \
  --output benchmarks/context-ability-char-baselines500.jsonl
```

If the retrieval sanity gate had passed, the comparison would have used:

```bash
python benchmarks/context_needle_tuned.py --tuning --steps 500 \
  --seeds 100 101 102 103 104 105 106 107 108 109 \
  --output benchmarks/context-ability-needle-tuning500.jsonl
python benchmarks/context_needle_tuned.py --steps 500 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --tuning-records benchmarks/context-ability-needle-tuning500.jsonl \
  --output benchmarks/context-ability-needle500.jsonl
```

Successor tuning chooses among two candidates per method using four-digit
held-out accuracy at positions 1024–4095 in a 4096-token sequence. Absolute
positions cannot run at 4096, so that method is tuned on its 256-token
accuracy instead. Three-digit successor evaluations use a cyclic repetition
of the training-range numbers 100–999, and are not held-out-number results.
Four-digit successor evaluations use numbers 1000–2999, held out from model
training but shared by tuning and reporting; disjoint reporting seeds do not
make the number ranges independent. The
reports distinguish digit-only accuracy from accuracy including
separators.

The tuned character runs select the candidate with the lowest late-position
NLL using a 1024-token prefix; absolute positions instead use 256-token
perplexity because they cannot be evaluated beyond their position table. Each
method, including the trained window variants, has two candidate configurations
and ten tuning seeds. Identical full-RoPE training runs are reused within a
process when linear or NTK scaling changes only inference positions; each
reused model has the same seed, data, optimizer and update count as an
independently trained model. The reporting
seeds are disjoint from tuning seeds. The character context-benefit curve is
a sanity check, not by itself a Stage 1 GO criterion. It scores eight target
positions from 1024 through 1920 per seed; each prefix is truncated before
the forward pass. A windowed model's sink positions therefore refer to the
beginning of that truncated prefix, not to the start of the original document.
For a 64-token window plus four sinks trained at length 256, the current log
scaling factor remains one because at most 68 keys are visible.
The two tuning candidates use initial temperatures 5 and 10 for cosine+log,
position scale factors 4 and 16 for linear and NTK RoPE, and learning rates
0.0007 and 0.001 for methods without either tunable quantity. Other candidate
learning rates are 0.001. Linear and NTK scaling are applied at evaluation
time to models trained with unscaled RoPE.
The successor runner also offers candidate 2 in new tuning runs via
`--candidates 0 1 2`: learning rate 0.0015 for methods tuned by learning rate,
initial temperature 15 for cosine+log, and position scale 8 for linear and NTK
RoPE. The default remains candidates 0 and 1. The hosted results above used
only candidates 0 and 1; candidate 2 has not been evaluated in a complete
tuning run and is not evidence of an improvement.
Successor candidate 3 is an opt-in initial temperature of 2 for
`rope-log-full` and `rope-log-window`; it cannot be used with other methods.
To test it, pass `--methods rope-log-window --candidates 3 --tuning`.
The original default candidates and the historical Stage 1 decision are
unchanged.

#### Exploratory successor temperature comparison

On local CPU, matched seeds 100–109 were used to compare `rope-log-window`
candidate 3 (temperature 2) with candidate 0 (temperature 5). At context
4096, positions 1024–4095 on the four-digit stream, mean accuracy was 0.4799
versus 0.4358: a paired difference of +0.0441, 95% seed-bootstrap CI
[+0.0239, +0.0641]. Candidate 3 improved 9 of 10 seeds. These are tuning
results, not an independent reporting-seed test; the first two tuning seeds
also informed the choice of the lower temperature. On the same seeds the
previously selected ALiBi candidate 0 averaged 0.5220, exceeding candidate 3
on all ten seeds; the paired difference was -0.0421, 95% CI
[-0.0576, -0.0287]. This local run did not retune all baselines. It does not
establish a Stage 1 pass or measure the training-length perplexity gate. The
original retrieval sanity gate remains failed.

Two-seed probes on seeds 100–101 also tried temperatures 1.3 and 1.6 at
learning rate 0.001; both had lower 4096-token accuracy than temperature 2
on both seeds. At temperature 2, rates 0.0015 and 0.002 had substantially
higher 256-token perplexity on both seeds than rate 0.001, although 0.002
slightly increased 4096-token accuracy. These probes are not complete tuning
runs and do not rule out other settings.

The local runs used Python 3.14.2, PyTorch 2.15.0+cu132 on CPU, one PyTorch
thread per run, and an AMD EPYC 7763 host. Their records are kept separate
from the hosted runs and can be reproduced with:

```bash
python benchmarks/context_successor.py --tuning --steps 500 --batch-size 16 --samples 8 \
  --methods rope-log-window --seeds 100 101 102 103 104 105 106 107 108 109 \
  --candidates 0 --output benchmarks/context-successor-temperature-control-tuning500.jsonl
python benchmarks/context_successor.py --tuning --steps 500 --batch-size 16 --samples 8 \
  --methods rope-log-window --seeds 100 101 102 103 104 105 106 107 108 109 \
  --candidates 3 --output benchmarks/context-successor-low-temperature-tuning500.jsonl
python benchmarks/context_successor.py --tuning --steps 500 --batch-size 16 --samples 8 \
  --methods alibi --seeds 100 101 102 103 104 105 106 107 108 109 \
  --candidates 0 --output benchmarks/context-successor-alibi-tuning500.jsonl
```

### Hosted candidate selection

The selected candidate indices, with candidates ordered as listed above, are:

| Method | Successor | Character |
|---|---:|---:|
| abs | 1 | 1 |
| rope-full | 0 | 0 |
| rope-linear | 1 | 1 |
| rope-ntk | 1 | 1 |
| alibi | 0 | 1 |
| rope-log-full | 0 | 0 |
| rope-window | 0 | 1 |
| rope-log-window | 0 | 1 |

The hosted 500-step paired tables can be regenerated with:

```bash
python benchmarks/context_report.py benchmarks/context-ability-char500-hosted.jsonl \
  --successor-records benchmarks/context-ability-successor500-hosted.jsonl \
  --successor-tuning benchmarks/context-ability-successor-tuning500-hosted.jsonl \
  --char-baselines benchmarks/context-ability-char-baselines500-hosted.jsonl \
  --char-tuning benchmarks/context-ability-char-tuning500-hosted.jsonl
```

The reporting script requires every specified method and all ten reporting
seeds; it does not calculate paired intervals from partial runs. It uses 10,000
seed-paired bootstrap resamples.

### Stage 1 hosted paired tables

#### Character controls

| Training | Evaluation | Context | PPL | Difference vs RoPE+log full [95% CI] |
|---|---|---:|---:|---:|
| rope-full | full | 256 | 10.9108 | -0.3295 [-0.6948, +0.0014] |
| rope-full | full | 512 | 13.5477 | +1.6268 [+1.2511, +1.9793] |
| rope-full | full | 1024 | 17.9442 | +4.5338 [+3.3128, +5.6196] |
| rope-full | full | 4096 | 23.6202 | +7.5033 [+5.1623, +9.5918] |
| rope-full | full | 16384 | 25.3627 | +8.5185 [+6.4833, +10.4074] |
| rope-full | window | 256 | 10.9036 | -0.3366 [-0.7047, -0.0098] |
| rope-full | window | 512 | 11.8071 | -0.1138 [-0.4180, +0.1728] |
| rope-full | window | 1024 | 11.2834 | -2.1270 [-2.8010, -1.5628] |
| rope-full | window | 4096 | 11.7669 | -4.3500 [-6.1188, -3.0528] |
| rope-full | window | 16384 | 11.7799 | -5.0643 [-6.9143, -3.6294] |
| rope-window | full | 256 | 10.7910 | -0.4493 [-0.7705, -0.1867] |
| rope-window | full | 512 | 13.7539 | +1.8330 [+1.4897, +2.1695] |
| rope-window | full | 1024 | 18.4463 | +5.0359 [+3.8738, +5.9938] |
| rope-window | full | 4096 | 24.4805 | +8.3636 [+5.8950, +10.3606] |
| rope-window | full | 16384 | 26.4891 | +9.6448 [+7.0114, +11.7869] |
| rope-window | window | 256 | 10.6653 | -0.5749 [-0.9239, -0.2935] |
| rope-window | window | 512 | 11.6308 | -0.2901 [-0.5980, -0.0136] |
| rope-window | window | 1024 | 11.0981 | -2.3123 [-2.9690, -1.7510] |
| rope-window | window | 4096 | 11.6366 | -4.4803 [-6.2490, -3.1357] |
| rope-window | window | 16384 | 11.6477 | -5.1965 [-6.9951, -3.7770] |
| rope-log-full | full | 256 | 11.2403 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | full | 512 | 11.9209 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | full | 1024 | 13.4104 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | full | 4096 | 16.1169 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | full | 16384 | 16.8442 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | window | 256 | 11.2529 | +0.0126 [-0.0191, +0.0456] |
| rope-log-full | window | 512 | 11.5524 | -0.3685 [-0.4532, -0.2776] |
| rope-log-full | window | 1024 | 11.2635 | -2.1469 [-2.8505, -1.5601] |
| rope-log-full | window | 4096 | 11.5860 | -4.5309 [-6.3382, -3.1946] |
| rope-log-full | window | 16384 | 11.5818 | -5.2624 [-7.2331, -3.7445] |
| rope-log-window | full | 256 | 11.1308 | -0.1095 [-0.2960, +0.0463] |
| rope-log-window | full | 512 | 12.0631 | +0.1422 [-0.0432, +0.3296] |
| rope-log-window | full | 1024 | 13.6123 | +0.2019 [-0.2726, +0.6686] |
| rope-log-window | full | 4096 | 16.4007 | +0.2838 [-0.6512, +1.0965] |
| rope-log-window | full | 16384 | 17.0747 | +0.2305 [-0.6933, +1.0296] |
| rope-log-window | window | 256 | 10.9447 | -0.2955 [-0.4673, -0.1563] |
| rope-log-window | window | 512 | 11.2018 | -0.7191 [-0.8468, -0.5863] |
| rope-log-window | window | 1024 | 10.9626 | -2.4478 [-3.1117, -1.9031] |
| rope-log-window | window | 4096 | 11.2887 | -4.8282 [-6.5637, -3.5436] |
| rope-log-window | window | 16384 | 11.2711 | -5.5731 [-7.4702, -4.1100] |

| Training | Evaluation | Position bucket | Mean NLL at context 16384 |
|---|---|---|---:|
| rope-full | full | 0_63 | 2.3418 |
| rope-full | full | 64_255 | 2.3557 |
| rope-full | full | 256_1023 | 3.0617 |
| rope-full | full | 1024_4095 | 3.2397 |
| rope-full | full | 4096_plus | 3.2491 |
| rope-full | window | 0_63 | 2.3418 |
| rope-full | window | 64_255 | 2.3566 |
| rope-full | window | 256_1023 | 2.4707 |
| rope-full | window | 1024_4095 | 2.4671 |
| rope-full | window | 4096_plus | 2.4676 |
| rope-window | full | 0_63 | 2.3244 |
| rope-window | full | 64_255 | 2.3500 |
| rope-window | full | 256_1023 | 3.0924 |
| rope-window | full | 1024_4095 | 3.2705 |
| rope-window | full | 4096_plus | 3.2881 |
| rope-window | window | 0_63 | 2.3244 |
| rope-window | window | 64_255 | 2.3332 |
| rope-window | window | 256_1023 | 2.4540 |
| rope-window | window | 1024_4095 | 2.4570 |
| rope-window | window | 4096_plus | 2.4562 |
| rope-log-full | full | 0_63 | 2.3935 |
| rope-log-full | full | 64_255 | 2.3850 |
| rope-log-full | full | 256_1023 | 2.6725 |
| rope-log-full | full | 1024_4095 | 2.8239 |
| rope-log-full | full | 4096_plus | 2.8253 |
| rope-log-full | window | 0_63 | 2.3935 |
| rope-log-full | window | 64_255 | 2.3870 |
| rope-log-full | window | 256_1023 | 2.4512 |
| rope-log-full | window | 1024_4095 | 2.4474 |
| rope-log-full | window | 4096_plus | 2.4508 |
| rope-log-window | full | 0_63 | 2.3555 |
| rope-log-window | full | 64_255 | 2.3859 |
| rope-log-window | full | 256_1023 | 2.6951 |
| rope-log-window | full | 1024_4095 | 2.8526 |
| rope-log-window | full | 4096_plus | 2.8458 |
| rope-log-window | window | 0_63 | 2.3555 |
| rope-log-window | window | 64_255 | 2.3579 |
| rope-log-window | window | 256_1023 | 2.4195 |
| rope-log-window | window | 1024_4095 | 2.4197 |
| rope-log-window | window | 4096_plus | 2.4239 |

| Training | Evaluation | Prefix | NLL | Difference vs RoPE+log full [95% CI] |
|---|---|---:|---:|---:|
| rope-full | full | 16 | 2.2576 | -0.0911 [-0.1423, -0.0356] |
| rope-full | full | 32 | 2.2681 | -0.0425 [-0.0996, +0.0158] |
| rope-full | full | 64 | 2.2530 | -0.0701 [-0.1267, -0.0164] |
| rope-full | full | 128 | 2.2514 | -0.0729 [-0.1170, -0.0303] |
| rope-full | full | 256 | 2.2668 | -0.0530 [-0.1030, -0.0060] |
| rope-full | full | 512 | 3.0111 | +0.4496 [+0.1184, +0.7870] |
| rope-full | full | 1024 | 3.0203 | +0.4526 [+0.1163, +0.7859] |
| rope-full | window | 16 | 2.2576 | -0.0911 [-0.1423, -0.0356] |
| rope-full | window | 32 | 2.2681 | -0.0425 [-0.0996, +0.0158] |
| rope-full | window | 64 | 2.2530 | -0.0701 [-0.1267, -0.0164] |
| rope-full | window | 128 | 2.2493 | -0.0751 [-0.1215, -0.0341] |
| rope-full | window | 256 | 2.2512 | -0.0686 [-0.1157, -0.0248] |
| rope-full | window | 512 | 2.7296 | +0.1681 [-0.0641, +0.4446] |
| rope-full | window | 1024 | 2.2613 | -0.3064 [-0.4328, -0.1743] |
| rope-window | full | 16 | 2.2401 | -0.1086 [-0.1476, -0.0654] |
| rope-window | full | 32 | 2.2525 | -0.0580 [-0.0929, -0.0172] |
| rope-window | full | 64 | 2.2431 | -0.0801 [-0.1253, -0.0326] |
| rope-window | full | 128 | 2.2485 | -0.0759 [-0.1203, -0.0308] |
| rope-window | full | 256 | 2.2979 | -0.0219 [-0.0782, +0.0338] |
| rope-window | full | 512 | 3.0233 | +0.4618 [+0.1120, +0.8391] |
| rope-window | full | 1024 | 3.0342 | +0.4665 [+0.1181, +0.8406] |
| rope-window | window | 16 | 2.2401 | -0.1086 [-0.1476, -0.0654] |
| rope-window | window | 32 | 2.2525 | -0.0580 [-0.0929, -0.0172] |
| rope-window | window | 64 | 2.2431 | -0.0801 [-0.1253, -0.0326] |
| rope-window | window | 128 | 2.2431 | -0.0812 [-0.1177, -0.0432] |
| rope-window | window | 256 | 2.2519 | -0.0679 [-0.0976, -0.0375] |
| rope-window | window | 512 | 2.7002 | +0.1388 [-0.1129, +0.4394] |
| rope-window | window | 1024 | 2.2547 | -0.3130 [-0.4419, -0.1797] |
| rope-log-full | full | 16 | 2.3487 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | full | 32 | 2.3105 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | full | 64 | 2.3232 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | full | 128 | 2.3244 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | full | 256 | 2.3198 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | full | 512 | 2.5615 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | full | 1024 | 2.5677 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | window | 16 | 2.3487 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | window | 32 | 2.3105 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | window | 64 | 2.3232 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | window | 128 | 2.3103 | -0.0141 [-0.0443, +0.0111] |
| rope-log-full | window | 256 | 2.3138 | -0.0060 [-0.0462, +0.0333] |
| rope-log-full | window | 512 | 2.4790 | -0.0825 [-0.1823, +0.0326] |
| rope-log-full | window | 1024 | 2.3512 | -0.2165 [-0.3833, -0.0606] |
| rope-log-window | full | 16 | 2.3354 | -0.0133 [-0.0635, +0.0446] |
| rope-log-window | full | 32 | 2.3222 | +0.0116 [-0.0229, +0.0548] |
| rope-log-window | full | 64 | 2.3065 | -0.0167 [-0.0536, +0.0230] |
| rope-log-window | full | 128 | 2.3060 | -0.0183 [-0.0674, +0.0222] |
| rope-log-window | full | 256 | 2.3736 | +0.0538 [+0.0001, +0.1149] |
| rope-log-window | full | 512 | 2.6886 | +0.1272 [-0.0238, +0.2834] |
| rope-log-window | full | 1024 | 2.6561 | +0.0884 [-0.0504, +0.2344] |
| rope-log-window | window | 16 | 2.3354 | -0.0133 [-0.0635, +0.0446] |
| rope-log-window | window | 32 | 2.3222 | +0.0116 [-0.0229, +0.0548] |
| rope-log-window | window | 64 | 2.3065 | -0.0167 [-0.0536, +0.0230] |
| rope-log-window | window | 128 | 2.2974 | -0.0269 [-0.0635, +0.0082] |
| rope-log-window | window | 256 | 2.2887 | -0.0311 [-0.0634, +0.0005] |
| rope-log-window | window | 512 | 2.4797 | -0.0818 [-0.1818, +0.0310] |
| rope-log-window | window | 1024 | 2.2890 | -0.2786 [-0.4076, -0.1495] |

#### Successor counting

Strongest eligible baseline selected on tuning seeds: `alibi`.

| Method | Digit range | Context | Position bucket | Accuracy | Difference vs alibi [95% CI] | Digit-only accuracy | Digit-only difference vs alibi [95% CI] |
|---|---|---:|---|---:|---:|---:|---:|
| abs | 3_digit_train_range_cyclic | 256 | 0_63 | 0.4449 | -0.2545 [-0.3219, -0.1961] | 0.3130 | -0.2904 [-0.3826, -0.2099] |
| abs | 3_digit_train_range_cyclic | 256 | 64_255 | 0.5142 | -0.2799 [-0.3322, -0.2322] | 0.3804 | -0.3454 [-0.4147, -0.2805] |
| abs | 4_digit_held_out | 256 | 0_63 | 0.2988 | -0.1662 [-0.1969, -0.1402] | 0.1606 | -0.1716 [-0.2105, -0.1332] |
| abs | 4_digit_held_out | 256 | 64_255 | 0.3397 | -0.1640 [-0.1831, -0.1428] | 0.1937 | -0.1859 [-0.2115, -0.1589] |
| rope-full | 3_digit_train_range_cyclic | 256 | 0_63 | 0.6420 | -0.0574 [-0.1412, +0.0240] | 0.5503 | -0.0531 [-0.1672, +0.0568] |
| rope-full | 3_digit_train_range_cyclic | 256 | 64_255 | 0.7475 | -0.0465 [-0.1309, +0.0307] | 0.6681 | -0.0577 [-0.1703, +0.0457] |
| rope-full | 3_digit_train_range_cyclic | 512 | 0_63 | 0.6633 | -0.0594 [-0.1293, +0.0102] | 0.5828 | -0.0505 [-0.1438, +0.0401] |
| rope-full | 3_digit_train_range_cyclic | 512 | 256_1023 | 0.5386 | -0.2462 [-0.3169, -0.1827] | 0.4192 | -0.2943 [-0.3917, -0.2063] |
| rope-full | 3_digit_train_range_cyclic | 512 | 64_255 | 0.7430 | -0.0531 [-0.1343, +0.0192] | 0.6630 | -0.0655 [-0.1728, +0.0304] |
| rope-full | 3_digit_train_range_cyclic | 1024 | 0_63 | 0.6527 | -0.0520 [-0.1299, +0.0215] | 0.5648 | -0.0461 [-0.1500, +0.0505] |
| rope-full | 3_digit_train_range_cyclic | 1024 | 256_1023 | 0.3677 | -0.4222 [-0.4872, -0.3646] | 0.2151 | -0.5054 [-0.5894, -0.4306] |
| rope-full | 3_digit_train_range_cyclic | 1024 | 64_255 | 0.7420 | -0.0525 [-0.1391, +0.0252] | 0.6613 | -0.0654 [-0.1797, +0.0372] |
| rope-full | 3_digit_train_range_cyclic | 4096 | 0_63 | 0.6582 | -0.0564 [-0.1357, +0.0158] | 0.5729 | -0.0497 [-0.1568, +0.0471] |
| rope-full | 3_digit_train_range_cyclic | 4096 | 1024_4095 | 0.2394 | -0.5412 [-0.6125, -0.4755] | 0.0681 | -0.6402 [-0.7195, -0.5707] |
| rope-full | 3_digit_train_range_cyclic | 4096 | 256_1023 | 0.3774 | -0.4044 [-0.4758, -0.3423] | 0.2201 | -0.4898 [-0.5847, -0.4075] |
| rope-full | 3_digit_train_range_cyclic | 4096 | 64_255 | 0.7479 | -0.0513 [-0.1315, +0.0236] | 0.6677 | -0.0651 [-0.1717, +0.0340] |
| rope-full | 4_digit_held_out | 256 | 0_63 | 0.3104 | -0.1547 [-0.1846, -0.1246] | 0.2323 | -0.0999 [-0.1470, -0.0603] |
| rope-full | 4_digit_held_out | 256 | 64_255 | 0.3479 | -0.1558 [-0.1816, -0.1307] | 0.2948 | -0.0848 [-0.1284, -0.0465] |
| rope-full | 4_digit_held_out | 512 | 0_63 | 0.3365 | -0.1355 [-0.1707, -0.1025] | 0.2674 | -0.0734 [-0.1203, -0.0361] |
| rope-full | 4_digit_held_out | 512 | 256_1023 | 0.3249 | -0.2013 [-0.2304, -0.1712] | 0.2728 | -0.1350 [-0.1656, -0.1041] |
| rope-full | 4_digit_held_out | 512 | 64_255 | 0.3591 | -0.1657 [-0.2005, -0.1348] | 0.3158 | -0.0903 [-0.1458, -0.0472] |
| rope-full | 4_digit_held_out | 1024 | 0_63 | 0.3418 | -0.1100 [-0.1295, -0.0902] | 0.2775 | -0.0367 [-0.0607, -0.0159] |
| rope-full | 4_digit_held_out | 1024 | 256_1023 | 0.3102 | -0.2101 [-0.2513, -0.1698] | 0.2464 | -0.1539 [-0.2091, -0.0994] |
| rope-full | 4_digit_held_out | 1024 | 64_255 | 0.3457 | -0.1597 [-0.1801, -0.1396] | 0.2988 | -0.0833 [-0.1146, -0.0575] |
| rope-full | 4_digit_held_out | 4096 | 0_63 | 0.3273 | -0.1504 [-0.1775, -0.1205] | 0.2565 | -0.0915 [-0.1263, -0.0579] |
| rope-full | 4_digit_held_out | 4096 | 1024_4095 | 0.2202 | -0.3012 [-0.3404, -0.2675] | 0.1131 | -0.2887 [-0.3246, -0.2469] |
| rope-full | 4_digit_held_out | 4096 | 256_1023 | 0.2970 | -0.2339 [-0.2684, -0.2001] | 0.2285 | -0.1852 [-0.2255, -0.1436] |
| rope-full | 4_digit_held_out | 4096 | 64_255 | 0.3464 | -0.1851 [-0.2173, -0.1540] | 0.2955 | -0.1189 [-0.1584, -0.0828] |
| rope-linear | 3_digit_train_range_cyclic | 256 | 0_63 | 0.3316 | -0.3678 [-0.4449, -0.2998] | 0.2674 | -0.3359 [-0.4286, -0.2534] |
| rope-linear | 3_digit_train_range_cyclic | 256 | 64_255 | 0.4292 | -0.3649 [-0.4310, -0.3041] | 0.3806 | -0.3451 [-0.4272, -0.2746] |
| rope-linear | 3_digit_train_range_cyclic | 512 | 0_63 | 0.3508 | -0.3719 [-0.4402, -0.3141] | 0.2956 | -0.3378 [-0.4250, -0.2604] |
| rope-linear | 3_digit_train_range_cyclic | 512 | 256_1023 | 0.3990 | -0.3858 [-0.4396, -0.3345] | 0.3496 | -0.3639 [-0.4351, -0.2988] |
| rope-linear | 3_digit_train_range_cyclic | 512 | 64_255 | 0.4236 | -0.3725 [-0.4327, -0.3183] | 0.3762 | -0.3523 [-0.4286, -0.2864] |
| rope-linear | 3_digit_train_range_cyclic | 1024 | 0_63 | 0.3285 | -0.3762 [-0.4443, -0.3156] | 0.2609 | -0.3500 [-0.4331, -0.2766] |
| rope-linear | 3_digit_train_range_cyclic | 1024 | 256_1023 | 0.3941 | -0.3958 [-0.4547, -0.3401] | 0.3514 | -0.3691 [-0.4461, -0.2982] |
| rope-linear | 3_digit_train_range_cyclic | 1024 | 64_255 | 0.4215 | -0.3729 [-0.4382, -0.3066] | 0.3733 | -0.3534 [-0.4345, -0.2742] |
| rope-linear | 3_digit_train_range_cyclic | 4096 | 0_63 | 0.3496 | -0.3650 [-0.4295, -0.3090] | 0.2859 | -0.3367 [-0.4151, -0.2667] |
| rope-linear | 3_digit_train_range_cyclic | 4096 | 1024_4095 | 0.3467 | -0.4340 [-0.4948, -0.3778] | 0.2945 | -0.4137 [-0.4861, -0.3457] |
| rope-linear | 3_digit_train_range_cyclic | 4096 | 256_1023 | 0.3845 | -0.3973 [-0.4558, -0.3421] | 0.3368 | -0.3731 [-0.4481, -0.3043] |
| rope-linear | 3_digit_train_range_cyclic | 4096 | 64_255 | 0.4261 | -0.3731 [-0.4329, -0.3184] | 0.3720 | -0.3609 [-0.4365, -0.2943] |
| rope-linear | 4_digit_held_out | 256 | 0_63 | 0.2734 | -0.1916 [-0.2246, -0.1580] | 0.2064 | -0.1258 [-0.1612, -0.0898] |
| rope-linear | 4_digit_held_out | 256 | 64_255 | 0.3105 | -0.1932 [-0.2250, -0.1621] | 0.2446 | -0.1349 [-0.1660, -0.1077] |
| rope-linear | 4_digit_held_out | 512 | 0_63 | 0.2756 | -0.1965 [-0.2254, -0.1688] | 0.2166 | -0.1242 [-0.1610, -0.0900] |
| rope-linear | 4_digit_held_out | 512 | 256_1023 | 0.3214 | -0.2048 [-0.2342, -0.1748] | 0.2740 | -0.1339 [-0.1673, -0.1023] |
| rope-linear | 4_digit_held_out | 512 | 64_255 | 0.3240 | -0.2008 [-0.2415, -0.1638] | 0.2719 | -0.1342 [-0.1896, -0.0842] |
| rope-linear | 4_digit_held_out | 1024 | 0_63 | 0.2857 | -0.1660 [-0.1961, -0.1348] | 0.2287 | -0.0856 [-0.1181, -0.0538] |
| rope-linear | 4_digit_held_out | 1024 | 256_1023 | 0.3531 | -0.1672 [-0.1895, -0.1456] | 0.3220 | -0.0783 [-0.1044, -0.0530] |
| rope-linear | 4_digit_held_out | 1024 | 64_255 | 0.3320 | -0.1734 [-0.2083, -0.1386] | 0.2751 | -0.1071 [-0.1527, -0.0649] |
| rope-linear | 4_digit_held_out | 4096 | 0_63 | 0.2516 | -0.2262 [-0.2574, -0.1826] | 0.1814 | -0.1666 [-0.2010, -0.1235] |
| rope-linear | 4_digit_held_out | 4096 | 1024_4095 | 0.3585 | -0.1630 [-0.1784, -0.1459] | 0.3381 | -0.0637 [-0.0850, -0.0457] |
| rope-linear | 4_digit_held_out | 4096 | 256_1023 | 0.3436 | -0.1873 [-0.2107, -0.1627] | 0.3098 | -0.1039 [-0.1362, -0.0743] |
| rope-linear | 4_digit_held_out | 4096 | 64_255 | 0.3257 | -0.2057 [-0.2300, -0.1820] | 0.2699 | -0.1444 [-0.1773, -0.1156] |
| rope-ntk | 3_digit_train_range_cyclic | 256 | 0_63 | 0.6744 | -0.0250 [-0.1098, +0.0549] | 0.6190 | +0.0156 [-0.0951, +0.1182] |
| rope-ntk | 3_digit_train_range_cyclic | 256 | 64_255 | 0.6046 | -0.1895 [-0.2558, -0.1276] | 0.5938 | -0.1319 [-0.2158, -0.0543] |
| rope-ntk | 3_digit_train_range_cyclic | 512 | 0_63 | 0.6895 | -0.0332 [-0.1094, +0.0377] | 0.6432 | +0.0099 [-0.0880, +0.1003] |
| rope-ntk | 3_digit_train_range_cyclic | 512 | 256_1023 | 0.5018 | -0.2831 [-0.3579, -0.2119] | 0.4779 | -0.2357 [-0.3218, -0.1545] |
| rope-ntk | 3_digit_train_range_cyclic | 512 | 64_255 | 0.6040 | -0.1921 [-0.2582, -0.1286] | 0.5957 | -0.1328 [-0.2167, -0.0543] |
| rope-ntk | 3_digit_train_range_cyclic | 1024 | 0_63 | 0.6705 | -0.0342 [-0.1141, +0.0361] | 0.6198 | +0.0089 [-0.0927, +0.0974] |
| rope-ntk | 3_digit_train_range_cyclic | 1024 | 256_1023 | 0.4882 | -0.3017 [-0.3759, -0.2303] | 0.4632 | -0.2573 [-0.3487, -0.1711] |
| rope-ntk | 3_digit_train_range_cyclic | 1024 | 64_255 | 0.5961 | -0.1984 [-0.2686, -0.1331] | 0.5884 | -0.1383 [-0.2247, -0.0562] |
| rope-ntk | 3_digit_train_range_cyclic | 4096 | 0_63 | 0.6926 | -0.0221 [-0.0963, +0.0422] | 0.6479 | +0.0253 [-0.0682, +0.1055] |
| rope-ntk | 3_digit_train_range_cyclic | 4096 | 1024_4095 | 0.2870 | -0.4936 [-0.5603, -0.4303] | 0.2087 | -0.4995 [-0.5753, -0.4296] |
| rope-ntk | 3_digit_train_range_cyclic | 4096 | 256_1023 | 0.4782 | -0.3036 [-0.3784, -0.2306] | 0.4550 | -0.2549 [-0.3468, -0.1690] |
| rope-ntk | 3_digit_train_range_cyclic | 4096 | 64_255 | 0.6013 | -0.1979 [-0.2609, -0.1379] | 0.5911 | -0.1417 [-0.2180, -0.0696] |
| rope-ntk | 4_digit_held_out | 256 | 0_63 | 0.2951 | -0.1699 [-0.2000, -0.1416] | 0.2030 | -0.1292 [-0.1696, -0.0969] |
| rope-ntk | 4_digit_held_out | 256 | 64_255 | 0.3127 | -0.1910 [-0.2169, -0.1652] | 0.2606 | -0.1190 [-0.1534, -0.0899] |
| rope-ntk | 4_digit_held_out | 512 | 0_63 | 0.3107 | -0.1613 [-0.1896, -0.1303] | 0.2327 | -0.1081 [-0.1436, -0.0722] |
| rope-ntk | 4_digit_held_out | 512 | 256_1023 | 0.3320 | -0.1942 [-0.2204, -0.1683] | 0.2964 | -0.1115 [-0.1384, -0.0870] |
| rope-ntk | 4_digit_held_out | 512 | 64_255 | 0.3262 | -0.1986 [-0.2307, -0.1686] | 0.2844 | -0.1217 [-0.1614, -0.0887] |
| rope-ntk | 4_digit_held_out | 1024 | 0_63 | 0.3352 | -0.1166 [-0.1406, -0.0910] | 0.2597 | -0.0545 [-0.0825, -0.0266] |
| rope-ntk | 4_digit_held_out | 1024 | 256_1023 | 0.3391 | -0.1812 [-0.2098, -0.1532] | 0.3143 | -0.0861 [-0.1181, -0.0546] |
| rope-ntk | 4_digit_held_out | 1024 | 64_255 | 0.3145 | -0.1909 [-0.2093, -0.1716] | 0.2739 | -0.1083 [-0.1318, -0.0855] |
| rope-ntk | 4_digit_held_out | 4096 | 0_63 | 0.3004 | -0.1773 [-0.1980, -0.1568] | 0.2212 | -0.1269 [-0.1551, -0.0988] |
| rope-ntk | 4_digit_held_out | 4096 | 1024_4095 | 0.2489 | -0.2725 [-0.3043, -0.2413] | 0.1987 | -0.2031 [-0.2285, -0.1757] |
| rope-ntk | 4_digit_held_out | 4096 | 256_1023 | 0.3328 | -0.1981 [-0.2180, -0.1766] | 0.3100 | -0.1037 [-0.1355, -0.0735] |
| rope-ntk | 4_digit_held_out | 4096 | 64_255 | 0.3271 | -0.2043 [-0.2296, -0.1771] | 0.2821 | -0.1323 [-0.1637, -0.1016] |
| alibi | 3_digit_train_range_cyclic | 256 | 0_63 | 0.6994 | +0.0000 [+0.0000, +0.0000] | 0.6034 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 256 | 64_255 | 0.7941 | +0.0000 [+0.0000, +0.0000] | 0.7258 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 512 | 0_63 | 0.7227 | +0.0000 [+0.0000, +0.0000] | 0.6333 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 512 | 256_1023 | 0.7848 | +0.0000 [+0.0000, +0.0000] | 0.7135 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 512 | 64_255 | 0.7961 | +0.0000 [+0.0000, +0.0000] | 0.7285 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 1024 | 0_63 | 0.7047 | +0.0000 [+0.0000, +0.0000] | 0.6109 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 1024 | 256_1023 | 0.7899 | +0.0000 [+0.0000, +0.0000] | 0.7205 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 1024 | 64_255 | 0.7945 | +0.0000 [+0.0000, +0.0000] | 0.7266 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 4096 | 0_63 | 0.7146 | +0.0000 [+0.0000, +0.0000] | 0.6227 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 4096 | 1024_4095 | 0.7806 | +0.0000 [+0.0000, +0.0000] | 0.7082 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 4096 | 256_1023 | 0.7818 | +0.0000 [+0.0000, +0.0000] | 0.7099 | +0.0000 [+0.0000, +0.0000] |
| alibi | 3_digit_train_range_cyclic | 4096 | 64_255 | 0.7992 | +0.0000 [+0.0000, +0.0000] | 0.7328 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 256 | 0_63 | 0.4650 | +0.0000 [+0.0000, +0.0000] | 0.3322 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 256 | 64_255 | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.3795 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 512 | 0_63 | 0.4721 | +0.0000 [+0.0000, +0.0000] | 0.3408 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 512 | 256_1023 | 0.5262 | +0.0000 [+0.0000, +0.0000] | 0.4078 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 512 | 64_255 | 0.5248 | +0.0000 [+0.0000, +0.0000] | 0.4061 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 1024 | 0_63 | 0.4518 | +0.0000 [+0.0000, +0.0000] | 0.3142 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 1024 | 256_1023 | 0.5203 | +0.0000 [+0.0000, +0.0000] | 0.4003 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 1024 | 64_255 | 0.5054 | +0.0000 [+0.0000, +0.0000] | 0.3822 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 4096 | 0_63 | 0.4777 | +0.0000 [+0.0000, +0.0000] | 0.3481 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 4096 | 1024_4095 | 0.5214 | +0.0000 [+0.0000, +0.0000] | 0.4018 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 4096 | 256_1023 | 0.5309 | +0.0000 [+0.0000, +0.0000] | 0.4137 | +0.0000 [+0.0000, +0.0000] |
| alibi | 4_digit_held_out | 4096 | 64_255 | 0.5314 | +0.0000 [+0.0000, +0.0000] | 0.4144 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | 3_digit_train_range_cyclic | 256 | 0_63 | 0.7158 | +0.0164 [-0.0625, +0.0879] | 0.6456 | +0.0422 [-0.0643, +0.1378] |
| rope-log-full | 3_digit_train_range_cyclic | 256 | 64_255 | 0.8906 | +0.0965 [+0.0301, +0.1546] | 0.8564 | +0.1306 [+0.0422, +0.2081] |
| rope-log-full | 3_digit_train_range_cyclic | 512 | 0_63 | 0.7320 | +0.0094 [-0.0537, +0.0672] | 0.6664 | +0.0331 [-0.0505, +0.1089] |
| rope-log-full | 3_digit_train_range_cyclic | 512 | 256_1023 | 0.6166 | -0.1683 [-0.2275, -0.1119] | 0.5230 | -0.1905 [-0.2622, -0.1225] |
| rope-log-full | 3_digit_train_range_cyclic | 512 | 64_255 | 0.8906 | +0.0945 [+0.0280, +0.1520] | 0.8567 | +0.1282 [+0.0398, +0.2047] |
| rope-log-full | 3_digit_train_range_cyclic | 1024 | 0_63 | 0.7260 | +0.0213 [-0.0502, +0.0857] | 0.6518 | +0.0409 [-0.0557, +0.1273] |
| rope-log-full | 3_digit_train_range_cyclic | 1024 | 256_1023 | 0.4050 | -0.3849 [-0.4455, -0.3296] | 0.2605 | -0.4600 [-0.5306, -0.3936] |
| rope-log-full | 3_digit_train_range_cyclic | 1024 | 64_255 | 0.8882 | +0.0938 [+0.0303, +0.1494] | 0.8554 | +0.1287 [+0.0444, +0.2030] |
| rope-log-full | 3_digit_train_range_cyclic | 4096 | 0_63 | 0.7205 | +0.0059 [-0.0734, +0.0803] | 0.6482 | +0.0255 [-0.0805, +0.1245] |
| rope-log-full | 3_digit_train_range_cyclic | 4096 | 1024_4095 | 0.2463 | -0.5344 [-0.6005, -0.4772] | 0.0866 | -0.6216 [-0.7016, -0.5499] |
| rope-log-full | 3_digit_train_range_cyclic | 4096 | 256_1023 | 0.4057 | -0.3761 [-0.4375, -0.3209] | 0.2608 | -0.4491 [-0.5234, -0.3810] |
| rope-log-full | 3_digit_train_range_cyclic | 4096 | 64_255 | 0.8911 | +0.0919 [+0.0249, +0.1520] | 0.8578 | +0.1250 [+0.0361, +0.2049] |
| rope-log-full | 4_digit_held_out | 256 | 0_63 | 0.3520 | -0.1131 [-0.1424, -0.0889] | 0.2621 | -0.0702 [-0.1061, -0.0388] |
| rope-log-full | 4_digit_held_out | 256 | 64_255 | 0.4040 | -0.0997 [-0.1172, -0.0803] | 0.3321 | -0.0474 [-0.0698, -0.0238] |
| rope-log-full | 4_digit_held_out | 512 | 0_63 | 0.3732 | -0.0988 [-0.1135, -0.0814] | 0.2908 | -0.0500 [-0.0729, -0.0281] |
| rope-log-full | 4_digit_held_out | 512 | 256_1023 | 0.3449 | -0.1813 [-0.2122, -0.1541] | 0.2782 | -0.1296 [-0.1779, -0.0904] |
| rope-log-full | 4_digit_held_out | 512 | 64_255 | 0.4158 | -0.1090 [-0.1273, -0.0896] | 0.3521 | -0.0540 [-0.0802, -0.0277] |
| rope-log-full | 4_digit_held_out | 1024 | 0_63 | 0.3777 | -0.0740 [-0.0996, -0.0510] | 0.3007 | -0.0135 [-0.0470, +0.0175] |
| rope-log-full | 4_digit_held_out | 1024 | 256_1023 | 0.3496 | -0.1707 [-0.2096, -0.1342] | 0.2953 | -0.1050 [-0.1612, -0.0571] |
| rope-log-full | 4_digit_held_out | 1024 | 64_255 | 0.4195 | -0.0859 [-0.1126, -0.0635] | 0.3558 | -0.0264 [-0.0605, -0.0013] |
| rope-log-full | 4_digit_held_out | 4096 | 0_63 | 0.3828 | -0.0949 [-0.1186, -0.0715] | 0.3035 | -0.0446 [-0.0796, -0.0098] |
| rope-log-full | 4_digit_held_out | 4096 | 1024_4095 | 0.2285 | -0.2930 [-0.3210, -0.2624] | 0.1477 | -0.2541 [-0.3014, -0.2078] |
| rope-log-full | 4_digit_held_out | 4096 | 256_1023 | 0.3279 | -0.2031 [-0.2422, -0.1709] | 0.2684 | -0.1453 [-0.2109, -0.0886] |
| rope-log-full | 4_digit_held_out | 4096 | 64_255 | 0.4135 | -0.1179 [-0.1478, -0.0889] | 0.3466 | -0.0678 [-0.1103, -0.0275] |
| rope-window | 3_digit_train_range_cyclic | 256 | 0_63 | 0.6955 | -0.0039 [-0.0928, +0.0762] | 0.6188 | +0.0154 [-0.1013, +0.1203] |
| rope-window | 3_digit_train_range_cyclic | 256 | 64_255 | 0.8590 | +0.0650 [-0.0065, +0.1277] | 0.8144 | +0.0886 [-0.0067, +0.1723] |
| rope-window | 3_digit_train_range_cyclic | 512 | 0_63 | 0.7088 | -0.0139 [-0.0740, +0.0432] | 0.6443 | +0.0109 [-0.0677, +0.0844] |
| rope-window | 3_digit_train_range_cyclic | 512 | 256_1023 | 0.8408 | +0.0560 [-0.0146, +0.1177] | 0.7923 | +0.0787 [-0.0163, +0.1622] |
| rope-window | 3_digit_train_range_cyclic | 512 | 64_255 | 0.8602 | +0.0641 [-0.0108, +0.1305] | 0.8158 | +0.0873 [-0.0122, +0.1757] |
| rope-window | 3_digit_train_range_cyclic | 1024 | 0_63 | 0.7049 | +0.0002 [-0.0857, +0.0768] | 0.6320 | +0.0211 [-0.0938, +0.1227] |
| rope-window | 3_digit_train_range_cyclic | 1024 | 256_1023 | 0.8504 | +0.0606 [-0.0115, +0.1237] | 0.8038 | +0.0834 [-0.0125, +0.1677] |
| rope-window | 3_digit_train_range_cyclic | 1024 | 64_255 | 0.8585 | +0.0640 [-0.0150, +0.1338] | 0.8141 | +0.0874 [-0.0175, +0.1802] |
| rope-window | 3_digit_train_range_cyclic | 4096 | 0_63 | 0.7104 | -0.0043 [-0.0813, +0.0660] | 0.6383 | +0.0156 [-0.0865, +0.1078] |
| rope-window | 3_digit_train_range_cyclic | 4096 | 1024_4095 | 0.8438 | +0.0632 [-0.0103, +0.1276] | 0.7955 | +0.0873 [-0.0106, +0.1735] |
| rope-window | 3_digit_train_range_cyclic | 4096 | 256_1023 | 0.8478 | +0.0659 [-0.0073, +0.1296] | 0.8003 | +0.0904 [-0.0075, +0.1757] |
| rope-window | 3_digit_train_range_cyclic | 4096 | 64_255 | 0.8641 | +0.0649 [-0.0084, +0.1302] | 0.8213 | +0.0885 [-0.0092, +0.1748] |
| rope-window | 4_digit_held_out | 256 | 0_63 | 0.3322 | -0.1328 [-0.1660, -0.1037] | 0.2521 | -0.0802 [-0.1260, -0.0433] |
| rope-window | 4_digit_held_out | 256 | 64_255 | 0.3885 | -0.1152 [-0.1356, -0.0938] | 0.3250 | -0.0545 [-0.0855, -0.0255] |
| rope-window | 4_digit_held_out | 512 | 0_63 | 0.3463 | -0.1258 [-0.1609, -0.0908] | 0.2788 | -0.0620 [-0.1069, -0.0230] |
| rope-window | 4_digit_held_out | 512 | 256_1023 | 0.3925 | -0.1337 [-0.1576, -0.1075] | 0.3299 | -0.0779 [-0.1069, -0.0474] |
| rope-window | 4_digit_held_out | 512 | 64_255 | 0.4046 | -0.1202 [-0.1524, -0.0898] | 0.3457 | -0.0604 [-0.1065, -0.0222] |
| rope-window | 4_digit_held_out | 1024 | 0_63 | 0.3455 | -0.1062 [-0.1236, -0.0900] | 0.2746 | -0.0396 [-0.0654, -0.0193] |
| rope-window | 4_digit_held_out | 1024 | 256_1023 | 0.4017 | -0.1186 [-0.1375, -0.0986] | 0.3385 | -0.0619 [-0.0901, -0.0353] |
| rope-window | 4_digit_held_out | 1024 | 64_255 | 0.3987 | -0.1067 [-0.1346, -0.0769] | 0.3362 | -0.0460 [-0.0782, -0.0148] |
| rope-window | 4_digit_held_out | 4096 | 0_63 | 0.3318 | -0.1459 [-0.1699, -0.1211] | 0.2622 | -0.0859 [-0.1137, -0.0569] |
| rope-window | 4_digit_held_out | 4096 | 1024_4095 | 0.3949 | -0.1266 [-0.1466, -0.1050] | 0.3281 | -0.0737 [-0.1008, -0.0489] |
| rope-window | 4_digit_held_out | 4096 | 256_1023 | 0.4019 | -0.1290 [-0.1543, -0.0961] | 0.3394 | -0.0743 [-0.1105, -0.0339] |
| rope-window | 4_digit_held_out | 4096 | 64_255 | 0.3937 | -0.1378 [-0.1707, -0.1002] | 0.3317 | -0.0827 [-0.1268, -0.0371] |
| rope-log-window | 3_digit_train_range_cyclic | 256 | 0_63 | 0.7859 | +0.0865 [+0.0033, +0.1605] | 0.7315 | +0.1281 [+0.0206, +0.2242] |
| rope-log-window | 3_digit_train_range_cyclic | 256 | 64_255 | 0.9660 | +0.1719 [+0.1086, +0.2271] | 0.9556 | +0.2299 [+0.1457, +0.3031] |
| rope-log-window | 3_digit_train_range_cyclic | 512 | 0_63 | 0.7914 | +0.0688 [+0.0070, +0.1238] | 0.7430 | +0.1096 [+0.0281, +0.1818] |
| rope-log-window | 3_digit_train_range_cyclic | 512 | 256_1023 | 0.9600 | +0.1751 [+0.1109, +0.2309] | 0.9486 | +0.2350 [+0.1499, +0.3089] |
| rope-log-window | 3_digit_train_range_cyclic | 512 | 64_255 | 0.9717 | +0.1757 [+0.1161, +0.2270] | 0.9631 | +0.2346 [+0.1554, +0.3030] |
| rope-log-window | 3_digit_train_range_cyclic | 1024 | 0_63 | 0.7869 | +0.0822 [+0.0059, +0.1480] | 0.7320 | +0.1211 [+0.0177, +0.2099] |
| rope-log-window | 3_digit_train_range_cyclic | 1024 | 256_1023 | 0.9648 | +0.1750 [+0.1124, +0.2281] | 0.9548 | +0.2343 [+0.1512, +0.3055] |
| rope-log-window | 3_digit_train_range_cyclic | 1024 | 64_255 | 0.9681 | +0.1736 [+0.1094, +0.2298] | 0.9592 | +0.2326 [+0.1470, +0.3070] |
| rope-log-window | 3_digit_train_range_cyclic | 4096 | 0_63 | 0.7955 | +0.0809 [+0.0090, +0.1449] | 0.7471 | +0.1245 [+0.0297, +0.2083] |
| rope-log-window | 3_digit_train_range_cyclic | 4096 | 1024_4095 | 0.9530 | +0.1724 [+0.1063, +0.2298] | 0.9401 | +0.2319 [+0.1449, +0.3074] |
| rope-log-window | 3_digit_train_range_cyclic | 4096 | 256_1023 | 0.9603 | +0.1785 [+0.1140, +0.2345] | 0.9486 | +0.2387 [+0.1531, +0.3132] |
| rope-log-window | 3_digit_train_range_cyclic | 4096 | 64_255 | 0.9736 | +0.1744 [+0.1108, +0.2291] | 0.9661 | +0.2332 [+0.1487, +0.3061] |
| rope-log-window | 4_digit_held_out | 256 | 0_63 | 0.3543 | -0.1107 [-0.1437, -0.0785] | 0.2755 | -0.0567 [-0.0971, -0.0166] |
| rope-log-window | 4_digit_held_out | 256 | 64_255 | 0.4487 | -0.0550 [-0.0832, -0.0271] | 0.3697 | -0.0099 [-0.0412, +0.0190] |
| rope-log-window | 4_digit_held_out | 512 | 0_63 | 0.3896 | -0.0824 [-0.1070, -0.0574] | 0.3181 | -0.0227 [-0.0503, +0.0046] |
| rope-log-window | 4_digit_held_out | 512 | 256_1023 | 0.4522 | -0.0740 [-0.1021, -0.0454] | 0.3737 | -0.0341 [-0.0668, -0.0013] |
| rope-log-window | 4_digit_held_out | 512 | 64_255 | 0.4502 | -0.0746 [-0.1096, -0.0394] | 0.3759 | -0.0302 [-0.0677, +0.0093] |
| rope-log-window | 4_digit_held_out | 1024 | 0_63 | 0.3797 | -0.0721 [-0.0994, -0.0504] | 0.3073 | -0.0069 [-0.0439, +0.0227] |
| rope-log-window | 4_digit_held_out | 1024 | 256_1023 | 0.4576 | -0.0627 [-0.0893, -0.0313] | 0.3780 | -0.0223 [-0.0511, +0.0112] |
| rope-log-window | 4_digit_held_out | 1024 | 64_255 | 0.4563 | -0.0491 [-0.0778, -0.0227] | 0.3830 | +0.0008 [-0.0353, +0.0303] |
| rope-log-window | 4_digit_held_out | 4096 | 0_63 | 0.3672 | -0.1105 [-0.1322, -0.0895] | 0.2903 | -0.0578 [-0.0925, -0.0293] |
| rope-log-window | 4_digit_held_out | 4096 | 1024_4095 | 0.4418 | -0.0796 [-0.1053, -0.0540] | 0.3629 | -0.0389 [-0.0703, -0.0093] |
| rope-log-window | 4_digit_held_out | 4096 | 256_1023 | 0.4459 | -0.0850 [-0.1170, -0.0560] | 0.3665 | -0.0472 [-0.0878, -0.0129] |
| rope-log-window | 4_digit_held_out | 4096 | 64_255 | 0.4333 | -0.0982 [-0.1305, -0.0681] | 0.3526 | -0.0618 [-0.1012, -0.0245] |

| Method | Four-digit PPL at 256 | Difference vs alibi [95% CI] | Relative difference of mean PPL |
|---|---:|---:|---:|
| abs | 7.1906 | +0.9485 [-0.2141, +1.8175] | +15.19% |
| rope-full | 7.3225 | +1.0804 [-0.2061, +2.2661] | +17.31% |
| rope-linear | 15.0446 | +8.8024 [+6.9100, +11.3728] | +141.02% |
| rope-ntk | 12.4941 | +6.2519 [+4.8307, +7.8674] | +100.16% |
| alibi | 6.2421 | +0.0000 [+0.0000, +0.0000] | +0.00% |
| rope-log-full | 7.8917 | +1.6496 [+0.4843, +2.7197] | +26.43% |
| rope-window | 6.9537 | +0.7116 [-0.5424, +1.7657] | +11.40% |
| rope-log-window | 8.5660 | +2.3238 [+0.8372, +3.7843] | +37.23% |

#### Tuned character context benefit

Strongest eligible character baseline selected on tuning seeds: `alibi`.

| Method | Prefix | Late-position NLL | Difference vs alibi [95% CI] |
|---|---:|---:|---:|
| rope-full | 16 | 2.3632 | +0.0599 [-0.1075, +0.2391] |
| rope-full | 32 | 2.3665 | +0.0418 [-0.1220, +0.2097] |
| rope-full | 64 | 2.3703 | +0.0574 [-0.1249, +0.2348] |
| rope-full | 128 | 2.3822 | +0.0784 [-0.0938, +0.2531] |
| rope-full | 256 | 2.3892 | +0.0891 [-0.0905, +0.2766] |
| rope-full | 512 | 2.8524 | +0.5531 [+0.2605, +0.8961] |
| rope-full | 1024 | 2.8667 | +0.5677 [+0.2800, +0.9067] |
| rope-linear | 16 | 2.4964 | +0.1930 [-0.0178, +0.4069] |
| rope-linear | 32 | 2.5245 | +0.1998 [-0.0571, +0.4800] |
| rope-linear | 64 | 2.5185 | +0.2056 [-0.0438, +0.4768] |
| rope-linear | 128 | 2.5002 | +0.1964 [-0.0313, +0.4406] |
| rope-linear | 256 | 2.4829 | +0.1828 [-0.0398, +0.4258] |
| rope-linear | 512 | 2.4916 | +0.1923 [-0.0219, +0.4364] |
| rope-linear | 1024 | 2.4962 | +0.1972 [-0.0131, +0.4379] |
| rope-ntk | 16 | 2.2930 | -0.0104 [-0.1605, +0.1379] |
| rope-ntk | 32 | 2.3079 | -0.0169 [-0.1655, +0.1148] |
| rope-ntk | 64 | 2.3111 | -0.0019 [-0.1587, +0.1303] |
| rope-ntk | 128 | 2.3149 | +0.0111 [-0.1455, +0.1499] |
| rope-ntk | 256 | 2.3808 | +0.0808 [-0.1109, +0.2727] |
| rope-ntk | 512 | 2.3694 | +0.0701 [-0.1100, +0.2440] |
| rope-ntk | 1024 | 2.3732 | +0.0742 [-0.1017, +0.2450] |
| alibi | 16 | 2.3034 | +0.0000 [+0.0000, +0.0000] |
| alibi | 32 | 2.3247 | +0.0000 [+0.0000, +0.0000] |
| alibi | 64 | 2.3129 | +0.0000 [+0.0000, +0.0000] |
| alibi | 128 | 2.3038 | +0.0000 [+0.0000, +0.0000] |
| alibi | 256 | 2.3001 | +0.0000 [+0.0000, +0.0000] |
| alibi | 512 | 2.2994 | +0.0000 [+0.0000, +0.0000] |
| alibi | 1024 | 2.2990 | +0.0000 [+0.0000, +0.0000] |
| rope-log-full | 16 | 2.3376 | +0.0342 [-0.0635, +0.1320] |
| rope-log-full | 32 | 2.3357 | +0.0110 [-0.1064, +0.1102] |
| rope-log-full | 64 | 2.2911 | -0.0218 [-0.1266, +0.0774] |
| rope-log-full | 128 | 2.3149 | +0.0111 [-0.0990, +0.1112] |
| rope-log-full | 256 | 2.3246 | +0.0246 [-0.0992, +0.1377] |
| rope-log-full | 512 | 2.4874 | +0.1881 [+0.0697, +0.2894] |
| rope-log-full | 1024 | 2.4817 | +0.1827 [+0.0487, +0.3030] |
| rope-window | 16 | 2.2401 | -0.0633 [-0.2106, +0.0732] |
| rope-window | 32 | 2.2523 | -0.0724 [-0.2135, +0.0437] |
| rope-window | 64 | 2.2431 | -0.0698 [-0.2082, +0.0463] |
| rope-window | 128 | 2.2431 | -0.0607 [-0.1908, +0.0515] |
| rope-window | 256 | 2.2521 | -0.0479 [-0.1871, +0.0738] |
| rope-window | 512 | 2.6998 | +0.4005 [+0.1071, +0.7203] |
| rope-window | 1024 | 2.2548 | -0.0442 [-0.1833, +0.0695] |
| rope-log-window | 16 | 2.3355 | +0.0321 [-0.1547, +0.1825] |
| rope-log-window | 32 | 2.3220 | -0.0027 [-0.1818, +0.1305] |
| rope-log-window | 64 | 2.3067 | -0.0062 [-0.1667, +0.1197] |
| rope-log-window | 128 | 2.2975 | -0.0063 [-0.1628, +0.1175] |
| rope-log-window | 256 | 2.2887 | -0.0113 [-0.1597, +0.1090] |
| rope-log-window | 512 | 2.4797 | +0.1804 [+0.0432, +0.3272] |
| rope-log-window | 1024 | 2.2892 | -0.0098 [-0.1702, +0.1240] |

| Method | PPL at 256 | Difference vs alibi [95% CI] | Relative difference of mean PPL |
|---|---:|---:|---:|
| abs | 14.3683 | +3.9241 [+3.3921, +4.5807] | +37.57% |
| rope-full | 12.3142 | +1.8701 [+1.4003, +2.4421] | +17.91% |
| rope-linear | 14.0831 | +3.6389 [+3.1010, +4.2088] | +34.84% |
| rope-ntk | 11.7217 | +1.2775 [+0.9970, +1.5688] | +12.23% |
| alibi | 10.4442 | +0.0000 [+0.0000, +0.0000] | +0.00% |
| rope-log-full | 11.4062 | +0.9621 [+0.7142, +1.2120] | +9.21% |
| rope-window | 10.6657 | +0.2216 [-0.0046, +0.4661] | +2.12% |
| rope-log-window | 10.9439 | +0.4997 [+0.2760, +0.7330] | +4.78% |

### Related work

| Technique | Earlier work |
|---|---|
| Length-dependent attention-logit scaling | [Chiang and Cholak, section 5.3](https://arxiv.org/abs/2202.12172); [SSMax](https://arxiv.org/abs/2501.19399) |
| Local attention | [Longformer](https://arxiv.org/abs/2004.05150) |
| Attention sinks with a rolling window | [StreamingLLM](https://arxiv.org/abs/2309.17453) |
| Rotary positions | [RoPE](https://arxiv.org/abs/2104.09864) |
| Linear attention-position biases | [ALiBi](https://arxiv.org/abs/2108.12409) |
| Position interpolation | [Position Interpolation](https://arxiv.org/abs/2306.15595) |
| NTK-aware interpolation and extensions | [YaRN](https://arxiv.org/abs/2309.00071) |

These are existing ideas. This repository contributes tested implementations
and evaluation tools; the present evidence does not establish a new method or
long-context ability.

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
- [Windowed attention versus full causal attention](#windowed-attention-versus-full-causal-attention)

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
AMD EPYC 7763 CPU, one thread, float32, batch size 2, four heads, head size 16,
window 64, four sink tokens, and query chunk size 128. Each measured case ran
in a fresh process with two warmups and three timed iterations. The attention
inputs Q/K/V require gradients; each backward computes Q/K/V and `raw_tau`
gradients, and the worker verifies that Q/K/V gradients were produced. Input
and parameter gradients are cleared between iterations. CPU memory is process
peak RSS, including Python and PyTorch startup; it is not incremental tensor
allocation. CUDA was unavailable. The dense attention and ALiBi limits prevent
large quadratic references from exceeding the configured caps. The mask
equality check is capped separately.

Dense builders return the same dense tensors as before; chunking bounds
intermediate construction allocations but does not make the returned tensor
sparse. Timings and peak RSS are reference -> chunked; speedup is reference time
divided by chunked time.

| Builder | Length | Time (ms) | Peak RSS (MB) | Speedup | Max error / RMSE |
|---|---:|---:|---:|---:|---:|
| Mask | 128 | 0.097 -> 0.113 | 536.9 -> 538.3 | 0.86x | 0 / 0 |
| Mask | 256 | 0.170 -> 0.260 | 537.3 -> 538.3 | 0.65x | 0 / 0 |
| Mask | 512 | 0.810 -> 0.678 | 538.5 -> 538.8 | 1.20x | 0 / 0 |
| Mask | 1,024 | 1.711 -> 3.363 | 543.8 -> 541.1 | 0.51x | 0 / 0 |
| Mask | 2,048 | 8.954 -> 11.972 | 561.1 -> 550.4 | 0.75x | 0 / 0 |
| Mask | 4,096 | 45.230 -> 31.571 | 633.2 -> 571.8 | 1.43x | 0 / 0 |
| Mask | 8,192 | 259.402 -> 201.944 | 857.3 -> 668.2 | 1.28x | 0 / 0 |
| Mask | 16,384 | 1,099.071 -> 670.723 | 1,817.4 -> 1,056.4 | 1.64x | Not compared |
| ALiBi | 128 | 0.109 -> 0.227 | 539.1 -> 539.9 | 0.48x | 0 / 0 |
| ALiBi | 256 | 0.688 -> 0.614 | 541.8 -> 543.1 | 1.12x | 0 / 0 |
| ALiBi | 512 | 5.395 -> 2.981 | 551.6 -> 551.0 | 1.81x | 0 / 0 |
| ALiBi | 1,024 | 12.807 -> 10.748 | 586.7 -> 574.7 | 1.19x | 0 / 0 |
| ALiBi | 2,048 | 101.399 -> 55.625 | 682.7 -> 674.3 | 1.82x | 0 / 0 |
| ALiBi | 4,096 | 482.644 -> 227.816 | 1,114.8 -> 1,065.1 | 2.12x | 0 / 0 |
| ALiBi | 8,192, 16,384 | Not run | Not run | — | Over configured dense-reference cap |

Builder timings vary: chunking is slower at some lengths and does not reduce
total output storage. Its benefits appear at larger lengths, especially for
ALiBi; the measured peak RSS reduction at 16,384 mask length was about 42%.

Attention timings include forward and backward through Q/K/V and `raw_tau`;
memory is process peak RSS. Dense versus sparse output max absolute error /
RMSE is shown where both paths ran. Dense reference attention was capped at
length 2,048 in the primary sweep.

| Length | Dense time / RSS | Sparse time / RSS | Speedup | Max error / RMSE |
|---:|---:|---:|---:|---:|
| 128 | 5.963 ms / 591.3 MB | 2.914 ms / 588.9 MB | 2.05x | 0 / 0 |
| 256 | 14.870 ms / 609.8 MB | 6.406 ms / 595.4 MB | 2.32x | 4.77e-7 / 3.23e-8 |
| 512 | 52.437 ms / 683.8 MB | 25.083 ms / 603.9 MB | 2.09x | 9.54e-7 / 3.76e-8 |
| 1,024 | 291.270 ms / 1,011.2 MB | 28.555 ms / 628.5 MB | 10.20x | 8.34e-7 / 3.89e-8 |
| 2,048 | 1,173.736 ms / 2,268.0 MB | 63.004 ms / 670.6 MB | 18.63x | 7.15e-7 / 3.83e-8 |
| 4,096 | Not in primary sweep | 135.334 ms / 756.2 MB | — | — |
| 8,192 | Not run | 345.601 ms / 953.5 MB | — | — |
| 16,384 | Not run | 962.155 ms / 1,289.8 MB | — | — |

A separate exploratory 4,096 run with one measured iteration completed under
the earlier benchmark setup: dense attention took 2,522.016 ms at 2,695.1 MB
peak RSS; sparse took 52.633 ms at 653.8 MB, with max absolute error 7.15e-7
and RMSE 3.74e-8. That run did not measure Q/K/V gradients and is not
comparable to the regenerated table. A three-iteration repeat was terminated,
so this single-iteration result is not treated as a stable benchmark. No dense
attention comparison was attempted at 8,192 or 16,384; sparse-only timings at
those lengths are not speedup claims.

These are CPU microbenchmarks for batch 2, width 64, and four heads. They show
that the additive sparse attention path avoids the full attention matrix and
matches the dense reference closely in measured comparisons. The chunk bound
describes per-chunk forward temporaries only: autograd retains saved activations
for all chunks, so training memory still grows with sequence length. These
measurements do not establish CUDA performance, CUDA memory use, or end-to-end
transformer throughput. Dense builder APIs remain dense by design; callers
seeking asymptotically reduced attention memory must use the sparse attention
API.

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

## Windowed attention versus full causal attention

This experiment was not present on any of the ten GitHub branches checked at the
start of the Stage 1 work. The harness and records below were recreated locally;
they are not evidence of a published experiment on a remote branch.

The quality experiment trains a 32-wide, two-layer character model with four
heads (head dimension 8) and full causal RoPE + log attention for 500 steps on
each of seeds 42–51. It then evaluates the same weights using full attention or
each window, so the paired differences measure an inference-time change, not
separately trained models. Training contexts are 64 tokens; with a window of
64 or 256 and four sinks, the training visibility is full causal, and log
scaling is one. Evaluation uses two held-out sequences per seed and context.
Differences are windowed minus full-attention perplexity, with 10,000 paired
bootstrap resamples over seeds. Positive values mean worse perplexity.

The timing benchmark measures a single attention operation on random fp32
Q/K/V, with one CPU thread, batch 1, four heads and head dimension 8. It excludes
RoPE, projection, and other model work. Its exact full causal reference includes
QK normalization and per-query log scaling; the windowed cases include QK
normalization and the specified scaling. Forward timing disables autograd;
forward+backward includes the backward pass of the mean squared output.
Timings are medians after one warm-up. These attention-only measurements are
not end-to-end language-model speedups. The longer contexts are included so
that a latency result is not paired with quality measured at another length.

The earlier 500-step full-attention needle trial did not satisfy the requested
training-length learning gate, so no windowed needle-retrieval comparison is
reported here. No acceptable perplexity-loss tolerance was specified.

Environment: AMD EPYC 7763 CPU, PyTorch 2.15.0+cu132, Python 3.14.2, one
thread, no CUDA. The shared host showed substantial timing variation; raw
individual samples are retained. The quality dataset is the 90/10 tiny
Shakespeare split with SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

```bash
python benchmarks/length_generalization.py --task char \
  --variants rope-log rope-window64 rope-log-window64 rope-window256 rope-log-window256 \
  --paired-window-eval --steps 500 --seeds 42 43 44 45 46 47 48 49 50 51 \
  --init-modes baseline --schedulers toolkit \
  --char-lengths 64 128 256 512 1024 4096 8192 16384 \
  --output benchmarks/window-quality.jsonl
python benchmarks/window_speed.py --head-dimension 8 \
  --lengths 64 128 256 512 1024 4096 8192 16384 \
  --windows 64 256 --repeats 5 > benchmarks/window-speed-d8.json
```

The [per-seed quality records](./window-quality.jsonl) and
[timing samples](./window-speed-d8.json) are included. Both speed columns are
ratios of exact full causal RoPE + log attention time to windowed attention time
for the indicated mode and context; the RoPE transformation itself is omitted
from both timings. Full and window perplexities are means over the same ten
seeds; the bracketed interval is the paired 95% confidence interval.

| Context | Window | Scaling | Full PPL | Window PPL | Difference [95% CI] | Forward ratio | Forward+backward ratio |
|---:|---:|---|---:|---:|---:|---:|---:|
| 64 | 64 | none | 10.52 | 10.52 | +0.00 [+0.00, +0.00] | 0.75 | 1.33 |
| 64 | 64 | log | 10.52 | 10.52 | +0.00 [+0.00, +0.00] | 0.72 | 1.33 |
| 64 | 256 | none | 10.52 | 10.52 | +0.00 [+0.00, +0.00] | 0.73 | 1.28 |
| 64 | 256 | log | 10.52 | 10.52 | +0.00 [+0.00, +0.00] | 0.54 | 1.17 |
| 128 | 64 | none | 10.83 | 10.70 | -0.13 [-0.18, -0.07] | 0.64 | 0.89 |
| 128 | 64 | log | 10.83 | 10.70 | -0.13 [-0.18, -0.08] | 0.66 | 1.17 |
| 128 | 256 | none | 10.83 | 10.86 | +0.03 [+0.01, +0.05] | 1.04 | 1.37 |
| 128 | 256 | log | 10.83 | 10.83 | +0.00 [+0.00, +0.00] | 0.92 | 1.03 |
| 256 | 64 | none | 11.70 | 11.15 | -0.55 [-0.64, -0.45] | 0.69 | 0.94 |
| 256 | 64 | log | 11.70 | 11.14 | -0.55 [-0.65, -0.46] | 0.62 | 0.73 |
| 256 | 256 | none | 11.70 | 11.78 | +0.08 [+0.02, +0.13] | 0.75 | 1.45 |
| 256 | 256 | log | 11.70 | 11.70 | +0.00 [+0.00, +0.00] | 0.62 | 0.91 |
| 512 | 64 | none | 13.02 | 11.49 | -1.54 [-1.77, -1.32] | 1.01 | 1.93 |
| 512 | 64 | log | 13.02 | 11.48 | -1.54 [-1.77, -1.32] | 1.32 | 1.87 |
| 512 | 256 | none | 13.02 | 12.65 | -0.37 [-0.51, -0.23] | 1.01 | 1.34 |
| 512 | 256 | log | 13.02 | 12.58 | -0.44 [-0.54, -0.33] | 0.66 | 1.12 |
| 1024 | 64 | none | 13.89 | 11.34 | -2.56 [-2.88, -2.28] | 2.34 | 2.50 |
| 1024 | 64 | log | 13.89 | 11.33 | -2.56 [-2.88, -2.28] | 2.31 | 2.65 |
| 1024 | 256 | none | 13.89 | 12.83 | -1.06 [-1.38, -0.78] | 1.59 | 1.89 |
| 1024 | 256 | log | 13.89 | 12.79 | -1.11 [-1.42, -0.83] | 1.36 | 1.75 |
| 4096 | 64 | none | 16.05 | 11.65 | -4.40 [-4.93, -3.89] | 5.62 | 4.44 |
| 4096 | 64 | log | 16.05 | 11.64 | -4.40 [-4.93, -3.89] | 7.90 | 6.93 |
| 4096 | 256 | none | 16.05 | 13.42 | -2.63 [-3.22, -2.07] | 5.72 | 4.47 |
| 4096 | 256 | log | 16.05 | 13.36 | -2.69 [-3.27, -2.14] | 5.63 | 4.55 |
| 8192 | 64 | none | 16.36 | 11.55 | -4.81 [-5.47, -4.17] | 13.01 | 12.63 |
| 8192 | 64 | log | 16.36 | 11.54 | -4.82 [-5.47, -4.18] | 12.86 | 12.81 |
| 8192 | 256 | none | 16.36 | 13.26 | -3.10 [-3.83, -2.41] | 8.02 | 7.26 |
| 8192 | 256 | log | 16.36 | 13.22 | -3.14 [-3.85, -2.46] | 7.21 | 7.56 |
| 16384 | 64 | none | 16.69 | 11.61 | -5.07 [-5.81, -4.39] | 26.30 | 20.11 |
| 16384 | 64 | log | 16.69 | 11.61 | -5.08 [-5.81, -4.40] | 23.57 | 20.92 |
| 16384 | 256 | none | 16.69 | 13.42 | -3.27 [-4.06, -2.55] | 15.92 | 14.88 |
| 16384 | 256 | log | 16.69 | 13.36 | -3.32 [-4.09, -2.61] | 15.47 | 13.86 |

Against exact full causal RoPE + log attention, the first tested length at
which both forward and forward+backward exceed 9x is 8,192 for window 64 and
16,384 for window 256, with either scaling setting. This is a grid-based
break-even observation, not a guarantee between tested lengths. On this small
task the long-context perplexity differences favor the windowed variants,
but that does not establish a general quality benefit or long-range retrieval.
In particular, full attention at 16,384 tokens is far outside its 64-token
training distribution, and window-64 perplexity is approximately flat beyond
128 tokens. Neither observation establishes use of distant information.

### Optimization against the original streaming path

The following forward-only timings compare the current implementation against
the original streaming implementation at `f924538`, not against full
attention. They use batch 1, four heads, head dimension 64, fp32, four sinks,
one CPU thread and the same host; each value is the median of 21 alternating
measurements. [Raw samples](./window-streaming-comparison.json) include both
scaling modes and retain the earlier seven-sample measurements separately.

| Length | Window | Scaling | Original ms | Current ms | Original/current |
|---:|---:|---|---:|---:|---:|
| 8192 | 64 | none | 80.80 | 56.01 | 1.44 |
| 8192 | 64 | log | 89.92 | 54.87 | 1.64 |
| 8192 | 256 | none | 111.59 | 92.00 | 1.21 |
| 8192 | 256 | log | 132.63 | 90.91 | 1.46 |
| 16384 | 64 | none | 145.18 | 98.67 | 1.47 |
| 16384 | 64 | log | 175.54 | 97.02 | 1.81 |
| 16384 | 256 | none | 252.43 | 176.58 | 1.43 |
| 16384 | 256 | log | 275.04 | 175.75 | 1.56 |

The requested 1.3x optimization target relative to the original streaming path
was **not met consistently**: the window-256 case without log scaling at 8,192
fell short in this run. At both long lengths and both windows, output differences versus the
original streaming implementation were at most `3.58e-6` in fp32; a separate
1,024-token check found maximum Q/K/V/temperature gradient differences below
`1.5e-8` for a mean-squared-output loss. At both long lengths, the maximum
Q/K/V/temperature gradient difference was below `2.5e-10` for a random
weighted-mean output.
