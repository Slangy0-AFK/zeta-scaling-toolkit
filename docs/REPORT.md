# Long-context engineering and negative-results report

## Decision

Stage 1 is **NO-GO**. On the specified held-out successor task, neither
cosine-plus-log variant met the gate against the tuned ALiBi baseline. The
retrieval sanity gate also failed. The 2,000-step run and later stages were
not run.

The repository provides tested attention and position implementations and
resumable experiments. These results do not establish a new long-context
method or reliable long-range ability.

## Earlier work

| Technique | Earlier work |
|---|---|
| Length-dependent attention-logit scaling | [Chiang and Cholak, section 5.3](https://arxiv.org/abs/2202.12172); [SSMax](https://arxiv.org/abs/2501.19399) |
| Local attention and attention sinks | [Longformer](https://arxiv.org/abs/2004.05150); [StreamingLLM](https://arxiv.org/abs/2309.17453) |
| Rotary positions | [RoPE](https://arxiv.org/abs/2104.09864) |
| Linear attention-position biases | [ALiBi](https://arxiv.org/abs/2108.12409) |
| Position scaling | [Position Interpolation](https://arxiv.org/abs/2306.15595); [YaRN](https://arxiv.org/abs/2309.00071) |

These are existing ideas; the contribution here is implementation and
evaluation, not their invention.

## Methods

The base was `f924538e7aef263b6dec9ea24850ab31bf6fa34e`. Models had two
layers, width 32, four heads, training length 256, batch size 16 and 500
training steps. The character controls trained full and 64-token-window
models, each evaluated with full and windowed attention; the window included
four sink tokens. The character dataset was tiny Shakespeare, SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

The successor task trained on numbers 0–999 and evaluated held-out numbers
1000–2999. Absolute positions, RoPE, linearly interpolated RoPE, NTK-scaled
RoPE, ALiBi, and cosine-plus-log variants had two tuning candidates each.
Tuning used seeds 100–109; reporting used seeds 42–51. The strongest eligible
successor baseline was selected on held-out four-digit tuning accuracy at
positions 1024–4095. All intervals below are paired 95% bootstrap intervals
over reporting seeds, using 10,000 resamples.

The planted-needle sanity check used disjoint noise, key and value alphabets,
3,000 training steps and 200 queries per seed. The full RoPE baseline had to
reach 0.9 accuracy at length 256 on at least 8 of 10 seeds before any
retrieval comparison could be interpreted.

The hosted jobs used Python 3.14.10, PyTorch 2.15.0+cpu, one PyTorch thread,
and x86-64 CPUs with four logical CPUs per virtual machine. CPU models varied
between jobs. Exact commands, raw records, per-job hardware, position buckets,
all intervals and the GitHub Actions run are in
[benchmarks/RESULTS.md](../benchmarks/RESULTS.md).

## Results

### Held-out successor counting

The tuning-selected baseline was ALiBi. At context 4096 and positions
1024–4095, counting accuracy includes digit and separator tokens.

| Method | Accuracy | Difference vs ALiBi [95% CI] | PPL at 256 | PPL difference vs ALiBi |
|---|---:|---:|---:|---:|
| ALiBi | 0.5214 | 0 | 6.2421 | 0% |
| RoPE cosine+log, full | 0.2285 | −0.2930 [−0.3210, −0.2624] | 7.8917 | +26.43% |
| RoPE cosine+log, window | 0.4418 | −0.0796 [−0.1053, −0.0540] | 8.5660 | +37.23% |

At context 1024 and positions 256–1023, windowed cosine+log accuracy was
0.4576 versus 0.5203 for the ALiBi baseline, paired difference −0.0627
[−0.0893, −0.0313]. Neither cosine-plus-log variant satisfied the 1%
training-length perplexity tolerance against the ALiBi baseline.

For the supplementary cyclic three-digit stream, which repeats training-range
numbers, windowed cosine+log accuracy at context 4096 and positions 1024–4095
was 0.9530 versus 0.7806 for the ALiBi baseline, difference +0.1724
[+0.1063, +0.2298]. This is not a held-out-number result and was not used to
pass the gate.

### Planted needle

The full RoPE baseline reached the required 0.9 accuracy on 6 of 10 seeds,
below the required 8 of 10. Under the specified gate, the task was **not
learnable at this scale**. No long-context retrieval comparison was made.

### Character controls

This table compares each trained-and-evaluated windowed cosine+log model with
the full-attention RoPE cosine+log control at the same context.

| Context | Full PPL | Window PPL | Difference vs full [95% CI] |
|---:|---:|---:|---:|
| 256 | 11.2403 | 10.9447 | −0.2955 [−0.4673, −0.1563] |
| 1024 | 13.4104 | 10.9626 | −2.4478 [−3.1117, −1.9031] |
| 4096 | 16.1169 | 11.2887 | −4.8282 [−6.5637, −3.5436] |
| 16384 | 16.8442 | 11.2711 | −5.5731 [−7.4702, −4.1100] |

The windowed model's perplexity is roughly flat beyond its training length,
while the full-attention control is evaluated far beyond its training
distribution. Neither observation establishes use of distant tokens. The
prefix-truncation curve is a sanity check with only eight late targets per
seed; it is not a standalone gate criterion.

For the separately tuned character baselines at length 256, windowed
cosine+log perplexity was 10.9439 versus 10.4442 for the tuning-selected ALiBi
baseline, a 4.78% increase. Full cosine+log perplexity was 11.4062 versus
10.4442 for the ALiBi baseline, a 9.21% increase.

## Limitations and what this does not show

- These are small CPU experiments on one architecture and one character corpus.
  Hosted jobs ran on different CPU models; their records are kept separate
  from local measurements.
- The held-out successor test is synthetic. The additional three-digit stream
  uses numbers seen during training.
- Many unadjusted 95% intervals are reported; the comparisons are exploratory.
- A 64-token window plus four sinks exposes at most 68 keys, so its
  log-length factor remains one when trained at length 256.
- The retrieval sanity failure prevents conclusions about long-range retrieval.
  Character perplexity alone cannot establish long-context use.
- No 2,000-step confirmation, larger model, GPU performance, or generalization
  beyond these measured settings is claimed.
