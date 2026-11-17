# Long-context research: interim passive evidence

Snapshot: **2026-11-17 03:52:08 UTC**. This report is a frozen interim
assessment, not a live dashboard. The active experiment is ongoing and was
not restarted, altered, or duplicated. All new data are outside its output
directory. [Raw snapshot](./snapshot.json), [checkpoint intervals](./checkpoints.csv),
and [published reference database](./industry-references.csv) accompany this
report. The snapshot contains full observed progress and health events; inputs
were read separately and can reflect different instants.

## Executive summary and classifications

| Question | Status | Evidence |
|---|---|---|
| 100,000 contiguous tokens processed reliably | **NOT TESTED** | Latest checkpoint in this snapshot: 14,336 processed, last zero-based position 14,335. |
| Valid cache and positions | **PARTIAL** | Nine checkpoints checked; no issues observed at those checkpoints. Gaps between snapshots are not independently observed token by token. |
| Numerical stability | **PARTIAL** | Checkpoint tensors were finite; runner checks each chunk's logits. No independent live-logit observation; completion untested. |
| Beyond 65,537 | **NOT TESTED** | No contiguous run reached it. A synthetic offset/RoPE preflight is not contiguous endurance evidence. |
| Long-range fact retrieval, reasoning, and official benchmarks | **NOT TESTED** | No final query results or official benchmark evaluations available. |
| Original 9x reproduction or long-context transfer | **NOT TESTED** | Historical attention-only measurements exist; no independent repeat or matched end-to-end comparison. |
| High-length limits and first degradation points | **PARTIAL** | The current adapter has a configured ceiling, not a demonstrated capability limit; no universal degradation threshold is established. |

This package separates pipeline endurance, model capability, and performance
efficiency. Success in one category does not establish success in another.

## Configuration and provenance

The active process is PID 35538; at 03:51:39 UTC the passive monitor recorded
it alive, not stalled, with no reported issues. PID 32343 was concurrently
running a separate retrieval sanity job; both workloads compete on the
two-visible-CPU shared AMD EPYC 7763 host. The machine is CPU-only for this run;
GPU model, utilization and VRAM are unavailable. CPU utilization and
host-wide peak RAM were not continuously recorded.

Model: SmolLM2-360M-Instruct, revision
`a10cc1512eabd3dde888204e902eca88bddb4951`, same tokenizer revision.
Precision: float32; batch 1; seed 20261117; sink 64, recent window 64,
chunk 64; checkpoints every 2,048 tokens plus specified milestones. The
model's native context configuration is 8,192 positions. The adapter is
configured for 100,512 positions, including query headroom; 128K and above
cannot be tested by this configuration unchanged. Changing that cap alone
does not validate the architecture, RoPE, memory, or model capability.

Python 3.14.2, PyTorch 2.15.0+cpu, Transformers 5.23.0, Tokenizers 0.23.3,
Safetensors 0.8.0. Snapshot HEAD:
`e717d8377ef97196a7492e551f827ed0a0c315f6`; preflight records prototype
source commit `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`, which is not
asserted to be the exact runtime commit. The tree contained pre-existing and
live uncommitted changes. The manifest hashes the actual runner, adapter, lock,
preflight and diagnostic scripts, and records the preflight model, tokenizer,
corpus and input-stream hashes. These recorded model hashes were not rehashed
for this snapshot. Source files, pinned environment and the local snapshot are
needed for reproduction; Git-ignored checkpoints are not guaranteed to exist
on the remote.

See [protocol](../context-100k-experiment-v2/EXPERIMENT_PROTOCOL.md),
[provenance](../context-100k-experiment-v2/PROVENANCE.md),
[environment](../context-100k-experiment-v2/ENVIRONMENT.md), and the
[runner](../context-100k-experiment-v2/run_experiment.py).

## Pipeline, cache, checkpoint and numerical integrity

The sidecar opened and hashed nine safetensors checkpoints through 14,336,
checked layer shapes, float32 cache finiteness, retained sink/recent positions,
cache bound, metadata, progress bounds and shared retained tensors between
successive checkpoints. It found no reported issues. The last checkpoint's
SHA-256 is `0da6331d43beaa9c1379aa707d1009b75edef514f68c6c5457e1648046d6bba2`.
Each checkpoint's fingerprint and outcome is in the CSV; this is **PASS only
for those specific checks**, not proof of whole-run correctness.

Checkpoint records appear in the expected schedule and their end positions
agree with the configured cache layout. This cannot independently establish
every intervening position or token was processed exactly once, correct
positional-encoding outputs, or the absence of transient errors. The runner
checks chunk logits for finiteness and cache state as it advances; passive
monitoring cannot inspect those logits or independently detect a hidden
transient failure. High-position RoPE passed the recorded synthetic fp64
comparison, including positions 65,536 and 100,000, but does not validate
contiguous processing there. No numerical failure was observed in the
available checks; first-failure position and recovery are **NOT TESTED**.

The monitor's safe-open verifies files are readable, but does not restore them
into the live model. A small preflight reports bit-identical incremental
continuation; that is not an end-to-end resumed-vs-uninterrupted 100K test.
Checkpoint restore, resumed processing, and final cache correctness for this
specific endurance run remain **PARTIAL/NOT TESTED** until independently
verified after completion. Stall detection is a 1,200-second threshold, not a
proof that processing never stalled. A process exit is not automatically
success: inspect its terminal status and final outputs.

## Throughput and memory observations

At 14,336 tokens the runner recorded 4,246.07 seconds of prefill, a reported
cumulative rate of 3.376 tokens/s, peak process RSS 2,560.14 MiB, available
system memory 1,721.58 MiB, and bounded cache 10,518,528 bytes (128 positions).
The first 2,048-token interval was 3.086 tokens/s. Derived checkpoint-interval
rates ranged from 2.583 to 3.793 tokens/s; the minimum is a special **one-token**
8,193 milestone, not a comparable full interval. The latest 2,048-token interval
was 3.221 tokens/s. Interval rates are completed tokens divided by the recorded
interval time; they are neither per-chunk latency nor a controlled rolling
average. Competing workload and host variation prevent attributing changes to
context length. The runner's cumulative formula divides total position by the
current process's prefill time; after a nonzero resume it can overstate actual
throughput. No per-chunk min/max, first-output latency, query latency, peak
system RAM, GPU utilization, compute cost or matched baseline reduction is
available. Peak RSS is process high-water mark, not total machine memory.
Cache bytes divided by processed tokens at this checkpoint is about 734 bytes
per processed token, an amortized ratio rather than allocation per token; the
bounded cache itself stays constant once full.

At the snapshot, nine observations are not independent experimental trials
and yield no valid seed-based confidence interval. Raw records preserve
intervals; do not cherry-pick the maximum as achievable sustained throughput.
The machine had approximately 9.0 GiB free disk shortly before the snapshot.
The configured minimum available memory is 1,024 MiB; competing processes,
additional checkpoints, and a stopped Codespace are ongoing risks.

## Retrieval and accuracy by position and length

Recorded actual fact starts are 128, 2,047, 8,002, 15,998, 24,999, 49,999,
75,001 and 90,000. The requested 1,024 fact was not configured. The runner
constructs the stream deterministically, validates fact spans, and the resume
path verifies the stored token IDs against the rebuilt stream; the recorded
provenance reports these starts. Independent decoded-span verification was not
repeated during the run. By 14,336, the records at 128, 2,047 and 8,002 are
within the processed prefix; the rest are not yet processed.

| Actual start | End-of-100K response | Accuracy / successful and failed counts |
|---:|---|---|
| 128 | NOT TESTED | NOT TESTED |
| 2,047 | NOT TESTED | NOT TESTED |
| 8,002 | NOT TESTED | NOT TESTED |
| 15,998 | NOT TESTED | NOT TESTED |
| 24,999 | NOT TESTED | NOT TESTED |
| 49,999 | NOT TESTED | NOT TESTED |
| 75,001 | NOT TESTED | NOT TESTED |
| 90,000 | NOT TESTED | NOT TESTED |

The query passes the identifier, not the expected value, to a cloned bounded
cache; the original fact list supplies scoring and separate short-prompt
controls, not hidden retrieval for the long-context query. Its existing
`exact_match` is a digit-boundary regex for the expected number anywhere in
the answer, **not strict whole-response exact match**. Preserve both raw output
and scoring definition; calculate any strict metric separately after results
exist. The control explicitly includes the fact and is not the long-range
answer. Direct fact retention and indirect cache persistence must be reported
separately. Accuracy by insertion point, position group, token distance,
context length, distractor, multi-hop and false/missing answer is **NOT
TESTED**. The baseline uses an 8,064-token prefix and cannot provide a matched
100K full-context comparison; neither its results nor 100K queries are yet
available.

## Official benchmark compatibility and planning

No official benchmark was run. Estimates below are feasibility planning, not
measured benchmark runtimes. At the observed 3.376 tokens/s, prefill of one
100K-token case is roughly 8.2 hours on the contended CPU, before decoding,
evaluation, or setup. Official sample counts make full suites far more
expensive. Actual runtime, RAM, tokenizer inflation, serving overhead and
hardware costs must be measured on a separate machine after the active run.
Model-specific tokenization and chat templates must be used; official protocol
token counts, truncation, prompt and output budgets must be retained and
reported. In each row, adapter compatibility is **UNVERIFIED**: the custom
bounded cache/position path is not a drop-in official model or server.

| Benchmark / official version | Task, score, lengths | Requirements and proposed integration | Initial resource assessment |
|---|---|---|---|
| [RULER](https://github.com/NVIDIA/RULER), [paper v3](https://arxiv.org/abs/2404.06654) | 13 configurations across retrieval, tracing, aggregation and QA; recall-based score; 4K–128K tokens, 500 examples per task/length in paper. | Official HF/vLLM/API harness; custom adapter and correct tokenizer/template needed. Paper evaluates open models with BF16 vLLM on 8 A100s; this CPU float32 run is different. | Thousands of long inputs per length; infeasible on the active host while running. Run a separately labeled smoke test first, then predeclare suite and hardware. |
| [MRCR v2](https://github.com/google-deepmind/eval_hub/blob/master/eval_hub/mrcr_v2/README.md) | Multi-round repeated prompts; DeepMind's v2 includes 8-needle variants through 8M. | Pin v2 code/data and report tool access. It is not necessarily identical to [OpenAI MRCR](https://huggingface.co/datasets/openai/mrcr), whose scorer uses a required random prefix and SequenceMatcher; do not conflate versions. Custom chat/long-context adapter needed. | Official hardware not specified; long multi-case evaluation substantially exceeds one 100K run; 1M/8M require new validated positional limits. |
| [NoLiMa](https://github.com/adobe-research/NoLiMa), [paper v3](https://arxiv.org/abs/2502.05167) | Minimal lexical overlap, one/two-hop; character-name accuracy; 250–32K in paper, later up to 128K. | Official data/scorer and local serving adapter; respect non-commercial research license. README serving example uses 8 GPUs of unspecified type; exact requirements depend on model. | Paper uses 7,540 tests per length; not suitable for current CPU. Validate a small official subset, explicitly not a full score. |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172), paper v3 | Position-sensitive multi-document QA and key-value retrieval; answer-containing accuracy; 10/20/30 documents, not fixed 100K-token lengths. | Use paper-linked official code/data, model template and position placement; no specific hardware requirement established. Requires bounded-cache adapter. | Multiple placements per question; first estimate tokenized case sizes and per-case cost. |
| [LongBench v2](https://github.com/THUDM/LongBench), [paper v2](https://arxiv.org/abs/2412.15204) | 503 four-choice reasoning questions, accuracy; contexts 8K–2M **words**, not tokens. | Official prompts and scorer; report middle truncation and direct/CoT condition. README shows vLLM tensor parallel 4 or 8 GPUs, type unspecified; custom adapter/server needed. | Full data is expensive; case lengths often exceed current cap. Measure token lengths and pilot separately before budgeting. |
| [InfiniteBench](https://github.com/OpenBMB/InfiniteBench), [paper v3](https://arxiv.org/abs/2402.13718) | 12 tasks; accuracy or ROUGE depending on task; task-average inputs roughly 43.9K–2,068.6K tokens. | Pin official dataset, per-task evaluator and templates; adapter integration needed; official hardware not specified. Many cases exceed current cap. | Start with token-length inventory and separately labeled short subset; full suite requires much more memory/time than one 100K run. |
| [LOCA-bench](https://github.com/hkust-nlp/LOCA-bench), [paper v1](https://arxiv.org/abs/2602.07962) | Agentic tool tasks; binary final-state accuracy; 8K–256K description tokens, five seeds per length. | Needs official environment, tool-capable ReAct agent and OpenAI-compatible endpoint, not merely text retrieval. Must account for local mock database-backed tools, tool results and context management. Hardware unspecified. | Environment setup and tool calls add unmeasured overhead; beyond-cap cases require separate validation. Tool use differs from no-external-retrieval planted-fact evaluation. |

Pin actual benchmark repository commits, datasets and licenses before future
execution; branch names and published leaderboards can change. Estimates of
official GPU RAM, model-serving support and costs are unknown until a measured
pilot. None of the current experiment's homemade fact queries are official
benchmark results.

## Published references and comparison

The [reference database](./industry-references.csv) preserves source links,
model versions, tested units, conditions and dates where available. Examples:
RULER lists GPT-4-1106-preview, advertised 128K, at 81.2 at 128K and effective
64K under RULER's threshold; NoLiMa reports GPT-4o falling from its 99.3%
short-length base to 69.7% at 32K; Lost in the Middle reports position effects
on its document QA tasks; LongBench v2 reports o1-preview 57.7% overall with
longer reasoning on 503 cases. These are distinct tasks and populations, not
universal failure lengths or directly comparable values. The RULER and
NoLiMa definitions of effective length differ.

| Measurement | Published reference | Zeta native baseline | Zeta bounded result | Comparison |
|---|---|---|---|---|
| Retrieval accuracy | RULER GPT-4-1106-preview: 81.2 at 128K | NOT TESTED | NOT TESTED | NOT COMPARABLE |
| Exact match | NoLiMa GPT-4o: 69.7% at 32K, answer-containing accuracy, not strict exact match | NOT TESTED | NOT TESTED | NOT COMPARABLE |
| Position sensitivity | Lost in the Middle document QA: position-dependent variation | NOT TESTED | NOT TESTED | NOT COMPARABLE |
| Reasoning | LongBench v2 o1-preview: 57.7% overall | NOT TESTED | NOT TESTED | NOT COMPARABLE |
| Throughput | Historical attention operation timings only, see below | NOT TESTED at matched 100K | 3.376 tokens/s, interim CPU prefill | NOT COMPARABLE |
| Peak memory | No matched published measurement established | NOT TESTED at matched 100K | 2,560.14 MiB process RSS so far | NOT COMPARABLE |

## Historical ~9x claim: scope and reproduction plan

[Original report](../../benchmarks/RESULTS.md#windowed-attention-versus-full-causal-attention),
[benchmark](../../benchmarks/window_speed.py), and
[raw samples](../../benchmarks/window-speed-d8.json) first appeared together
in commit `7a728570ac7d073f1e9b3dd7487c79955285f45d`. At 8,192, window 64,
log scaling, full-causal median forward time was 333.644 ms versus windowed
25.950 ms (12.86x); forward+backward 1,437.228 ms versus 112.230 ms (12.81x).
At 4,096 the corresponding ratios were 7.90x and 6.93x. Thus the report's
first *tested grid length* above 9x for both modes was 8,192, not a universal
break-even or guaranteed 9x speedup.

This was one attention operation on random fp32 Q/K/V: batch 1, four heads,
dimension 8, four sinks, one CPU thread, AMD EPYC 7763, PyTorch 2.15.0+cu132,
no CUDA, one warm-up and five samples. Both paths use QK normalization and
the reference uses per-query log scaling. RoPE rotation, projections and full
model work are excluded. Windowing changes attention visibility; numerical
equality to full attention is not generally expected. A separate optimization
comparison against original streaming commit `f924538` measured only
1.21x–1.81x in the reported cases, not the above full-attention ratio.

**Reproduction NOT TESTED.** After the active run and competing workload
finish, use a fresh output directory and the exact command and environment in
the original report, preserve every raw sample and software/source digest,
confirm output/scaling semantics, run repeated independent trials with paired
ordering, and measure peak RSS separately. Compare the historical median
ratios, not only a selected run. Then design a separate controlled,
same-model/tokenizer/precision/cache-policy inference benchmark against an
appropriate native baseline, measuring prefill and decode throughput, latency,
RSS and numerical/output behavior. This transfer test is **NOT TESTED** and
the current 3.376 tokens/s cannot establish a speedup.

## Degradation points and staged future work

| Category | First observed degradation point |
|---|---|
| Pipeline/cache/position integrity | NOT OBSERVED in nine checked checkpoints; intervening positions and later lengths not independently verified. |
| Memory limits | NOT OBSERVED; available memory fell to 1,721.58 MiB in the latest checkpoint while another job ran. No causal length threshold established. |
| Throughput | Interval variability observed, but no predeclared degradation threshold or controlled causal length effect; first point NOT ESTABLISHED. |
| Retrieval accuracy and effective usability | NOT TESTED at any target length. |

After completion and without changing existing artifacts, first audit the
final progress/checkpoints, restore equivalence on a separate copy, final
logit/cache checks, answers, scores, failure logs and baseline. Then design
independent, pre-registered stages at 8K, 16K, 32K, 64K and 100K using
identical inputs and multiple seeds where appropriate. Above 100K, stages are
128K, 256K, 512K and 1M: require a separately reviewed adapter limit change,
RoPE correctness tests, position/dtype checks, storage and resource budgets,
and resume validation before each launch. Higher targets are not supported by
the current 100,512-position configuration and are not capability claims.
Stop at first recorded engineering failure; do not extrapolate a universal
model limit. Predefine accuracy baselines, degradations and confidence intervals
before results; retain every trial and report counts.

The most informative next capability experiment is the existing
query-dependent associative-retrieval sanity gate with distractors and
short-context controls, followed only on success by matched, independently
seeded longer-context comparisons. The existing 500-step successor analysis
did not establish distant retrieval; see
[analysis](../../benchmarks/context-retention-500-analysis.md).

## Refresh and integrity of this package

After the run, capture a **new** directory rather than overwriting this frozen
snapshot:

```bash
python tools/snapshot_context_100k_diagnostics.py \
  --experiment research/context-100k-experiment-v2 \
  --health research/context-100k-diagnostics/health-20261117.jsonl \
  --output-dir research/<new-unique-diagnostics-directory>
```

The command is read-only with respect to the experiment and refuses existing
snapshot filenames. Validate its reported input hashes and separately verify
model files and checkpoint availability in the reproduction environment.
It does not launch a benchmark, recreate an experiment, or guarantee that a
later snapshot includes final retrieval outputs. A stopped or deleted
Codespace terminates its processes; checkpoints must be preserved separately.
