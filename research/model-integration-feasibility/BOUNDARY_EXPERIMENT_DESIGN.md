# Matched boundary experiment design

This is a proposal, not authorization or a result. Freeze a detailed protocol,
datasets and decision rules before reading evaluation outcomes. Existing
benchmark thresholds and historical NO-GO findings are unchanged.

## Conditions and reproducibility

Use the same pinned SmolLM2 checkpoint, tokenizer, chat template, dtype,
device, attention backend, generation parameters and pre-generated
conversation content. Baseline uses normal full-history native inference and
the native full KV cache; it must not silently truncate. Zeta uses the same
model and inputs with the actual per-layer `StreamingKVCache` intervention
from [INTEGRATION_OPTIONS.md](./INTEGRATION_OPTIONS.md). Freeze window and sink
budgets before report evaluation; a pilot value such as window 1,024 and
4 sinks is a **proposal**, not an existing research candidate or an
optimized setting. Count sink/window overlap once. Do not change model weights,
RoPE or the existing experiment's candidate settings.

Generate fixed, independently seeded dialogues with early and recent facts,
instructions, distractors, corrections and multi-hop questions. A separate
versioned oracle resolves current facts and instruction requirements. Balance
question age and position where possible, and stratify their remaining
correlation. Split development and held-out reporting dialogues; freeze
choices using development only. Record hashes for dataset, prompts, source,
tokenizer, weights and environment, plus IDs, state sizes and all failures.

For exact pairing, first use a **scripted trajectory**: the same externally
specified user and assistant turns are teacher-forced into both caches and
answers to fixed probe questions are scored without inserting them into the
future scripted transcript. For each probe, clone or rebuild the same state so
probe generation cannot contaminate the next scripted turn. Separately test
**free-running conversations** with identical user turns and questions but
actual prior outputs fed back into their own system; those histories may
diverge, so they are not identical model inputs and must be reported
separately. Verify tokenizer boundary handling and cache continuation; never
feed oracle labels to the model.

## Boundaries and proposed preregistration

Measure actual tokenized prompt lengths including chat template and reserve
for generation. Use a prespecified ascending grid, for example 512, 1,024,
2,048, 4,096, 8,192, then 12,288 and 16,384 only if supported and approved.
Fix generation length, exact grid, sample counts, resource ceilings and
stop rules before evaluation; an 8,192-token prompt plus generated output is
not the same as an 8,192-token total context.

Before collecting report outcomes, define e.g. a provisional quality rule:
at every tested length, at least 90% exact answers for each of the early-fact,
recent-fact, critical-instruction and update/multi-hop strata, with at least
20 independently labeled probes per stratum. A system is quality-usable at
that length only when all strata meet the rule and required inference
completes. This is a **new proposed rule**, not a change to the existing
0.95/8-of-10 retrieval gate, nor a claim that 20 probes are adequately
powered. Preregister the actual sample size and uncertainty method after
development feasibility and before report evaluation. Report confidence
intervals and denominators; failures and timeouts count against completion and
quality, never silently disappear.

Report separately:

1. **Hard/application limit:** first tested input or total length rejected by
   the model, tokenizer, server or declared policy, with exception and any
   observed truncation. A model configuration is not itself a measured
   rejection. Do not bypass an enforced positional limit silently.
2. **Effective limit:** last tested length meeting the frozen completion and
   quality criteria, and first tested failure; no claim about untested lengths
   or a monotonic boundary without evidence.
3. **Degradation:** per-length and age-stratum exact-match and critical-fact /
   instruction retention, update and multi-hop correctness; compare paired
   differences with confidence intervals over independent conversations.
4. **Resource limits:** first tested out-of-memory, timeout or prespecified
   latency ceiling separately from context overflow and inaccurate answers.
   Record whether a failed length leaves later lengths untested.
5. **Continuity beyond baseline:** demonstrate completion *and* the frozen
   retention/quality criteria at at least one tested length beyond the
   baseline's first measured failure, for the same task and model. Report
   whether the baseline failure was a hard, quality or resource failure; a
   resource-only win is not proof of recall beyond a hard position limit.
   Do not extrapolate between grid points or treat a surviving response as
   retained information.

## Cost accounting

Per turn record actual prompt and generated token counts, unique content,
cumulative tokens processed by the model, cache hits/recomputation and total
Zeta state bytes. Record end-to-end, prefill and decode latency, process peak
RSS, device allocated/reserved memory if applicable, and update overhead.
Record CPU time and measured energy or FLOPs only when instrumentation supports
them; mark unavailable rather than infer them from token count. Count cache
cloning, preprocessing and all auxiliary work for both arms. Distinguish
prefill versus reused KV and document shared or differing transcript lengths.
Compare efficiency only alongside matched task quality.
