# Proposed persistent-context comparison

## Status and question

This is a design, not an implemented benchmark or measured result. The
associative-retrieval task addresses recall from a single input; it does not
establish persistent interaction memory, lower token consumption, lower
compute, or reduced energy. A bounded state can omit, confuse, or overwrite
facts. No improvement, losslessness, or elimination of token exhaustion is
assumed.

## Paired systems

- **Full-history baseline:** send the identical system instruction and the
  complete ordered interaction history, including prior outputs, with each
  new request. No truncation or hidden summarization. If the context limit is
  exceeded, record a capacity failure rather than silently dropping history.
- **Bounded-state system:** send the same instruction and current interaction
  with information retrieved from an explicitly bounded persistent state.
  Update the state after each turn. Freeze the state representation, update,
  retrieval algorithm, token and byte budgets, eviction behavior, prompts,
  model/checkpoint, tokenizer, decoding and error handling before reporting.
  Count every update/retrieval call and resource cost, including auxiliary
  model calls, index maintenance, persistent storage, prompts and outputs.
  Return an explicit miss when information cannot be retrieved.

Both systems receive the same input turns, target facts, task requirements,
model where technically possible, decoding settings, environment and
hardware. Their prior outputs can diverge; log actual outputs and state for
each system, not a fictional shared transcript. Use a pre-generated
interaction sequence independent of either model's answers so divergence
does not change future user inputs. Do not feed the oracle's answer to the
state updater or expose it in query metadata.

## Workload and splits

Generate versioned dialogues containing randomly assigned facts, distractors,
updates that supersede prior facts, and queries at varying lags and
positions. Include absent-fact queries requiring abstention, conflicting
updates, repeated references, and cases where related distractors make
retrieval ambiguous. Labels are computed by an independent versioned
ground-truth ledger, respecting update order. Test the generator and oracle
with counterfactual queries and data-leakage checks before any study.

Predeclare interaction counts, fact density, query mix, state budget and
random seeds; balance query age and fact position as far as possible. Use
independently generated train/development/report partitions and record split
fingerprints. Freeze all candidate choices and budgets using development only.
Run matched report sequences for both systems, with complete interaction IDs,
source/environment manifests, and per-turn records. Analysis should stratify
by turn number, age of fact, updated versus unchanged fact, distractor count,
and explicit state-eviction events. Age and turn number can be correlated;
avoid causal conclusions from unadjusted marginal comparisons.

## Measurements

For each interaction and system, record the numerator and denominator for:

1. **Input tokens processed:** actual tokenizer counts for every model call,
   including repeated history, retrieval context and state updates. Also
   report unique user-content tokens separately to avoid conflating them.
2. **Output tokens:** all generated tokens, including auxiliary calls, with
   task answers and state-update output reported separately.
3. **Retrieval accuracy:** correct retrievals / answerable queries; separately
   record misses, incorrect retrievals and retrieval precision when the
   system exposes retrieval. Mark unavailable instrumentation as unavailable,
   not zero.
4. **Historical fact retention:** correctly answered, still-valid historical
   fact queries / such queries, by age and position. Report incorrect
   superseded-fact answers, omissions, abstentions, and false positives on
   absent-fact queries separately.
5. **Exact-answer accuracy:** exact matches / all answerable queries, using a
   frozen normalization rule; report abstention and failure denominators.
6. **Latency:** end-to-end wall time per interaction and cumulative wall time,
   with model inference, state update, retrieval, network and queue time
   separated when measurable. Report median and tail percentiles as well as
   totals. Document warmup, synchronization and concurrency.
7. **Memory:** peak device allocated and reserved bytes plus process peak
   host memory and persistent-state/index bytes. Report measurement method,
   sampling interval and overlapping allocations.
8. **Compute and energy:** measured device time, hardware utilization,
   profiler FLOPs or provider-reported usage when available, and joules from
   calibrated power telemetry integrated over the complete workload when
   available. Record sampling, idle baseline and measurement scope. Clearly
   label theoretical FLOPs and power-derived estimates; use `unavailable`
   for missing values.

Aggregate totals and per-turn trajectories over matched sequences, paired
differences and uncertainty intervals over independent sequences/seeds. Report
failures, incomplete interactions and capacity limits, rather than dropping
them. Compare efficiency at matched or explicitly reported task quality;
lower token count with degraded answers is a trade-off, not an unconditional
improvement. Track discrepancies and error propagation after incorrect state
updates, with their denominators.

## Proposed first controlled experiment

After separate approval and a working bounded-state implementation, run a
small engineering pilot of 10 independent, pre-generated 20-turn dialogues,
with identical inputs and a fixed state budget selected before execution.
Include early and recent fact queries, one superseding update, an absent
query, and distractors in each dialogue. Run each pair on the same hardware,
using the same tokenizer and decoding configuration. Check per-turn logs,
ground-truth labels, state bounds, complete cost accounting and failure
handling; report raw matched trajectories. This is a feasibility pilot, not a
powered confirmatory study or evidence of an industry-wide improvement.
Pre-register a larger sample and comparison criteria separately before
analyzing a confirmatory reporting set.

## Limitations

The concrete bounded-state implementation, task-specific budgets, production
workloads, statistical power and energy instrumentation remain unspecified
and must be frozen before a confirmatory experiment. Synthetic facts do not
establish natural-dialogue quality. A full-history baseline can exceed the
model context limit; report that boundary explicitly rather than treating an
unanswered interaction as a successful comparison.
