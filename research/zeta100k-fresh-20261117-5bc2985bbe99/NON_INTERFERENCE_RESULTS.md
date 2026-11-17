# Non-Interference Results

For subsequent checks and remaining gates, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Current status: QUALIFICATION PASSED; FULL RUN IN PROGRESS

The standalone preflight and a post-resize preflight run while support services
were available produced identical correctness fields with the same pinned
inputs. Both passed native/adapted parity, sequential cache continuation,
causal invariance, bounded-cache reference parity, finite-value checks, cache
positions, and high-position RoPE checks. See
`qualification/post-resize-20261117/integrated-preflight/CORRECTNESS_RESULTS.json`
and `qualification/standalone/CORRECTNESS_RESULTS.json`.

Additional bounded non-interference checks:

- The preflight correctness JSON records identical numerical outputs and
  stream SHA-256 `c9c43fb35050bc31f8d60e2a9bd600ae960f96d7a7ac1618b41505b2955da322`.
- A fresh checkpoint save/restore at position 4 resumed to position 8 with
  logits bit-identical to uninterrupted execution.
- While a telemetry sender outage was injected, Zeta still produced finite
  logits and passed cache checks. Live MathMesh telemetry separately delivered
  the qualification checkpoint asynchronously.
- During the full run, MathMesh telemetry and the Stability advisory watcher
  run in separate processes; neither receives tensors, logits, or cache state.
  Stability recommendations are recorded as advisory-only and never applied.

The full run has reached checkpoint 6,144 and has so far recorded finite logits,
sequential positions, and bounded cache state. This is prefix evidence, not a
claim that the complete 100K run has passed.

The outage case was injected at the asynchronous telemetry sender boundary; the
persistent services themselves were not stopped during inference. The full
run's end-to-end equivalence and final checkpoint recovery remain pending.
Historical pre-resize blockers below this updated status are retained as
historical evidence only.
