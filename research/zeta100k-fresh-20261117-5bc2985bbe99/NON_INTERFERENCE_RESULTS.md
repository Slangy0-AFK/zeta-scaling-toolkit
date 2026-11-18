# Non-Interference Results

For pre-launch history and the final run outcome, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Final status: 100K processing completed

The standalone preflight and a post-resize preflight run while support services
were available produced identical correctness fields with the same pinned
inputs. Both passed native/adapted parity, sequential cache continuation,
causal invariance, bounded-cache reference parity, finite-value checks, cache
positions, and high-position RoPE checks. See
[integrated-preflight/CORRECTNESS_RESULTS.json](qualification/post-resize-20261117/integrated-preflight/CORRECTNESS_RESULTS.json)
and [standalone/CORRECTNESS_RESULTS.json](qualification/standalone/CORRECTNESS_RESULTS.json).

Additional bounded non-interference checks:

- The preflight correctness JSON records identical numerical outputs and
  stream SHA-256 `c9c43fb35050bc31f8d60e2a9bd600ae960f96d7a7ac1618b41505b2955da322`.
- A fresh checkpoint save/restore at position 4 resumed to position 8 with
  logits bit-identical to uninterrupted execution.
- While a telemetry sender outage was injected, Zeta still produced finite
  logits and passed cache checks. Live MathMesh telemetry separately delivered
  the qualification checkpoint asynchronously.
- During the completed 100K run, MathMesh telemetry and the Stability advisory
  watcher ran in separate processes; neither received tensors, logits, or
  cache state. MathMesh delivered 56 records through token 100,000 with no
  failures or drops. Stability recorded 56 recommendations, none applied.

The runner completed all 100,000 contiguous tokens with finite logits and
bounded cache state. All 56 checkpoints passed hash, provenance, retained
position, tensor-shape, and finite-cache validation. The final checkpoint was
not reloaded after completion.

The outage case was injected at the asynchronous telemetry sender boundary; the
persistent services themselves were not stopped during inference. Therefore
the smoke establishes isolation from injected sender failures, not from a
physical service outage during the full run. See
[checkpoint-telemetry-isolation-results.json](qualification/post-resize-20261117/checkpoint-smoke-20261117-02/checkpoint-telemetry-isolation-results.json)
and [FINAL_REPORT.md](FINAL_REPORT.md). Historical pre-resize blockers below
this updated status are retained as historical evidence only.
