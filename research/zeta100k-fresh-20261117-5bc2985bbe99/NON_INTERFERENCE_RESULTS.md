# Non-Interference Results

For subsequent checks and remaining gates, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Status: NOT ESTABLISHED

The Zeta standalone preflight passed. It measured native/adapted logits,
sequential cache continuation, future-token invariance, bounded-cache
reference parity, finite-value checks, cache positions, and high-position
RoPE correctness. The detailed machine-readable record is
`qualification/standalone/CORRECTNESS_RESULTS.json`.

Examples from that standalone run:

- Native sequential versus adapter: maximum absolute difference 0.0; bitwise
  equal.
- Incremental cache continuation: bit-identical to single prefill.
- Earlier logits under future-token changes: bit-identical.
- Bounded adapter versus its native dense sink/window reference: maximum
  absolute difference `0.000148773193359375`, zero out of tolerance at
  `atol=0.0002`, `rtol=0.0001`.
- Standalone stream SHA-256:
  `c9c43fb35050bc31f8d60e2a9bd600ae960f96d7a7ac1618b41505b2955da322`.

These are standalone Zeta results, not connected-versus-standalone
comparisons. The MathMesh-connected preflight stopped before model loading;
Stability-only, RAG-only, and all-connected numerical preflights were not run.
No integrated logits, cache traces, positions, or checkpoints exist to compare.

`integration_control.py` is a separate run-level process and the existing Zeta
runner has no support-service call in its token loop. That source boundary is
useful design evidence, but it does not replace measured non-interference.
Support-service outage behavior during model execution, integrated checkpoint
recovery, and full-run numerical equivalence remain unverified. Therefore this
report does not declare non-interference.
