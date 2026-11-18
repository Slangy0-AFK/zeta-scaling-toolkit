# Baseline vs Integrated Benchmark

For final run measurements and pre-launch qualification history, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Final status: run completed; no matched full-run comparison

The fresh run completed 100,000 contiguous tokens at 10.7199 tokens/s, with
9,328.4 seconds of stream prefill and 9,636.7 seconds total wall time. This is
an observed result for one configuration, not a matched
baseline-versus-integrated comparison. The native baseline processed only an
8,064-token prefix, and only 3 of 8 native fact queries completed. There is no
matched native 100K full-history baseline.

The separate matched 4,096-token attention benchmark remains unchanged:
telemetry off 7.33× and telemetry on 7.67× for two-pair medians in one
configuration. The difference is inconclusive; these are not end-to-end
language-model speedups. The historical 8,192-token 9× crossover remains
unverified.

## Measured run-level integration overhead

The all-service admission phase took 0.3266 seconds total:

| Service operation | Measured latency |
|---|---:|
| MathMesh health | 52.0 ms |
| MathMesh login | 52.5 ms |
| MathMesh key issuance | 104.4 ms |
| MathMesh admission | 31.4 ms |
| MathMesh metadata create | 3.6 ms |
| MathMesh audit write | 9.4 ms |
| RAG health | 3.4 ms |
| RAG registration | 7.5 ms |
| RAG credit | 6.6 ms |
| RAG audit request | 40.6 ms |
| Stability recommendation calculation | 0.031 ms |
| All run-level calls | 326.6 ms |

These are one-time coordination measurements, not forward-pass costs. The live
MathMesh observer delivered checkpoint telemetry asynchronously with no
failures observed through the 2,048-token event. The separate Stability watcher
records resource recommendations per checkpoint. The telemetry off/on
throughput difference remains inconclusive; this run is not a matched overhead
benchmark.

## Resources and service reliability

- Host: 4 logical CPUs, 15,993.6 MiB total memory, CPU-only execution.
- Full-run resource results: 18,748.1 process CPU-seconds, about 194.5%
  average process CPU, 3,400.3 MiB peak runner RSS.
- Stability recommendations were recorded at all 56 checkpoints through
  100,000 and were not applied to Zeta.
- Zeta load gate: at least 3,584 MiB available; it passed before model load.
- MathMesh delivered all 56 telemetry records through token 100,000 with no
  failures or drops; the sidecar used 41.38 CPU-seconds and 27.25 MiB peak
  RSS. This is measured usage during one run, not a telemetry-off comparison.
- MathMesh and RAG health/admission/audit calls succeeded and their records were
  preserved. The RAG audit chain verified after restart; MathMesh DB export
  restored successfully.
- The RAG container needed forced termination at shutdown (exit 137,
  `OOMKilled=false`) because its shell wrapper did not forward SIGTERM. A
  restart succeeded and its persisted audit chain verified.
- A physical support-service outage during active 100K token processing and
  recovery under load were not measured.

The standalone and post-resize actual-model preflight records have identical
correctness outputs. The completed run's 100K synthetic long-range retrieval
evaluation returned 0/8 exact answers, with 8/8 short-prompt controls passing.
No general retrieval or factual-recall claim follows from pipeline completion.
The final execution measurements are in
[`resource-results.json`](resource-results.json), and retrieval results are in
[`retention-results.jsonl`](retention-results.jsonl).
