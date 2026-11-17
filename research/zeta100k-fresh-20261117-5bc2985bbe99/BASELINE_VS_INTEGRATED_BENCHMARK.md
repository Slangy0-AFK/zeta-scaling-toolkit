# Baseline vs Integrated Benchmark

For subsequent sidecar measurements and remaining gates, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Current status: full run active; no matched full-run comparison yet

The fresh 100K process is active. Its latest checkpoint interval measured
10.548 tokens/s, with 4,096/100,000 contiguous tokens complete. This is an
observed run rate, not a matched baseline-versus-integrated comparison.
The previous matched 4,096-token attention benchmark remains unchanged:
telemetry off 7.33×, telemetry on 7.67×, two-pair medians; the difference is
inconclusive. The historical 8,192-token 9× crossover remains unverified.

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

- Host: 4 CPUs, 15,993.6 MiB total memory, CPU-only execution; latest host
  `MemAvailable` 9,844.6 MiB.
- Stability recommendations at checkpoints 2,048 and 4,096: 4 CPUs, 9,969
  MiB available at sample time, concurrency 4, buffer 2,823, task size 186.
  Recommendations were recorded and not applied to Zeta.
- Zeta load gate: at least 3,584 MiB available; it passed before model load.
- Runner-reported peak RSS through checkpoint 4,096: 2,480.3 MiB.
- MathMesh and RAG health/admission/audit calls succeeded and their records were
  preserved. The RAG audit chain verified after restart; MathMesh DB export
  restored successfully.
- The RAG container needed forced termination at shutdown (exit 137,
  `OOMKilled=false`) because its shell wrapper did not forward SIGTERM. A
  restart succeeded and its persisted audit chain verified.
- A support-service outage during active token processing and recovery under
  load were not measured.

The standalone and post-resize actual-model preflight records have identical
correctness outputs. Full-run CPU and peak RSS will continue to be recorded;
no retrieval or factual-recall claim is made by this pipeline-integrity report.
