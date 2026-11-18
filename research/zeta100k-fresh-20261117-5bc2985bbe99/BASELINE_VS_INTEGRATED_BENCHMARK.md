# Baseline vs Integrated Benchmark

For subsequent sidecar measurements and remaining gates, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Status: PARTIAL; no full-run throughput comparison

No standalone or integrated 100K processing run was launched. Consequently,
matched tokens/second, per-token CPU usage, peak model-run memory, checkpoint
write overhead, and retrieval behavior are unavailable. Do not compare these
preflight timings to the previous experiment's interim throughput: they are
not a matched run on this process/configuration.

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

These are one-time coordination measurements, not forward-pass or telemetry
cost measurements. There were no support-service requests in Zeta inference.
The coordinator's requests are sequential and individually bounded by a
three-second timeout. Asynchronous telemetry overhead is not measured because
an asynchronous queue has not been implemented.

## Resources and service reliability

- Host: 2 visible CPU cores, 7,942 MiB total memory, CPU-only execution.
- Stability recommendation at admission: 3,494.31 MiB available; concurrency
  2; buffer 2,823; task size 186. Recommendations were not applied.
- Zeta load gate: at least 3,584 MiB available. The latest observed value was
  2,794 MiB, so the 100K launch was blocked.
- MathMesh and RAG health/admission/audit calls succeeded and their records were
  preserved. The RAG audit chain verified after restart; MathMesh DB export
  restored successfully.
- The RAG container needed forced termination at shutdown (exit 137,
  `OOMKilled=false`) because its shell wrapper did not forward SIGTERM. A
  restart succeeded and its persisted audit chain verified.
- A support-service outage during active token processing and recovery under
  load were not measured.

The standalone actual-model preflight passed before memory became insufficient.
The matching MathMesh-connected preflight stopped before model load. No
integrated throughput, CPU, peak RSS, or matched baseline-versus-connected
execution result is claimed.
