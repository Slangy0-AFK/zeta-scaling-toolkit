# Qualification Recheck

## Decision: blocked before launch

Run ID: `zeta100k-fresh-20261117-5bc2985bbe99`. This is a new experiment;
the prior experiment was not restarted. No 100K processing was started.
This addendum records subsequent checks without replacing the earlier gate
snapshot or historical evidence.

## Memory diagnosis

The runner reads `MemAvailable` from `/proc/meminfo` and requires at least
`--min-available-mib + 2560` MiB before model load. With the unchanged
1,024 MiB reserve, the minimum is 3,584 MiB. The earlier 2,794 MiB figure
was estimated host available memory, not Zeta usage or peak RSS. The latest
snapshot at 2026-11-17 21:06:36 UTC was 2,469.1 MiB available, a shortfall
of 1,114.9 MiB; host total was 7,942.3 MiB and swap was zero. This is a
snapshot, not a reservation or guarantee. An actual `--preflight-only`
recheck with the original threshold failed before model load with
`RuntimeError: insufficient available memory to load the model safely`.

At an earlier process snapshot, the two largest VS Code extension hosts
used approximately 1,023.2 and 685.5 MiB PSS. Two TypeScript servers had
approximately 331,512 and 143,052 KiB RSS, including shared pages. These
are editor-owned processes and were not terminated. The dev container's
cgroup had `memory.max=max`; a separately observed cgroup current charge
was 3,873,734,656 bytes and its lifetime peak 5,931,794,432 bytes. Neither
number is a fresh Zeta process peak. Two logical CPUs were available.

- Zeta/model/tokenizer: no process remained running at the idle snapshot.
  The earlier standalone preflight passed but did not record its peak RSS.
  Model weights are 723,674,912 bytes on disk; tokenizer JSON is 2,104,556
  bytes. These file sizes are not model-loading RSS measurements.
- MathMesh: its API and PostgreSQL were stopped after testing; API RSS was
  not independently measured. PostgreSQL was observed at 20.44 MiB in one
  live Docker snapshot, not an experiment-wide peak.
- Stability Toolkit: a short-lived advisory call only; no dedicated
  long-running process or independently measured peak RSS.
- RAG Governor: the corrected container was observed at 55.61 MiB, capped
  at 256 MiB and 0.25 CPU. This is not a measurement under Zeta load.
- Telemetry: the synthetic local-only 56-event probe took 0.2964 seconds
  (188.9 events/s), with 25.9 MiB reported child peak RSS and queue
  high-water 56/128. In an earlier live MathMesh test, an earlier sidecar
  revision delivered 56 synthetic events in 0.5844 seconds, using 0.1973
  CPU seconds and 27.1 MiB peak RSS. Neither is Zeta token throughput or
  a matched model-run overhead measurement; the final sidecar revision has
  not been benchmarked with a running model.

The support containers were stopped after qualification. No mathematical
setting, numerical safeguard, or memory threshold was changed, and no
editor-owned process was forcibly terminated. The largest observed potential
reclaim is in editor extension hosts; its ownership and effect on this active
session make unattended termination inappropriate. Use a larger-memory host or
close genuinely unused editor sessions under operator control, then recheck
memory while all required services are configured.

## Telemetry and support recovery

`telemetry_sidecar.py` reads progress JSONL outside the model process. Its
queue has a hard maximum of 128 events; full-queue records are deferred and
read again. Network work runs in a separate worker thread with bounded HTTP
timeouts. Progress is opened read-only; the sidecar writes only its own
status and MathMesh metadata. Focused tests cover partial records, bounded
queue pressure, malformed long records, transient failures, ordered replay,
and source-file immutability. The final synthetic suite passed 11 tests.
The run-level coordinator records failures atomically in a separate summary,
and two focused coordinator tests passed; the MathMesh completion metadata
lookup was corrected. Ruff lint and formatting checks passed for the four
touched Python files.

In a controlled MathMesh PostgreSQL outage, an earlier sidecar revision
recorded eight failed requests and later delivered a pending synthetic event
to one existing PipelineRun row after recovery. In a separate RAG outage,
admission failed at health, was recorded, and the recovered request created
one audit row. A missing Stability Toolkit path also produced a recorded
pre-admission failure. The corrected RAG container uses `exec uvicorn`;
it stopped with exit code 0, restarted, and reported `chain_valid=true`.
These are component/service-boundary tests, not service outages while Zeta
was running. The final sidecar revision has not been rechecked against live
services or with concurrent inference.

MathMesh admission and RAG credit/govern operations are not proven idempotent
after a timeout with an ambiguous server-side outcome. The tests above failed
before the relevant admission mutations or used an update to one existing
record. Do not retry an uncertain admission automatically: reconcile service
records first. Exactly-once integration recovery remains unqualified.

The new MathMesh qualification export is
`support/mathmesh/mathmesh-qualification-20261117.dump`, SHA-256
`fdab58e6d5b8fef5fb898334d4ff57beca12dcb10fc847b764d36eadf1a08e38`;
`pg_restore --list` successfully read its table of contents. The RAG SQLite
database after the new tests has SHA-256
`815557537d3709c83b7c729dc4d555f6643440d92ec5e66a977bb8c6774152ff`.
The prior RAG hash in `PROVENANCE.md` describes the earlier snapshot, before
these additional qualification events.

## Remaining mandatory gates

1. Provide enough available memory to meet the unchanged 3,584 MiB runner
   gate, including support-process headroom, and re-run the actual preflight.
2. Compare standalone, each individual integration, and all-integrated Zeta
   on identical inputs/configuration, including finite logits, numerical
   tolerance, causal behavior, position progression, and cache/checkpoint
   integrity. No integrated model output is available in this recheck.
3. Exercise each service outage during actual Zeta execution and verify no
   blocked forward operation, corrupt state, duplicate or missing records,
   and safe recovery from ambiguous request outcomes.
4. Measure final-sidecar CPU, memory, throughput, and boundedness under a
   concurrent model workload; validate failure recovery and one observer
   writer per run. Synthetic measurements alone do not qualify this gate.
5. Recheck service health, input/model/dependency hashes, CPU and peak memory,
   and checkpoint recovery before authorizing a fresh 100K launch.

No retrieval or factual-recall claim follows from the standalone preflight or
the component tests.