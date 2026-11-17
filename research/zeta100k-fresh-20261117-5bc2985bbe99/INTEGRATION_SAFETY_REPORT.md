# Integration Safety Report

For subsequent checks and remaining gates, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Current status: RUNNING; launch qualifications passed

The fresh runner is active as PID `12915`, at checkpoint 4,096/100,000.
The host had 9,844.6 MiB `MemAvailable` at the latest check, above the unchanged
3,584 MiB load gate. Model, corpus, tokenizer, and stream hashes match the
pinned provenance.

| Qualification | Current result |
|---|---|
| Standalone and post-resize actual-model preflight | Passed; correctness result fields match. |
| MathMesh, Stability, and RAG run-level integration | Passed; the exact commits and calls are in `integration-state.json` and `PROVENANCE.md`. |
| Checkpoint save/load recovery | Passed on the first 8 stream tokens; resumed logits were bit-identical. |
| Live asynchronous MathMesh telemetry | Delivered the 2,048 and 4,096 progress summaries; no failures or drops observed. |
| Stability checkpoint advisories | Recorded at 2,048 and 4,096; both are advisory-only. |
| Support-service failure isolation | Injected telemetry sender failures did not prevent finite model logits/cache checks; coordinator failure-record tests passed. |
| Full-run completion and final recovery | Pending while the run continues. |

The outbound telemetry queue is capped at 128 records, with bounded HTTP
timeouts. Support observers run outside Zeta and do not receive model tensors,
logits, or cache state. The live runner uses the pinned CPU FP32 configuration
without resume.

## Historical pre-resize decision

The previous decision was blocked because memory was below the launch gate.
The following inventory and results preserve that earlier qualification
snapshot; they do not describe the current resource or run state.

## Repository and dependency identity

Exact commits and dependency hashes are in [PROVENANCE.md](PROVENANCE.md).
The support repositories were cloned into `/tmp/zeta-100k-support/`, checked
out detached at the recorded commits, and left outside Zeta's source tree.

- Zeta runtime: Python 3.14.2, PyTorch 2.15.0+cpu, Transformers 5.23.0,
  Tokenizers 0.23.3, Safetensors 0.8.0, Hugging Face Hub 2.4.0. `pip check`
  passed in `/tmp/zeta100k-fresh-5bc2985bbe99`.
- MathMesh: Node 24.22.2, npm 11.19.1; `npm ci` installed 642 locked packages.
  The deterministic suite reported 51 passed, 1 skipped, 0 failed.
- Stability Toolkit: Python standard library only; 8/8 unit tests passed.
- Deterministic RAG Governor: Python 3.11 container, pinned image digest and
  API dependencies recorded in `PROVENANCE.md`; API tests 5/5 passed and the
  author-written golden set passed 27/27 cases.

The RAG repository's pinned `pydantic-core` dependency could not build under
Python 3.14 because its PyO3 version supports Python only through 3.13. It was
isolated in the repository's declared Python 3.11 container boundary. The
unused Chroma/LangChain retrieval stack was not installed; the run used a
static adapter with the service's API/control dependencies only.

## Required qualification matrix

| Case | Result |
|---|---|
| Zeta standalone baseline | Actual-model preflight passed. |
| Zeta + MathMesh | Admission, metadata, and audit calls passed. The matching Zeta preflight stopped before model load: insufficient available memory. |
| Zeta + Stability Toolkit | Recommendation generated and recorded, not applied. Matching Zeta preflight not run after the resource gate failed. |
| Zeta + Deterministic RAG Governor | Governed audit request accepted; audit chain healthy. Matching Zeta preflight not run after the resource gate failed. |
| Zeta + all three | All three run-level calls succeeded in 0.327 seconds total. Integrated Zeta preflight and full run were blocked by available memory. |

A successful service call is not a successful integrated-model qualification.
Identical numerical outputs, positions, cache integrity, and checkpoint
recovery have not been compared for the integrated cases.

## Resource and isolation checks

The Stability Toolkit observed 2 CPU cores and 3,494.31 MiB available during
admission. It recommended buffer 2,823, concurrency 2, and task size 186; none
was applied. The Zeta runner requires at least 3,584 MiB available (1,024 MiB
configured reserve plus 2,560 MiB model-load headroom). Later checks recorded
3,150 MiB, 3,325 MiB after service shutdown, 2,677 MiB, and finally 2,794 MiB.

The integrated preflight's exact failure was:

```text
RuntimeError: insufficient available memory to load the model safely
```

No numerical setting or safety threshold was changed. The MathMesh database
export was restored into a disposable database with `psql`; it contains the
expected admission, audit, and metadata records. The RAG database reports
`chain_valid: true`. Both stores are preserved under `support/`.

The services were bound to localhost and stopped after qualification. Their
outage during an actual Zeta execution was not verified. The coordinator uses
bounded synchronous run-level requests; the required asynchronous telemetry
queue is not implemented. Both items remain launch gates.

The RAG container exited with code 137 during shutdown, while Docker reported
`OOMKilled=false`. Its shell wrapper did not forward SIGTERM, so Docker forced
termination after the stop grace period. The container restarted successfully
and `/health` then reported a valid persisted audit chain. This is a service
shutdown limitation and recovery observation, not an inference-time failure.
