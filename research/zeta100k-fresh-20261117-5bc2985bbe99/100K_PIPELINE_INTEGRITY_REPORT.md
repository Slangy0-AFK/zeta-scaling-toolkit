# 100K Pipeline Integrity Report

For the later launch-gate recheck, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Outcome: BLOCKED BEFORE TOKEN PROCESSING

Run ID: `zeta100k-fresh-20261117-5bc2985bbe99`

No full-run Zeta process was started. The standalone preflight generated and
verified the deterministic 100,000-token stream, but did not process that
stream as a contiguous model run.

| Requirement | Status |
|---|---|
| Contiguous token processing | Not started; 0 of 100,000 processed. |
| Position progression | Standalone unit/preflight checks passed; no contiguous-run positions recorded. |
| Cache integrity during 100K | Not tested on a full stream. |
| Numerical stability during 100K | Not tested. |
| New-run checkpoints/recovery | None created; no new run checkpoint to restore. |
| Full 100,000-token completion | Not completed. |
| Retrieval accuracy/factual recall | Not tested. |
| Position-dependent model behavior | Not tested. |

The standalone correctness preflight passed its native/adapted parity,
sequential cache, causal invariance, bounded reference, and high-position RoPE
checks. It is not evidence that the full contiguous pipeline completed or that
facts remain retrievable. The preflight stream hash is recorded in
`qualification/standalone/CORRECTNESS_RESULTS.json` and
[PROVENANCE.md](PROVENANCE.md).

The full run was blocked because the latest available memory was 2,794 MiB,
below the runner's 3,584 MiB requirement. The threshold and model configuration were not
changed. The exact error and latest measurement are recorded in `launch-gate.json`.

No degradation, retrieval score, or successful pipeline-integrity claim is
inferred from the standalone preflight or the supporting services' audit
records.
