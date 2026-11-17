# 100K Pipeline Integrity Report

For the later launch-gate recheck, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Current outcome: RUNNING

Run ID: `zeta100k-fresh-20261117-5bc2985bbe99`

The full Zeta runner is active as PID `12915` from starting commit
`cc2b8dfb45dd887f6f202ca69b8c874410fc4645`. At the latest verified checkpoint,
4,096 of 100,000 contiguous tokens were processed, ending at absolute position
4,095. No token was skipped or duplicated in the recorded prefix. The latest
interval throughput was 10.548 tokens/s.

| Requirement | Status |
|---|---|
| Contiguous token processing | Running; 4,096/100,000 processed contiguously from position 0. |
| Position progression | Checkpoint records end at 2,047 and 4,095, with matching token counts. |
| Cache integrity during run | Verified at checkpoints; 64 sink plus 64 recent positions retained. |
| Numerical stability during run | Finite logits and cache verified through the current prefix. |
| New-run checkpoints/recovery | Checkpoints at 2,048 and 4,096 exist; a separate save/restore smoke produced bit-identical resumed logits. |
| Full 100,000-token completion | In progress; not yet complete. |
| Retrieval accuracy/factual recall | Not yet evaluated; pipeline integrity is not a retrieval-quality result. |
| Position-dependent model behavior | Only qualification and current-prefix evidence exists; final interpretation pending. |

The standalone correctness preflight passed its native/adapted parity,
sequential cache, causal invariance, bounded reference, and high-position RoPE
checks. It is not evidence that the full contiguous pipeline completed or that
facts remain retrievable. The preflight stream hash is recorded in
`qualification/standalone/CORRECTNESS_RESULTS.json` and
[PROVENANCE.md](PROVENANCE.md).

At the latest checkpoint the runner reported 9,946 MiB available and peak RSS
of 2,480.3 MiB; host `MemAvailable` was 9,844.6 MiB at the subsequent check.
The 3,584 MiB launch threshold and model configuration were not changed. The
2,794 MiB result remains as historical pre-resize evidence in older gate
snapshots; it is not the current resource state.

Checkpoint `position-004096.safetensors` SHA-256:
`478c79fa23d0942370f594f0c011b4c4f154f1919724e13a865ec1d5b3b1458e`.
The full-run integrity result remains provisional until 100,000 tokens and
final checkpoint verification complete. No retrieval score or model-quality
claim is inferred from pipeline progress or support-service audit records.
