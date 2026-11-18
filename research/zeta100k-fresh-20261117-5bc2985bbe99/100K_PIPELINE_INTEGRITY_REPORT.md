# 100K Pipeline Integrity Report

## Final outcome: processing pass

Run `zeta100k-fresh-20261117-5bc2985bbe99` completed 100,000 contiguous tokens
from position 0 through 99,999. The runner exited with code 0. This supports a
pipeline-integrity result for the specified model and configuration only; it
does not establish general retrieval quality or an end-to-end speedup.

| Requirement | Final result |
|---|---|
| Contiguous token processing | 100,000/100,000; all 56 progress records start at position 0 and have matching completed-token/last-position fields. |
| Checkpoint integrity | All 56 checkpoint hashes, provenance metadata, retained positions, tensor shapes, and finite cache values verified. |
| Numerical stability | The runner checks stream logits and cache state during processing; it completed without a numerical exception. |
| KV cache | 64 sink + 64 recent positions; final cache is 10,518,528 bytes. |
| Checkpoint restoration | The separate 8-token save/restore smoke resumed at position 4 with bit-identical logits. The final checkpoint was not reloaded after the successful run. |
| Retrieval | 0/8 exact long-range synthetic-fact answers; all 8 short-prompt controls passed. |
| Supporting observers | MathMesh delivered 56 records through 100,000 with no failures or drops; Stability emitted 56 advisory-only records. |
| Full-run service outage isolation | Not tested; support services remained available during the run. |

Final checkpoint path `checkpoints/position-100000.safetensors` (large
checkpoint files remain local and are ignored by Git), SHA-256:
`43e7eee27e0287a01f631025f3d11b8311632a58de6c1faaf1943d2097e7a076`.
Its cache contains positions 0–63 and 99,936–99,999; each of the 64 key/value
tensors has shape `[1, 5, 128, 64]`.

The measured stream throughput was 10.7199 tokens/s; stream prefill took
9,328.4 seconds and total wall time was 9,636.7 seconds. Peak runner RSS was
3,400.3 MiB. There is no matched 100,000-token native/full-history baseline,
so no end-to-end speedup is claimed. Detailed timing and retrieval outcomes
are in [FINAL_REPORT.md](FINAL_REPORT.md); pinned inputs and support commits
are in [PROVENANCE.md](PROVENANCE.md).

The launch-time snapshot in
[`launch-gate.json`](launch-gate.json) and the earlier sections of
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md) preserve
pre-run status and should not be read as final run status. The final checkpoint
hashes, observer state, and exit status are recorded in
[`qualification/monitoring-20261117.jsonl`](qualification/monitoring-20261117.jsonl).
