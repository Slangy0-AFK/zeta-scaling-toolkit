# Final report

**Status: 100K PROCESSING PASS.**

Processed 100,000 of 100,000 contiguous tokenizer-produced tokens.
The 100,000-token processing result is separate from useful-context retention.

## Fact retrieval

| Fact start position | Distance from stream end | Exact match | Entire fact span retained after query |
|---:|---:|:---:|:---:|
| 128 | 99826 | no | no |
| 2047 | 97905 | no | no |
| 8002 | 91952 | no | no |
| 15998 | 83958 | no | no |
| 24999 | 74957 | no | no |
| 49999 | 49951 | no | no |
| 75001 | 24949 | no | no |
| 90000 | 9954 | no | no |

The values were randomly generated from a recorded deterministic seed. Questions did not contain expected values. A correct answer after eviction is not sufficient to distinguish indirect persistence from guessing.

## Native baseline

The native model's configured context length is 8,192 tokens. The matched native prefix run used 8064 tokens, with query prompts kept within that configured limit where possible. It is not a matched 100,000-token baseline.
Completed native fact queries: 3 of 8.
The experimental path processes a single 100,000-token stream; the native baseline is only a same-stream prefix.

## Resources

- Total wall time: 9636.662430999 seconds
- Stream prefill wall time: 9328.404234915 seconds
- Stream throughput: 10.719947107964408 tokens/second
- Peak resident memory: 3400.32421875 MiB
- Peak GPU memory: None MiB
- Final bounded KV cache: 10518528 bytes

## Final integrity and provenance audit

All 56 recorded checkpoints were verified against their recorded SHA-256,
model revision, model hash, stream hash, absolute retained positions, cache
tensor shapes, and finite cache values. The final checkpoint is
`checkpoints/position-100000.safetensors`, SHA-256
`43e7eee27e0287a01f631025f3d11b8311632a58de6c1faaf1943d2097e7a076`.
Progress records cover position 0 through 99,999 without a gap or duplicate.
The runner exited with code 0.

MathMesh delivered all 56 checkpoint telemetry records through token 100,000
with no failures or drops. Stability recorded 56 advisory-only recommendations;
none were applied to Zeta. The telemetry failure-injection smoke preserved
finite logits and cache state, but no physical support-service outage was
performed during the full run. A separate 8-token checkpoint restore smoke
passed; the final checkpoint was not reloaded after normal completion.

## Scientific conclusion

No compute or latency savings are claimed: there is no matched native 100K full-history baseline. Processing tokens, retaining useful information, reducing reprocessed tokens, and reducing compute are distinct claims. No numeric context-retention pass threshold was supplied, so this report does not assign CONTEXT RETENTION PASS.

All raw per-stage and per-fact records are retained in `progress.jsonl`, `retention-results.jsonl`, `native-baseline.jsonl`, and `resource-results.json`.
