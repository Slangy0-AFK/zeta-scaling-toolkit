# Failure analysis and decision

## Initial parity failure

The first FP32 multi-token comparison used `atol=1e-4`, `rtol=1e-4`.
For batch two, two logits failed; maximum absolute difference was
0.00014400482177734375. Investigation compared the pretrained model's own
token-by-token native cache path with its native multi-token prefill: it
showed exactly the same maximum, mean difference and two failing logits.
The adapter versus native token-by-token logits were bit-identical in both
tested batch sizes.

The final exploratory FP32 tolerance is `atol=2e-4`, `rtol=1e-4`; all
recorded comparisons pass. This tolerance was selected **after** diagnosing
the native execution-order difference, not preregistered. The control and
initial failure are reported rather than hidden. It should be independently
reviewed and frozen before any scientific study; exact equality on this small
input and CPU does not guarantee equality for other shapes or hardware.

## Current decisions

| Question | Decision | Scope |
|---|---|---|
| A. Native parity | PASS | Pinned pretrained model, FP32 CPU, eager backend and listed short cases only |
| B. Zeta integration | PASS | Per-layer adapter and native attention in the tested configuration |
| C. Bounded policy | PASS | Short dense-mask and cache-shape/position checks; not proof of long-context recall |
| D. CPU execution | PASS for short tests | Approximately 2.75 GiB process peak; long sequences and a full experiment are unmeasured |
| E. Boundary experiment | **NO-GO** | No validated boundary harness or longer-context resource/quality checks; adapter currently rejects positions beyond 8,192 |

The cache policy discards non-sink entries outside its recent window. It does
not implement arbitrary fact retrieval. Native absolute RoPE positions are
preserved, but no conclusion is available about model quality or numerical
behavior beyond its configured position range. Extending the guard or changing
RoPE requires a separately reviewed implementation and validation, not a
silent modification. The current prototype also lacks padding, variable
sequence lengths within a batch, beam search, efficient prefill, cache
serialization and independent per-row cache lifecycle.

The per-row EOS test uses a deliberately selected surrogate stopping token;
the real EOS token was exercised as input, but its natural emission was not
observed. Longer runs, CUDA, alternative precisions and backends have not been
validated. No context-extension, recall, latency or memory-savings claim is
made.
