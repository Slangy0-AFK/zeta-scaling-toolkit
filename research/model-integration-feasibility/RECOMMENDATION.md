# Recommendation and next steps

**Recommend:** [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/a10cc1512eabd3dde888204e902eca88bddb4951),
revision `a10cc1512eabd3dde888204e902eca88bddb4951`, with option A: an opt-in
per-layer adapter using the existing `StreamingKVCache` and the model's native
attention. It is a public Apache-2.0 instruction-tuned decoder, has relatively
small nominal weights and a published 8,192-token position configuration.
None of these establish a usable boundary or successful recall. Its small
capacity and GQA/cache integration are material risks.

Before a valid boundary experiment:

1. Obtain separate approval to acquire the pinned weights, tokenizer and
   dependencies; verify their hashes and licenses. No weights were acquired
   for this study. Pin a compatible runtime and hardware/precision.
2. Implement the explicit per-layer cache and mask adapter described in
   [ZETA_INTEGRATION_AUDIT.md](./ZETA_INTEGRATION_AUDIT.md), with native RoPE,
   native attention and unchanged pretrained weights. Reject unsupported modes.
3. Verify no-eviction parity with native inference; GQA mapping; absolute
   positions and masks during eviction, prefill and multi-turn continuation;
   state reset and unsupported cases; and actual weight/tokenizer identity.
   Test the Python adapter before assuming any quantized runtime can use it.
4. Build and independently validate the fixed-dialogue generator and answer
   oracle, output protection, complete provenance and per-turn measurement.
   Freeze development/report separation, sample sizes, budgets, failure
   criteria and error handling before report evaluation.
5. With separate approval, run a small integration/resource pilot and check
   task learnability. If the model or adapter cannot perform the task, report
   that limitation rather than weakening the criteria. A subsequent boundary
   experiment requires its own approval.

This is not a claim that Zeta extends context, recovers evicted facts, reduces
cost, or avoids token exhaustion. An existing cache may preserve information
indirectly in retained states, but provides no general retrieval guarantee.
Historical results and checkpoints remain untouched. Existing uncommitted
retrieval work was not modified.

Details: [candidates](./MODEL_CANDIDATES.md),
[integration audit](./ZETA_INTEGRATION_AUDIT.md),
[options](./INTEGRATION_OPTIONS.md),
[boundary design](./BOUNDARY_EXPERIMENT_DESIGN.md), and
[CPU estimates](./CPU_FEASIBILITY.md).
