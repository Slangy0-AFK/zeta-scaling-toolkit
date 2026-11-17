# Comparison report

## Final status: BLOCKED

- **Execution length in this session:** 0 actual model tokens.
- **Information-retention accuracy:** Not measured; no facts or questions were
  tokenized or sent to either model.
- **100,000-token outcome:** Not run and not passed.

### Why the experiment did not start

1. The local checkout is at `a4e8a35`; the requested implementation commit
   `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed` is not present in the checkout.
2. No local SmolLM2 checkpoint or tokenizer snapshot is available. The
   requested “do not download another model” restriction was respected.
3. PyTorch, Transformers, Tokenizers, and Safetensors are not installed.
4. The inspected opt-in prototype caps `max_positions` at 65,537, so it cannot
   process the required 75,000- or 100,000-token stages.

### Baseline interpretation

The existing default adapter guard at 8,192 is a safeguard, not an observed
native-model failure boundary. No matched native run was made in this session.
The historical 8,193-token result from commit `ee3a7c8` is a Zeta engineering
execution on a repeated sentence; it contains no retention questions and is
not evidence of long-context information use. Historical measurements have
not been altered or treated as current results.

The prior bounded parity record (not rerun here) reports max absolute logit
difference `0.00014400482177734375`, mean absolute difference
`0.00001115258874051505`, and zero values outside `atol=0.0002`,
`rtol=0.0001` for its 9-token native dense-mask comparison. This says nothing
about parity or numerical drift at 100,000 positions.

### Largest safe run

No new actual-model run is safe or reproducible in this environment because
the model, tokenizer, and inference runtime are absent. The prototype's
largest code-supported contiguous input is 65,537 tokens, but that is only a
software ceiling, not a safe run recommendation for this unavailable setup.
The historical 8,193-token run is the largest prior recorded execution found,
not a result of this task and not a retention pass.

### Interpretation of the cache

The inspected 64-window/4-sink setup retains at most 68 positions per layer:
the first four sink positions and the latest 64; all other cached K/V
positions are evicted. Therefore a future retention study must explicitly
evaluate information from evicted positions. The cache's bounded memory or
continued finite inference alone cannot satisfy the primary hypothesis.
