# Failure analysis

## Blocking conditions

### Required actual model unavailable

The machine has no Hugging Face model cache at the checked locations and no
local `/models` directory. Python cannot import `torch`, `transformers`,
`tokenizers`, or `safetensors`. No weights or tokenizer were downloaded, in
keeping with the restriction. Consequently no native or adapter inference,
token-count validation, causal checks, or retention scoring can be performed.

### Requested source is not in the local checkout

The local HEAD is
`a4e8a35c4ae53461f1b88e1334bb50995cdd5e93`; the requested commit
`ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed` was verified through the GitHub
repository service but is not in local history. Its files were inspected
remotely; the working tree was not changed to switch revisions.

### Prototype position ceiling is below target

The opt-in `ExperimentalPositionAdapter` rejects a `max_positions` value
above 65,537. Its largest representable contiguous positions are 0–65,536.
The requested 75,000 and 100,000 stages cannot run using this implementation
without a separately reviewed extension. The default adapter's 8,192 guard
remains unchanged and must not be bypassed implicitly.

## Risks that remain unmeasured

- Native inference beyond 8,192 has no measured execution or quality failure
  boundary in this session.
- The historical 8,193-token engineering run used repeated tokenizer-encoded
  text and did not ask questions about early facts.
- The 64-window/4-sink cache evicts every non-sink key/value outside the most
  recent window. Whether any early fact remains usable indirectly is unknown.
- No 100K position-ID, RoPE precision, mask, cache-continuation, finite-logit,
  throughput, or resource check was attempted.
- No fact locations, exact answers, multi-fact reasoning results, or accuracy
  numbers exist for this session.

## Decision

The appropriate outcome is **BLOCKED**, not PARTIAL or 100K GO: this task ran
zero experiment tokens, and the required range is unsupported by the
inspected prototype as it stands. Do not infer success from the prior
bounded engineering validation.
