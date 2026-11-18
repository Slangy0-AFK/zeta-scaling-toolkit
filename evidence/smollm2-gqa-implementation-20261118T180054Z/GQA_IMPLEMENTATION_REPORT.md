# Native GQA implementation report

## Result

Implemented and exported a separate `GroupedQueryAttention` API. It passed
independent numerical equivalence checks against explicit KV-head expansion
and ordinary scaled-dot-product attention. The API does not modify or fall
back to `NonCollapsingAttention`.

The GQA operation is ready for API-level Recall integration after the caller
supplies the documented tensor layout. This is not a claim of full
SmolLM2/Recall model compatibility: RoPE, compressed-cache positions,
checkpoint projection mapping, and model inference integration remain
separate work. The requested historical directory
`evidence/zeta-gqa-acceptance-20261118T175054Z` was absent from this checkout,
so its exact failed test could not be rerun. The independent tests here use
the reported architecture and failure description as their acceptance
specification.

## Architecture and API contract

Pinned target metadata in the repository identifies
`HuggingFaceTB/SmolLM2-360M-Instruct`, revision
`a10cc1512eabd3dde888204e902eca88bddb4951`, with 15 query heads, 5 KV heads,
and head dimension 64. The implementation uses the specified 3:1 mapping and
therefore returns 960 merged features per query position.

Public API:

```python
GroupedQueryAttention(
    num_query_heads: int = 15,
    num_key_value_heads: int = 5,
    head_dim: int = 64,
)

forward(
    query: torch.Tensor,  # [B, 15, Q, 64]
    key: torch.Tensor,    # [B, 5, K, 64]
    value: torch.Tensor,  # [B, 5, K, 64]
    *,
    is_causal: bool = True,
) -> torch.Tensor        # [B, Q, 960]
```

For each KV head, its contiguous group of three query heads attends to that
KV head. Logits are raw QK dot products multiplied by `1/sqrt(64)`; there is
no Q/K normalization, trainable temperature, learned parameter, projection,
or output projection. Causal query indices are aligned to the end of the key
sequence (`K-Q+i`) so a suffix can attend to a longer contiguous prefix.
`Q > K` is rejected in causal mode. The implementation groups contractions
without physically expanding K/V, but computes the full grouped attention
logits and is not a specialized fused kernel.

Q/K/V must be rank-four floating tensors with matching batch size, dtype,
device, sequence compatibility, and configured head counts/dimensions. The
accumulator is FP64 for FP64 inputs and FP32 otherwise; output dtype/device
are preserved. Invalid layouts, configuration, and causal lengths raise
clear errors.

## Numerical and gradient results

The independent reference explicitly applies `repeat_interleave` to K/V,
then computes scaled QK matrix multiplication, causal masking, softmax, and
the value product. It never calls `GroupedQueryAttention`. FP32 equivalence
used `atol=1e-5`, `rtol=1e-4`, unchanged from the requested starting values.
Across the requested acceptance shapes plus the additional dtype/gradient
shape, the maximum absolute error was
`2.384185791015625e-07`. The maximum relative error was
`1.665404488448985e-05`, computed only where `abs(reference) > 1e-3` to avoid
the undefined/unstable relative error near zero. All tested cases met their
tolerances.

FP16 used `atol=rtol=1e-3`, BF16 used `atol=rtol=1e-2`, and FP64 used
`atol=rtol=1e-12`. In the per-dtype gradient test, maximum absolute errors
were FP16 `7.62939453125e-06`, BF16 `0`, and FP64
`3.3306690738754696e-16`; all matched their stated tolerance. Outputs and
Q/K/V gradients were finite in all four dtypes on CPU.

The focused GQA test module passed: **17 passed**. It covers output dimensions,
batch/sequence shapes, unequal query/key lengths, contiguous head mapping,
causal future isolation, noncausal mode, independent-reference equivalence,
four dtypes, finite gradients, and invalid inputs.

## RoPE, cache, and model integration

- **Attention:** Numerically compatible with the specified raw scaled-dot
  product GQA operation for the tested shapes and dtypes.
- **RoPE:** Not wired into this API and not directly compatible by convention.
  Repository integration notes describe SmolLM2/Llama split-half RoPE with
  `rope_theta=100000`, while Zeta's current `rotary_embedding` uses
  interleaved even/odd pairs and defaults to base `10000`. Do not apply the
  existing Zeta RoPE to these pretrained projections without a separate
  compatible implementation and tests.
- **KV cache:** The new API's K/V shape can consume a contiguous cache shaped
  `[B, 5, K, 64]`; the existing `StreamingKVCache` can structurally store
  `[B, 5, 1, 64]` one-token appends. This pairing has not been integration
  tested. The attention API has no absolute-position argument. It cannot
  correctly infer causal visibility for a compressed sink/recent cache with
  discontiguous retained positions; that requires an absolute-position mask
  or a cache-aware adapter. The cache is detached/inference-only and has no
  per-row position or beam-reordering interface.
- **Tensor layout/checkpoint mapping:** Callers must reshape projected Q
  channels to `[B, 15, S, 64]`, K/V channels to `[B, 5, S, 64]`, and flatten
  returned heads to 960 channels before any model output projection. No
  weights were loaded and no checkpoint-key mapping was exercised.
- **Remaining integration:** Model projections/output projection, native
  Llama RoPE, absolute position and mask handling, layer/cache lifecycle,
  model loader, and end-to-end inference are not implemented or validated.

No SmolLM2 weights or tokenizer were loaded, no model inference or 360M
smoke test was run, and no Recall or other repository was modified.

## Regression and provenance

Existing attention regression tests were not changed. The focused set passed
with **488 passed, 1 skipped**. The complete CI-style test run passed with
**698 passed, 6 skipped**, and **96.62% total coverage** against the 95%
threshold. Ruff, repository formatting, strict mypy, and `git diff --check`
passed. See `REGRESSION_TEST_RESULTS.json` for commands and environment.

Source base commit: `45da1c25d93e55655940b891436bc1e268daac73`.
Branch: `copilot/experiment-infrastructure-phase-2`.
Package version: `0.5.1` (unchanged). Changes were not committed or pushed.
