# Failure analysis

## Decision: BLOCKED

The experiment was not launched. No model output, token position, retrieval
answer, or baseline result was fabricated.

### Blocking conditions

1. The exact pinned SmolLM2 snapshot and tokenizer are not present locally.
   The environment cannot resolve `huggingface.co` or `hf.co`; the tokenizer
   endpoint check failed before any transfer.
2. The isolated venv has no PyTorch, Transformers, Tokenizers, or Safetensors.
   Resolving the CPU PyTorch wheel failed because the index redirected to
   `download-r2.pytorch.org`, which also could not be resolved.
3. Transformers 5.23.0 was source-inspected but not installed. Its cache uses
   per-layer `CacheLayerMixin` objects and no longer supplies the
   `cache_kwargs["cache_position"]` expected by the restored adapter. No
   compatibility port or parity test is available, so changing a version
   pin alone is unsafe.
4. No model-dependent correctness gate or staged run was completed. The
   8,193-token historical result is separate evidence from the prototype's
   original Transformers 4.52.2 environment and cannot serve as validation
   for this secure runtime.

### Actions taken

- Inspected checkout state and preserved the historical v1 experiment files.
- Fetched and selectively restored only the adapter, opt-in position
  prototype, its two validators, and the focused adapter/position tests from
  commit `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`; no branch merge or reset
  was performed.
- Created an isolated venv without modifying system Python.
- Reviewed Transformers 5.23.0 wheel source and checked advisories for the
  direct candidate packages. The Transformers wheel was used for inspection
  only and was not installed.
- Stopped the PyPI generic PyTorch dry run after it selected CUDA libraries;
  no packages were installed and no checkpoint was downloaded.

### What must not be inferred

The requested 100K processing, native position failure boundary, answer
accuracy, eviction-conditioned retention, native-versus-Zeta comparison, and
resource cost are all unknown. Successful processing and useful context
retention are distinct claims; neither was measured here.
