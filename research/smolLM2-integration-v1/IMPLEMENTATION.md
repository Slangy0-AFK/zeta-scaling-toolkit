# Implementation and validation

The implementation is [adapter.py](./adapter.py); the offline validation
runner is [validate.py](./validate.py). Existing benchmark and toolkit
behavior is unchanged. The native model and tokenizer are pinned to
`HuggingFaceTB/SmolLM2-360M-Instruct` revision
`a10cc1512eabd3dde888204e902eca88bddb4951`. Weight SHA-256:
`e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.
The validation runner checks this digest, records model/tokenizer/code hashes,
refuses to overwrite results, and requires a matching successful parity record
before its bounded phase.

## Measured bounded CPU checks

On two available AMD EPYC 7763 logical CPUs, Python 3.14.2, PyTorch
2.15.0+cu132 (CPU execution), Transformers 4.52.2, FP32 eager attention:

- [Native parity results](./NATIVE_PARITY_RESULTS.json): pass for one-token,
  batch sizes one and two, five-token prefill, multiple continuation calls,
  greedy per-row stopping, and the model's EOS token as input. Maximum
  prefill difference was 0.0001441 (rounded up); tolerance was
  `atol=0.0002`, `rtol=0.0001`. Identical token-by-token native and adapter
  paths produced exactly identical logits in both tested batches.
- [Bounded cache results](./CACHE_VALIDATION_RESULTS.json): pass against a
  native dense causal sink/window mask for nine tokens, batch two, window 3,
  sinks 2. All 32 layers retained absolute positions `[0, 1, 6, 7, 8]`,
  each KV tensor had shape `[2, 5, 5, 64]`, and cache and logits were finite.
  Maximum logit difference was 0.0001441 (rounded up).
- The parity process peaked at 2,812.5 MiB RSS, including the model and
  runtime. Its overall CPU time / wall time was 1.54 cores (about 77% of two
  available cores); this is not sampled per-call CPU utilization.
- One five-token prefill, batch one: native 0.323 s, adapter 0.901 s.
  One-token continuation, batch two: adapter 0.229 s and 0.247 s in two
  sequential calls. These are individual small-run timings, not throughput
  estimates or evidence of improved efficiency.
- The full five-token batch-one cache occupied 410,880 tensor bytes; the
  bounded batch-two cache occupied 820,480 tensor bytes. These include cached
  K/V and position tensors, not allocator or temporary memory.

The EOS stopping check sets token ID 28 as a deliberate surrogate EOS so one
batch row ends before the other; it matched native greedy generation. The
checkpoint's actual EOS ID 2 was separately tested as an input token, not
observed as a naturally generated stop. See
[FAILURE_ANALYSIS.md](./FAILURE_ANALYSIS.md) for the initial tolerance failure
and interpretation.

## Automated coverage

`tests/test_smollm2_adapter.py` uses a small random Llama configuration for
offline contract tests, not as a substitute for the recorded pretrained
validation. It checks full-cache parity, batch and continuation handling,
per-row EOS stopping, GQA cache shape, multiple sink/window policies against
an independent dense native mask, position continuity, finite tensors,
rejected padding/positions/batch changes/overflow, and invalidation on a
partial layer error. Together with existing Zeta cache tests: **16 passed,
326 deselected**. Commands and environment setup are in
[REPRODUCIBILITY.md](./REPRODUCIBILITY.md).
Repository-wide `ruff check .` and `ruff format --check .` both passed.
