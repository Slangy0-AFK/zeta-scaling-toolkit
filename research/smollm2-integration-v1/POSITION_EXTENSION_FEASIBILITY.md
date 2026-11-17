# Beyond-8,192 position-extension feasibility

**Decision: NO-GO for the later context-boundary experiment.** Native unscaled
RoPE and an opt-in adapter executed a real contiguous 8,193-token input and
short additional high-position probes. This establishes finite execution and
the recorded cache-position invariants for that engineering case, not a native
failure boundary, useful long-context retrieval, or Zeta's ability to recover
evicted information. The experimental adapter is a validation prototype, not
an approved change to the default integration.

## Root cause and architecture

The actual SmolLM2-360M-Instruct checkpoint configuration has
`max_position_embeddings=8192`, `rope_theta=100000`, and `rope_scaling=null`;
its tokenizer also advertises 8,192 tokens. The existing
[default adapter](../smolLM2-integration-v1/adapter.py) copies that configured
maximum and rejects `start + length > max_positions`. This is an explicit
adapter safety policy, not evidence that native inference fails at 8,192.
The model's configured length is not an empirically measured quality boundary.

In the installed Transformers 4.52.2 Llama implementation, default RoPE
computes inverse frequencies once for the 64-dimensional head, then computes
cosine and sine from the supplied position IDs on each call. It uses
split-half rotation and has no learned absolute-position lookup table or
8,192-entry RoPE table in that path. Its `max_seq_len_cached` is a scalar
for this default configuration; the inverse-frequency buffer has 32 elements.
Angles are calculated in FP32, not exact arithmetic, and numerical error
grows with position. Real hardware, precision, backend, resource and quality
limits remain empirical questions.

Native inference normally constructs `cache_position` from the cache's logical
length, then defaults `position_ids` to those values. The Zeta adapter supplies
both explicitly and retains already-rotated native keys, preserving absolute
positions even after eviction. The native eager path builds causal masks for
the requested dimensions, or accepts an explicit 4D mask. The prototype
processes one token at a time and supplies a mask for the retained causal
keys; it does not allocate an 8,192-position mask buffer. Larger real inputs
still consume time and memory, and some other backends or cache types may have
different limits.

## Approaches considered

| Approach | Pretrained weights | Behavior and original-range parity | Configuration and limitations |
|---|---|---|---|
| Native unscaled RoPE with larger IDs | Unchanged | Same angles and native attention for the same positions; original-range parity can be tested | No model-config change. Out-of-training-range quality and FP32 precision remain unverified. Chosen for the isolated probe. |
| Dynamic NTK scaling | Unchanged | Native angles can remain unchanged initially, but frequencies change as the configured range is crossed | Requires `rope_scaling` configuration. Previously cached already-rotated keys may use older frequencies after an update; correct cache handling and parity need separate validation. Not selected. |
| Linear position interpolation | Unchanged | Scales all position angles; generally changes attention even inside the original range | Requires scaling configuration and a frozen factor. It does not preserve exact native original-range behavior; untested for this checkpoint. Not selected. |
| Other methods such as YaRN/LongRoPE | Unchanged in principle | Alter position-dependent frequencies and sometimes attention scaling | Need model-specific factors, careful cache handling, independent evaluation and possibly additional training. No compatibility or quality is assumed. Not selected. |

## Isolated implementation

[position_extension.py](./position_extension.py) subclasses the existing
adapter without modifying it. A caller must explicitly instantiate
`ExperimentalPositionAdapter` and specify an experimental maximum greater
than the model's configured value; the prototype imposes its own 65,537
exclusive ceiling. It rejects scaled RoPE, retains native attention, leaves
model configuration and weights untouched, and supplies a `manifest()` with
the position mode, both limits, window, sinks, initial position and RoPE
settings. The original adapter's 8,192 guard and its existing result files
remain unchanged.

For bounded probes only, `initial_position` can start an **empty** cache at a
larger absolute position. This deliberately does not pretend that preceding
tokens were processed. A nonzero offset rejects sink tokens because no
original sink history is present. The validation record explicitly labels
these probes `synthetic_empty_offset_history`; they are not long
conversations. Normal experimental operation starts at zero.

## Results

The full machine-readable record, including per-probe manifests and hashes, is
[POSITION_EXTENSION_RESULTS.json](./POSITION_EXTENSION_RESULTS.json). Tests
used the actual pinned pretrained model and native tokenizer, two CPU
threads, FP32, eager native attention, and no training. The original weights'
SHA-256 was
`e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.

### Original range and independent position reference

- On a five-token input starting at position zero, the unchanged and
  experimental adapters produced bit-identical logits. Compared with native
  multi-token prefill, maximum absolute logit difference was 0.0000852
  (rounded up), within `atol=0.0002`, `rtol=0.0001`. The tolerance is the
  earlier exploratory FP32 tolerance, derived after investigating native
  execution-order variation; it was not preregistered for a research study.
- Native split-half RoPE transformations at positions 0, 8,191, 8,192,
  16,384, 32,768 and 65,536 were compared with an independently calculated
  FP64 rotation. All were finite and within absolute tolerance 0.003. Maximum
  error was approximately 0.0001933 at 8,192 and 0.0012453 at 65,536.
  This tolerance is an engineering check, not a claim that the resulting
  position precision is adequate for model quality.

### Masks, cache continuity and high positions

Four five-token, **empty-history offset** probes began at positions 8,190,
16,382, 32,766 and 65,532. Each used a three-token recent window and no
sinks, split across two continuation calls. The final probe processed
position 65,536; each layer retained the expected last three absolute
positions and advanced its next-position counter correctly. All logits and
cache values were finite. Each probe matched the native model using an
independently constructed dense causal window mask within the exploratory
logit tolerance; maximum absolute difference was 0.0000852 (rounded up).
Changing the final token did not change earlier logits. These checks cover
short prefill/continuation and causal visibility, not attention over a real
thousands-token history.

### Actual contiguous input and CPU

With actual input starting at position zero, window 64 and four sinks:

| Processed tokens | Cumulative wall time | Cache tensor bytes | Peak process RSS |
|---:|---:|---:|---:|
| 16 | 2.84 s | 1,314,816 | 2,818.2 MiB |
| 32 | 5.37 s | 2,629,632 | 2,818.2 MiB |
| 64 | 10.28 s | 5,259,264 | 2,818.2 MiB |
| 128 | 21.49 s | 5,587,968 | 2,818.2 MiB |

All logits were finite. Peak RSS includes model and runtime; cache figures
include position tensors but exclude allocator overhead and temporary
allocations. The entire validation process used 54.60 CPU-seconds over
32.84 wall seconds. These are single runs on a shared two-core machine, not
throughput benchmarks. At the observed early 128-token rate, a linear
projection to 8,193 tokens would be roughly 22.9 minutes; this is only an
estimate and may be inaccurate as context and system contention change.
The bounded test stopped at 128 real contiguous tokens; no actual
8,193-token input was run during that short measurement.

A separate, resource-bounded continuation then processed **8,193 actual
contiguous tokens** from position zero with the same window and sinks. The
input was a repeated tokenizer-encoded sentence for engineering validation,
not a multi-turn retrieval dataset. All logits were finite; after every
128-token chunk, all 32 layers were checked for the expected retained
positions and next absolute position. The final cache contained positions
`0..3` and `8129..8192`, with 5,587,968 tensor bytes. The run took 1,290.82
wall seconds (21 minutes 30.82 seconds) and 2,220.78 CPU seconds; process
peak RSS was 2,812.8 MiB and minimum available memory sampled between chunks
was 2,391.9 MiB. The budget was 2,100 seconds and at least 1,024 MiB
available memory. It completed without hitting either limit. The record,
including intermediate measurements, input fingerprint and settings, is
[CONTIGUOUS_POSITION_RESULTS.json](./CONTIGUOUS_POSITION_RESULTS.json).
It is a finite-execution/cache-invariant check, not an independent
full-sequence logit oracle at 8,193 positions.

## Eviction, limitations and next steps

The existing Zeta `StreamingKVCache` retains initial sinks and a recent
window. Other K/V entries are discarded and cannot be directly retrieved.
Later retained states could contain indirect traces of earlier information,
but their persistence and correctness are untested. No compression, semantic
memory or arbitrary recovery is implemented here.

The contiguous input demonstrates processing beyond 8,192 for one repeated
token stream, but not a meaningful conversational history or long-range
quality. It is not a full native-versus-Zeta comparison or proof of
independence from attention and cache implementation details. Long-range
causality, numerical error accumulation, broader memory behavior, tokenizer
handling, generation and actual EOS behavior remain to be tested. The default cache
and guard remain in force unless the explicit experimental class is used.
The earlier result is not a measured native failure boundary.

Before any boundary experiment, validate additional contiguous inputs and
independent long-range causal references; preserve original parity; implement
and validate the matched multi-turn scoring harness; freeze tolerances,
failure criteria and provenance; and obtain approval for the research run.

## Reproduction and checks

Use the existing pinned snapshot and the isolated environment from
[the previous reproducibility guide](../smolLM2-integration-v1/REPRODUCIBILITY.md).
The following writes a new file and refuses to overwrite an existing one:

```bash
PY="$HOME/.cache/zeta-smollm-venv/bin/python"
SNAP="/path/to/pinned/a10cc1512eabd3dde888204e902eca88bddb4951/snapshot"
OUT="$(mktemp -d)/position-results.json"
PYTHONPATH="$PWD" "$PY" research/smollm2-integration-v1/validate_positions.py --snapshot "$SNAP" --output "$OUT"
LONG_OUT="$(mktemp -d)/contiguous-results.json"
PYTHONPATH="$PWD" "$PY" research/smollm2-integration-v1/validate_contiguous.py --snapshot "$SNAP" --output "$LONG_OUT" --length 8193 --max-seconds 2100 --min-available-mib 1024
PYTHONPATH="$PWD" "$PY" -m pytest -q tests/test_smollm2_positions.py tests/test_smollm2_adapter.py tests/test_properties.py -k 'smollm2_positions or smollm2_adapter or streaming_kv_cache'
ruff check research/smollm2-integration-v1/position_extension.py research/smollm2-integration-v1/validate_positions.py research/smollm2-integration-v1/validate_contiguous.py tests/test_smollm2_positions.py
ruff format --check research/smollm2-integration-v1/position_extension.py research/smollm2-integration-v1/validate_positions.py research/smollm2-integration-v1/validate_contiguous.py tests/test_smollm2_positions.py
```

The focused test command passed: 20 passed, 326 deselected. No full boundary
experiment, training, weight modification or additional model download was
performed.
