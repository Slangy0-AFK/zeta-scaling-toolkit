# Full-length causal validation and context continuity

## Scope and safeguards

This is an engineering and exploratory retention check, not a measured native
failure boundary, a multi-seed evaluation, or evidence of usable-context
extension. No model was trained, no weights or historical results were changed,
and no full boundary experiment was run. The ordinary adapter's 8,192
position guard remains intact.

The implementation inspected was commit
`ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`. The
[experimental adapter](./position_extension.py) explicitly opts into
unscaled native RoPE above the configured limit. It passes absolute
`position_ids` and `cache_position`, keeps one
[`StreamingKVCache`](../../zeta_scaling_toolkit/positions.py) per layer, and
stores already-rotated keys with their absolute positions. The underlying
[adapter](../smolLM2-integration-v1/adapter.py) processes every input token
individually, including prefill; it supplies native eager Llama attention
with a zero mask over the already selected sink-plus-recent keys. Each layer
checks the next absolute position before appending. The cache keeps positions
below the sink count and the latest `window` positions; other K/V entries are
evicted, not archived.

The checkpoint uses 32 layers, 15 query heads, five KV heads, head size 64,
`rope_theta=100000`, and no RoPE scaling. Default RoPE calculates inverse
frequencies and applies rotations from supplied positions rather than indexing
a fixed 8,192-entry table; this is mathematical execution, not evidence that
the checkpoint generalizes to those positions. Dynamic or rescaled RoPE is
not enabled. Prior short-position RoPE checks and original-range parity are
documented in [the feasibility report](./POSITION_EXTENSION_FEASIBILITY.md).

## Protocol

[The validation runner](./validate_full_length.py) uses the pinned
SmolLM2-360M-Instruct snapshot, native tokenizer and chat template, FP32 CPU,
Transformers 4.52.2 eager attention, and greedy decoding. A single
deterministically generated stream contains four distinct six-digit facts;
seed `20261117` is fixed in the script. It does not tune facts or use
additional seeds. The stream starts at absolute position zero and the cache
uses 64 sinks and a 64-token recent window. It targets sequential cumulative
lengths 8,193, 8,256, 8,512, 9,000 and 10,000, with a 3,000-second
wall-time budget and 1,024 MiB minimum host available memory. Those limits
are checked between chunks, not continuously. Actual results, status, source
and snapshot hashes, timings and individual answers are in the JSON record.

The runner independently replays the **entire** stream from position zero
with native multi-token forwards, its own `DynamicCache`, an independently
constructed causal sink/window mask, and separately selected retained K/V.
It compares every logit against the adapter's token-at-a-time path using the
exploratory `atol=2e-4`, `rtol=1e-4` established after observing native
FP32 execution-order variation, not preregistered. Every token's position is
explicit on both paths. This reference matches the bounded visibility policy;
it does not compare the adapter against ordinary full-history inference.

At each target, a second native suffix check starts from a clone of the
adapter's retained prefix. It validates attention probabilities in all
layers for finite values, normalized rows and zero masked probabilities. The
final token is changed in a second evaluation, and the preceding three
logits must remain exactly unchanged. The adapter checks every token's layer
positions; the harness independently checks all layers' retained positions,
tensor shapes and finite K/V after each chunk. All adapter and independent
reference logits are checked finite.

The captured-prefix suffix oracle is conditional on the adapter prefix, while
the independent bounded native replay is not. Attention-probability checks
are sampled at the listed suffixes, not every score of the whole run.
Token-by-token execution prevents later input tokens from being supplied at
earlier adapter steps. Native replay above 8,192 uses explicitly supplied
experimental position IDs; it is not a comparison with ordinary native
full-history inference or evidence of normal-range quality.

Four isolated branches from the same final cache ask for each code using the
native chat template. The question contains no answer. The runner records the
fact's token span, whether the entire span remains in cache after the question
prefill, the exact output and whether it contains the correct six-digit code.
The short native controls present each fact immediately before its question;
they check whether the model can follow the prompt, not full-history parity.
Only one deterministic stream and one question per fact are used, so outcomes
are illustrative, not statistically reliable. A correct answer for an evicted
span could reflect indirect information in retained hidden states, guessing,
or other confounds; the test does not establish a recovery mechanism.

## Reproduction

Use the already downloaded pinned snapshot and the compatible isolated
environment described in [the adapter guide](../smolLM2-integration-v1/REPRODUCIBILITY.md).
Check current available memory and CPU allocation first. The runner requires
the known weight SHA-256, uses only local files, refuses to overwrite an
existing result, and records an incomplete result if a resource budget is
reached. A failure is reported explicitly and terminates with an error.

```bash
PY="/path/to/environment/bin/python"
SNAP="/path/to/pinned/a10cc1512eabd3dde888204e902eca88bddb4951/snapshot"
OUT="$(mktemp -d)/full-length-results.json"
PYTHONPATH="$PWD" "$PY" research/smollm2-integration-v1/validate_full_length.py \
  --snapshot "$SNAP" --output "$OUT" --max-seconds 3000 --min-available-mib 1024
PYTHONPATH="$PWD" "$PY" -m pytest -q tests/test_smollm2_positions.py \
  tests/test_smollm2_adapter.py tests/test_properties.py \
  -k 'smollm2_positions or smollm2_adapter or streaming_kv_cache'
ruff check research/smollm2-integration-v1/validate_full_length.py tests/test_smollm2_positions.py
ruff format --check research/smollm2-integration-v1/validate_full_length.py tests/test_smollm2_positions.py
```

For a separate 256-token end-to-end harness check, add `--smoke` and use a
different output path. Never overwrite the recorded scientific artifacts.
