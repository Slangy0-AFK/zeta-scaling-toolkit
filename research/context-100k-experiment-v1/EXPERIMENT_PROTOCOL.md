# Zeta 100,000-token context continuity experiment

## Status: BLOCKED — no experiment execution

This folder records the preflight for the requested experiment. No model was
loaded, no prompt was tokenized, and no native or Zeta inference was run in
this session. Empty measurements are explicitly marked not run; they are not
passes and are not estimates of model quality.

## Preflight findings

- Requested implementation commit:
  `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`
  (“Add SmolLM2 position extension feasibility”). The local checkout is
  `a4e8a35c4ae53461f1b88e1334bb50995cdd5e93`; it does not contain that commit
  or its research files. The commit was inspected through the repository
  service without changing the checkout.
- The prototype at that commit is explicitly opt-in:
  `ExperimentalPositionAdapter(model, max_positions=...)`. The original
  adapter's 8,192-position guard is unchanged. The experimental constructor
  rejects `max_positions > 65,537`; thus its largest supported contiguous
  sequence has 65,537 positions (0 through 65,536). It cannot run the required
  75,000- or 100,000-token stages.
- The model and tokenizer are not available locally. The Hugging Face cache
  directories and `/models` are absent. No model was downloaded.
- This Python environment has none of `torch`, `transformers`, `tokenizers`,
  or `safetensors`; even importing the model stack is impossible.
- The historical engineering record in commit `ee3a7c8` reports one actual
  8,193-token contiguous Zeta run, but it used a repeated sentence and was
  explicitly not a retention study. It is not a result of this experiment and
  is not copied into the stage result files as a new pass.

## Intended protocol (not executed)

The eventual study must use the pinned `HuggingFaceTB/SmolLM2-360M-Instruct`
weights and native tokenizer, CPU FP32 inference, the existing Zeta Llama
adapter, and an explicitly enabled position extension capable of positions
through at least 99,999. No fine-tuning, weight changes, or model downloads
are permitted. Verify the pinned weight and tokenizer hashes before loading.

For each exact target length (8,193; 10,000; 16,384; 25,000; 50,000; 75,000;
100,000), complete the entire tokenizer-verified input and its continuation
before recording a pass or starting the next stage. Record progress, absolute
position IDs, retained and evicted cache positions, finite logits, causal-mask
checks, wall and CPU time, peak RSS, cache bytes, and exact token counts.
Write each completed stage to a new, non-existing output path.

Use the same realistic, unpredictable fact-bearing conversation and fixed
questions in native and Zeta runs at common supported lengths. Tokenize the
full conversation with the pinned tokenizer, then record fact positions from
the resulting token IDs. Ask the same questions at the specified checkpoints.
Score exact answers and multi-fact answers separately for recent, retained,
evicted, and early facts. Do not equate finite execution, position acceptance,
or a recent-only cache with useful context continuity.

The cache policy observed in the inspected adapter is a per-layer sink plus
recent-window cache. The existing 8,193-token engineering record used a
64-token recent window and 4 sink tokens: maximum 68 retained positions per
layer, retaining positions 0–3 and the latest 64 positions, evicting all
other key/value positions. This policy must be frozen and reported exactly
for any subsequent study; evicted information is not directly accessible
through this cache.

## Result interpretation

The native adapter's 8,192 guard is an implementation safeguard, not a
measured native-model quality or execution-failure boundary. The historical
prototype probes did not establish that boundary or test long-range recall.
This session has no native-baseline or Zeta measurements; see the result
JSONL files and `comparison-report.md`.
