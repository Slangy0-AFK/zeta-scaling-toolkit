# Experiment protocol

## Objective

Process one contiguous 100,000-token input with the experimental position
adapter and bounded KV cache, checking numerical correctness, cache behavior,
retrieval, and resource use. This measures processing and retrieval behavior;
it does not establish compute or latency savings versus a matched native 100K
full-history run.

## Inputs and deterministic construction

- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`, pinned to the revision in
  `PROVENANCE.md`.
- Source text: the local tiny Shakespeare corpus identified by its SHA-256 in
  `CORRECTNESS_RESULTS.json`.
- Tokenizer output: exactly 100,000 token IDs. Facts are inserted into the
  source at seeded locations; seed `20261117` makes their identifiers and
  numeric values deterministic. The question prompts do not include expected
  values.
- The final token stream hash is recorded in both correctness and run outputs.

## Adapter and checkpoints

- Native model context: 8,192 positions.
- Explicit experimental positions: up to 100,512.
- KV retention: 64 sink tokens and a 64-token recent window, as implemented by
  `compat_adapter.py`.
- Stream chunks: 64 tokens.
- Checkpoints: every 2,048 tokens and at requested stages
  `8,193`, `10,000`, `16,384`, `25,000`, `50,000`, `65,537`, `75,000`,
  `90,000`, and `100,000`.
- Checkpoint writes use temporary files and atomic replacement. Resume verifies
  model/stream provenance and cache positions before continuing.

The position-65,537 gate is passed only by the contiguous run from position
zero. An empty-history synthetic offset test is reported separately and is not
a substitute for this gate.

## Correctness gates

Preflight checks compare the native and adapted paths, sequential cache
continuation, future-token invariance, bounded attention against a dense
sink/window reference, masked attention probability, cache shapes and
positions, finite logits/cache values, and high-position RoPE against an
independent FP64 reference. Tolerances are recorded in
`CORRECTNESS_RESULTS.json`.

The current CPU-only actual-model preflight passed. The contiguous stage gate
remains pending until the full stream reaches position 65,537.

## Retrieval evaluation

At the end of the stream, the runner asks for each inserted fact using the
bounded cache and records exact-match output, the fact span's retained
positions, and a short-prompt control. An answer after direct fact eviction
does not prove long-range retention; indirect persistence or guessing remains
possible.

There is no supplied numeric retention pass threshold. Report exact retrieval
counts and per-fact results without labeling them `CONTEXT RETENTION PASS`.
Processing 100,000 tokens and retaining useful context are distinct outcomes.

## Native baseline and interpretation

The native baseline processes only an 8,064-token prefix, with query prompts
kept within the model's 8,192-token native limit where possible. It is not a
full-history 100K baseline and cannot support matched compute, latency, or
quality claims against the experimental run. Its prefix length and per-query
status are reported explicitly.

## Run budgets

The runner defaults to a 43,200-second prefill budget and at least 1,024 MiB
available memory. It records wall time, process CPU time, throughput, peak
resident memory, peak GPU memory (if applicable), and KV-cache bytes. The
current runtime is CPU-only.
