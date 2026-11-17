# Zeta 100,000-token experiment protocol

## Status: BLOCKED — no model execution

This run did not load a model, tokenize an interaction, or execute native or
Zeta inference. The exact model revision could not be fetched because the
sandbox cannot resolve `huggingface.co` or `hf.co`; the checkpoint and its
tokenizer are absent from all searched local model-cache locations. The
isolated Python environment also has no installed PyTorch runtime. The CPU
PyTorch wheel index redirected to an unresolvable host. No alternate model,
revision, tokenizer, or device was substituted.

The prior validated prototype was selectively restored from commit
`ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`. It has not been ported to the
patched Transformers Cache API or validated in this checkout. Consequently,
no 100K position setting was enabled and no experiment command was launched.
The restored files do not alter the previous experiment history or unrelated
toolkit code.

## Fixed identities

- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Model/tokenizer revision:
  `a10cc1512eabd3dde888204e902eca88bddb4951`
- Historical model SHA-256:
  `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`
- Historical tokenizer SHA-256:
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`

The hashes are copied from prior result records and were not checked against
local artifacts in this run.

## Execution record

- Current source checkout: `b2d441bdbf496b01a3a3538816925a006f357fd4`
- Prototype source: `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`
- Proposed secure candidate examined: Transformers `5.23.0`, PyTorch
  `2.15.0+cpu` / `2.15.0`; advisory lookup found no vulnerabilities for the
  two direct candidate packages. The candidates were not installed.
- A `pip install --dry-run` CPU-index resolution failed when
  `download-r2.pytorch.org` could not be resolved. The PyPI PyTorch wheel
  resolver selected the CUDA distribution and began downloading CUDA
  dependencies; it was stopped before installation to avoid unnecessary
  downloads on this CPU-only host.
- Model/tokenizer endpoint checks failed DNS resolution for both
  `huggingface.co` and `hf.co`. No checkpoint download was attempted.
- No correctness gate passed or failed in this run; all model-dependent gates
  are **NOT RUN**.

## Planned experiment (not run)

Do not treat this section as an executable reproduction command. Before any
run, port the restored cache adapter to a patched Transformers release,
produce and verify a complete hash lock, install the CPU-only runtime in an
isolated environment, provision the exact pinned model snapshot, and pass the
focused, original-range, cache, causal, RoPE, and staged high-position gates.
Only then create a varied tokenizer-produced 100,000-token interaction with
facts near positions 128, 2,048, 8,000, 16,000, 25,000, 50,000, 75,000 and
90,000; verify their actual token offsets after final tokenization; and query
the early facts after processing the full stream. Save resumable progress
without overwriting any prior result.

The existing prototype's cache retains four sink positions and a recent
window (64 positions in its historical 8,193-token engineering run), and
evicts all other K/V entries. No auxiliary retrieval or semantic memory is
part of this protocol. A matched native baseline must use identical weights,
tokens, queries and scoring at a length safe for full-history attention.

There is no safe command for this checkout because the compatibility port,
runtime, model snapshot, and validation are unavailable.
