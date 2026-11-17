# Reproduction and provenance

## This blocked preflight

Captured on 2026-11-17 at 01:39:03 UTC from local checkout
`a4e8a35c4ae53461f1b88e1334bb50995cdd5e93`.

Commands used to inspect resources and package/model availability:

```bash
nproc
uptime
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
df -B1 --output=source,size,used,avail,pcent /
python -c 'import torch, transformers, tokenizers, safetensors'
```

The package check found all four modules absent. Model cache checks found
`/home/runner/.cache/huggingface/hub`, `/root/.cache/huggingface/hub`, and
`/models` absent. No download command was run.

## Source and model identity

- Inspected implementation commit:
  `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`.
- Model identity in its prior validation records:
  `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Recorded model revision:
  `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Recorded weight SHA-256:
  `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.
- Recorded tokenizer SHA-256:
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.
- These are prior commit records, not hashes recomputed from locally available
  model files. Actual tokenizer counts for this task are unavailable.

## Existing bounded validator (not a 100K reproduction command)

At commit `ee3a7c8`, the existing contiguous validator accepts lengths only
through 65,537 and uses a required local `--snapshot`; it refuses to overwrite
an output path. A bounded run with a locally held, checksum-verified snapshot
would be invoked as follows, using a fresh output directory:

```bash
PYTHONPATH="$PWD" python research/smollm2-integration-v1/validate_contiguous.py \
  --snapshot /path/to/already-local/pinned/snapshot \
  --output /path/to/new-output/contiguous-results.json \
  --length 8193 \
  --max-seconds 2100 \
  --min-available-mib 1024
```

That script only establishes engineering execution and cache-position
invariants. It does not implement the required long-conversation retention
test. It does not support 100,000 tokens, and the 100K task cannot be
reproduced from this checkout until both the requested source and an already
local pinned model/runtime are available and the position ceiling is addressed.
Do not use a remote model-loading command or overwrite any existing result.

## Historical checks (not rerun here)

The prior commit's records report 20 focused tests passed (326 deselected),
bit-identical original-vs-experimental logits in the original range, a
5-token native comparison, and one 8,193-token Zeta contiguous run. They are
bounded historical engineering evidence only. This environment could not
rerun those tests because PyTorch, Transformers, and pytest are absent.
