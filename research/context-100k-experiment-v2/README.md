# SmolLM2 100K position-extension experiment (v2)

This directory contains an isolated Transformers 5 compatibility port and a
reproducible, contiguous 100,000-token experiment using
`HuggingFaceTB/SmolLM2-360M-Instruct`. The original 8,192-position adapter and
the earlier integration work are not modified by this experiment.

The refreshed actual-model preflight passed under the pinned CPU-only runtime.
The contiguous run is separate from the synthetic high-position probe; check
`CORRECTNESS_RESULTS.json` for the latest gate state and `FINAL_REPORT.md` for
the completed experiment results.

## Reproduce or resume

Install the exact hashed runtime dependencies:

```bash
python -m pip install --require-hashes \
  -r research/context-100k-experiment-v2/requirements-lock.txt
```

Download or provide the model snapshot at revision
`a10cc1512eabd3dde888204e902eca88bddb4951`, then run:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python research/context-100k-experiment-v2/run_experiment.py \
  --snapshot /path/to/pinned/model/snapshot \
  --output-dir research/context-100k-experiment-v2 \
  --max-seconds 43200 \
  --min-available-mib 1024 \
  --resume
```

`--resume` starts from the preflight artifacts when no progress exists, or
restores the latest validated checkpoint after an interrupted/incomplete run.
Completed runs are not repeated. The runner validates model, tokenizer, corpus,
stream, and checkpoint provenance before proceeding.

## Artifacts

- `compat_adapter.py`: Transformers 5.23 cache API port.
- `run_experiment.py`: correctness gates, streaming/checkpoint runner,
  retention queries, native-prefix baseline, resource capture, and report
  generation.
- `requirements.in` and `requirements-lock.txt`: direct pins and hashed
  transitive runtime dependencies.
- `CORRECTNESS_RESULTS.json`: actual-model and contiguous-run gate status.
- `dependency-audit.json`: pip-audit result and unsupported-package note.
- `progress.jsonl`, `checkpoints/`, `retention-results.jsonl`,
  `native-baseline.jsonl`, and `resource-results.json`: generated run records.
- `FINAL_REPORT.md`: generated outcome, retrieval table, baseline limits, and
  measured resource results.

The two `*.pre-cpu-wheel.*` files preserve the superseded preliminary preflight;
they are not evidence for the final CPU-only run.

See `EXPERIMENT_PROTOCOL.md`, `ENVIRONMENT.md`, `PROVENANCE.md`, and
`COMPATIBILITY_REPORT.md` for the protocol, pinned environment, artifact
hashes, and security/compatibility caveats.
