# Fresh 100K Run Protocol

For subsequent qualification and the latest gate check, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Status: BLOCKED BEFORE FULL LAUNCH

Run ID: `zeta100k-fresh-20261117-5bc2985bbe99`

This is a new experiment, not a continuation of the paused experiment. No
100,000-token inference run was started. Do not start it until the resource and
integration gates below pass.

## Preservation and isolation

- The previous experiment remains at `research/context-100k-experiment-v2/`.
- Its progress, logs, stream, and checkpoints were not changed or reused.
- Verified original checkpoint SHA-256 values:
  - 14,336: `0da6331d43beaa9c1379aa707d1009b75edef514f68c6c5457e1648046d6bba2`
  - 16,384: `e7e68476707359afa451e50c9222f89319c1291ddfde3654f13dfbeb856ae2ce`
  - 18,432: `ffaea7046c50487407f22385d0e8fea7e4d42cf8011bc27472fb4818532b14f3`
- New qualification and support artifacts are confined to this run-ID directory.
- The full-run command must omit `--resume` and use this new output directory.

## Frozen configuration

| Setting | Value |
|---|---|
| Model | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Model revision | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| Seed | 20261117 |
| Target | 100,000 contiguous tokenizer-produced tokens |
| Chunk size | 64 |
| Sink cache | 64 positions |
| Recent cache | 64 positions |
| Precision/device | float32 / CPU |
| Runtime safety gate | At least 1,024 MiB configured reserve and 2,560 MiB model-load headroom |

Model weights, tokenizer, config, corpus, and generated stream identities are
listed in [PROVENANCE.md](PROVENANCE.md). The standalone preflight regenerated
the expected stream SHA-256 and passed its correctness gates.

## Qualification completed

- Standalone Zeta actual-model preflight: passed in the new
  `qualification/standalone/` directory.
- MathMesh, Stability Toolkit, and RAG Governor: all handled pre-run
  coordination/advisory/audit calls. Their exact results are in
  `integration-state.json` and `resource-recommendation.json`.
- Support-service unit checks passed as recorded in
  [INTEGRATION_SAFETY_REPORT.md](INTEGRATION_SAFETY_REPORT.md).
- The matched Zeta preflight with MathMesh connected stopped before model load
  because available memory was below the runner gate. The RAG-only,
  Stability-only, and all-connected Zeta preflights were not completed.

## Launch gate

The runner requires at least 3,584 MiB available before model load. The latest
recorded availability was 2,794 MiB, a 790 MiB shortfall. No threshold was
lowered and no Zeta cache, position, precision, or chunk setting was changed.
See `launch-gate.json` and [100K_PIPELINE_INTEGRITY_REPORT.md](100K_PIPELINE_INTEGRITY_REPORT.md).

Before a later launch, all four repositories must remain pinned to the commits
in `PROVENANCE.md`; available memory must meet the runner gate; and the
single-service and all-service actual-model preflights must pass with the same
model, corpus, stream hash, and configuration. Support-service outages must
also be tested without invoking network calls from the Zeta forward path.

## Fresh launch command (not authorized while blocked)

After every gate passes, first run a fresh preflight-only invocation into a new
qualification subdirectory. Then launch the full run with the following shape,
with no `--resume` flag:

```bash
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/tmp/zeta100k-fresh-5bc2985bbe99/bin/python \
research/context-100k-experiment-v2/run_experiment.py \
  --snapshot /home/codespace/.cache/zeta-smollm-hf/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951 \
  --corpus /home/codespace/.cache/zeta-scaling-toolkit/tinyshakespeare.txt \
  --output-dir research/zeta100k-fresh-20261117-5bc2985bbe99 \
  --max-seconds 43200 --min-available-mib 1024
```

This command is documented for reproducibility only. The current run status is
blocked; do not execute it until the launch gate and qualifications are
re-evaluated. The runner writes its own progress, atomic checkpoints, resource
measurements, retrieval results, and final report under the new directory.
