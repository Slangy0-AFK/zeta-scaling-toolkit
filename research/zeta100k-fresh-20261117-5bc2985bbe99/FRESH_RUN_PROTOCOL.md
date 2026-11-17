# Fresh 100K Run Protocol

For subsequent qualification and the latest gate check, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## Current status: RUNNING

Run ID: `zeta100k-fresh-20261117-5bc2985bbe99`

This is the authorized fresh experiment, not a continuation of the paused
experiment. The full runner is active as PID `12915`, launched from starting
commit `cc2b8dfb45dd887f6f202ca69b8c874410fc4645`, without `--resume`.

At the 2026-11-17 22:19 UTC status check, it had committed a verified checkpoint
at 6,144/100,000 contiguous tokens (last position 6,143), averaging 10.389
tokens/s over the latest checkpoint interval. The checkpoint SHA-256 is
`7d71caa1be6ec4cd881a4472d2c97bdd2c5699fa624aab4678a94ad9a051ccf6`.
The runner, MathMesh telemetry sidecar (PID `12787`), and Stability advisory
sidecar (PID `17641`) are separate processes.

The existing Codespace reported the approved `standardLinux32gb` specification;
no additional resize operation was needed. It has 4 CPUs, 15,993.6 MiB total
RAM, no cgroup memory cap, and 9,799.7 MiB host `MemAvailable` at the latest
check. The launch gate remained 3,584 MiB. The runner-reported peak RSS through
the latest checkpoint was 2,480.3 MiB. Workspace storage had 4.7 GiB free.

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

## Qualification completed before launch

- Standalone and post-resize actual-model preflights passed with the same
  model, tokenizer, corpus, and stream hashes. Their correctness result fields
  match; both reported finite logits/cache, causal invariance, position
  behavior, and parity checks passed.
- All three support systems passed run-level admission/advisory/audit calls.
  MathMesh `/api/health` and RAG `/ready` were healthy at launch.
- Checkpoint save/load recovery was verified on the first 8 real stream tokens:
  resumed logits were bit-identical to uninterrupted logits.
- Focused telemetry, service-failure, and Stability watcher tests passed.
  Live MathMesh telemetry delivered the 2,048-token checkpoint; injected
  telemetry failure did not prevent finite model inference.
- Detailed evidence and limits are in
  [INTEGRATION_SAFETY_REPORT.md](INTEGRATION_SAFETY_REPORT.md) and
  [NON_INTERFERENCE_RESULTS.md](NON_INTERFERENCE_RESULTS.md).

## Launch gate

The initial 3,584 MiB model-load gate passed with 9,799.7 MiB currently
available. No threshold or mathematical setting was changed. Current gate
details and checkpoint evidence are in `launch-gate.json` and
[100K_PIPELINE_INTEGRITY_REPORT.md](100K_PIPELINE_INTEGRITY_REPORT.md).

## Reproducible launch command

The following is the command already running; do not launch a duplicate:

```bash
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/codespace/.cache/zeta100k-fresh-20261117-5bc2985bbe99-v2/bin/python \
research/context-100k-experiment-v2/run_experiment.py \
  --snapshot /home/codespace/.cache/zeta-smollm-hf/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951 \
  --corpus /home/codespace/.cache/zeta-scaling-toolkit/tinyshakespeare.txt \
  --output-dir research/zeta100k-fresh-20261117-5bc2985bbe99 \
  --max-seconds 43200 --min-available-mib 1024
```

The runner writes progress, atomic checkpoints, resource measurements,
retrieval results, and its final report under this run-ID directory. The old
experiment and all of its checkpoints remain untouched.
