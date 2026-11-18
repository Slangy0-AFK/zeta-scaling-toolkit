# Provenance

For subsequent audit-store hashes and qualification evidence, see
[QUALIFICATION_RECHECK_20261117.md](QUALIFICATION_RECHECK_20261117.md).

## New run identity

- Run ID: `zeta100k-fresh-20261117-5bc2985bbe99`.
- Status: blocked before the full contiguous inference run.
- Primary Zeta checkout: `cc2b8dfb45dd887f6f202ca69b8c874410fc4645`.
- Zeta experiment source prototype recorded by the runner:
  `ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed`.
- Zeta branch: `copilot/experiment-infrastructure-phase-2`.
- Pre-existing unrelated edits in retrieval benchmark records/manifests were
  left untouched.

## Supporting repositories

| Repository | Exact commit | Runtime / notes |
|---|---|---|
| MathMesh Governor | `4ef77424b7ef4c5b131f558e85fa1eae24d890ab` | Node 24.22.2, npm 11.19.1; `package-lock.json` SHA-256 `cc4e6467b8bc56ee36a6c093c7f9c1c96887fe125cce06cfe3e96c2f6a80e30f`; 642 packages installed. |
| System-Stability-Toolkit | `9e7a254ee9607b23ae35b756110d3713ad0dac45` | Python standard library; `crash_fix.py` SHA-256 recorded by source commit. |
| Deterministic RAG Governor | `7eee42bc77757e8a23138b9b65c8a1072c88ca9d` | Python 3.11 container `python@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535`; pinned API packages. |

MathMesh `package.json` SHA-256:
`dc9ffcbde5eb7c354af22fe323c204183f7574bee32b1cfad15cf53139cd089e`.
Stability Toolkit `crash_fix.py` SHA-256:
`4720c2b37871365ad36dc97981610eef7719d9943db62b931d29024216a1cbd1`.
RAG `requirements.txt` SHA-256:
`d9ed0247dc7d6ad77d228e556d2563c91d2fd457b89e74b15d4f9b334e6d596a`.
RAG `governor.py` SHA-256:
`d8ae83d54e8d649cbd70c409868220ed799f91b1ee83e9028703654af5ae2820`.
The runtime container also used PostgreSQL image
`postgres@sha256:a77aa70e641402a21411b92b721c58d10f23b50ae5fdd2333efd22d166e1e5f7`.

## Zeta software and input identities

- Python 3.14.2; PyTorch 2.15.0+cpu; Transformers 5.23.0;
  Tokenizers 0.23.3; Safetensors 0.8.0; Hugging Face Hub 2.4.0.
- Zeta lockfile SHA-256:
  `2fb39bb9cfde274d733425f50002b69d21b37693a6302976d510fd6e9d28095c`.
- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`, revision
  `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Model weights SHA-256:
  `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.
- Tokenizer SHA-256:
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.
- Config SHA-256:
  `224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee`.
- Corpus SHA-256:
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- Stream SHA-256, 100,000 token IDs:
  `c9c43fb35050bc31f8d60e2a9bd600ae960f96d7a7ac1618b41505b2955da322`.
- Configuration: seed 20261117; chunk 64; 64 sink positions + 64 recent
  positions; float32; CPU.

## Audit and preservation artifacts

- Run-level integration calls and timings: `integration-state.json`.
- Resource recommendation, separately from applied configuration:
  `resource-recommendation.json`.
- Latest launch-gate snapshot: 2,794 MiB available at 2026-11-17T20:43:55Z;
  790 MiB below the required threshold.
- MathMesh database export SHA-256:
  `78ef72c2c876af41c9a3f28b03fab6c61e0a738bc2b0109d543e04a6253cc20a`.
  The plain-SQL dump was restored with `psql` into a disposable database; the
  restored entities included 6 AuditLog rows and 2 PipelineRun rows.
- RAG SQLite audit database SHA-256:
  `98f84284d1e5ba13dc0936aa936d3be6c0fc2a4e354f01627bab46b2f05d1bca`.
  The service reported `chain_valid: true` after restart.
- Original experiment checkpoint hashes are preserved in
  [FRESH_RUN_PROTOCOL.md](FRESH_RUN_PROTOCOL.md).

The previous experiment's checkpoint, progress, and log artifacts were not
used to initialize this run. No full-run checkpoint exists for this run ID.
