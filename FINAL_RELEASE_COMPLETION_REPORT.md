# Final Research Completion and Release Audit

**Status:** Research evidence documented; GitHub release v0.5.1 published;
PyPI publication is blocked by HTTP 403. No experiment or long benchmark was
started during this finalization.

## Research results

### Completed 100K run

The run `zeta100k-fresh-20261117-5bc2985bbe99` processed 100,000 contiguous
tokens through position 99,999 and exited with code 0. Its results are
separate from information retention:

- **Checkpoint and numerical integrity:** 56/56 recorded checkpoints passed
  their integrity checks; cache/logit values were finite. The final cache was
  bounded to 128 positions (64 sink plus 64 recent), 10,518,528 bytes.
- **Provenance:** SmolLM2-360M-Instruct, model and tokenizer revision
  `a10cc1512eabd3dde888204e902eca88bddb4951`, seed `20261117`, CPU FP32.
  Model, tokenizer, corpus, and stream hashes and support-repository commits
  are recorded in [PROVENANCE.md](research/zeta100k-fresh-20261117-5bc2985bbe99/PROVENANCE.md).
- **Performance:** 10.719947 tokens/second; 9,636.662 seconds total wall time;
  peak process RSS 3,400.324 MiB; 18,748.1 process CPU-seconds (about 194.5%
  average process CPU, not whole-host utilization). No matched full-history
  100K baseline exists, so no end-to-end speedup is established.
- **Restoration and outages:** A short checkpoint save/restore smoke passed,
  but the final 100K checkpoint was not restored into a new runner. A bounded
  injected observer-failure smoke passed; physical support-service outage
  isolation during the full run was not tested.

The final checkpoint hash, resource measurements, contiguous-position
records, and runner exit status are in the
[100K final report](research/zeta100k-fresh-20261117-5bc2985bbe99/FINAL_REPORT.md)
and [evidence checklist](ZETA_RESEARCH_EVIDENCE_CHECKLIST.md). The final
checkpoint itself remains local and is not included in the repository or
release distributions.

### Integration evidence and roles

| Repository/service | Observed result | Boundary |
|---|---|---|
| MathMesh | 56/56 telemetry records delivered; zero recorded failed or dropped records. | Governance/audit telemetry; it did not alter model execution. |
| System-Stability-Toolkit | 56 advisory recommendations recorded; zero were applied to Zeta. | Advisory only. |
| Deterministic RAG Governor | Run-level and checkpoint audit calls recorded metadata. | Outside the mathematical execution path; it supplied no tensors, logits, cache, or retrieval context. No retrieval score is assigned to this audit integration. |

These are distinct integration outcomes, not retrieval results. The
[architecture report](research/zeta100k-fresh-20261117-5bc2985bbe99/MULTI_REPOSITORY_ARCHITECTURE.md)
and [evidence checklist](ZETA_RESEARCH_EVIDENCE_CHECKLIST.md) document the
boundaries. Observer failure isolation is established only at the bounded
smoke-test boundary, not for a physical full-run outage. One-time service
coordination took 0.3266 seconds; the available short telemetry on/off
microbenchmark was variable and does not establish a stable throughput cost.

### Information retrieval

| Evidence | Exact-match retrieval | Positive control | Negative control | Runtime / peak RSS | Interpretation |
|---|---:|---:|---|---|---|
| Original 100K result, audited from saved evidence | **0/8** | 8/8 | Not part of the original test | 45.2 s query evaluation / 3,400.3 MiB for the full runner | Audit only; not rerun or independently reproduced. |
| 8,192-token diagnostic | **0/3** | 3/3 | Passed | 767.4 s / 2,479.4 MiB | Tested long-distance facts were absent from the bounded cache after query prefill. |
| 16,384-token diagnostic | **0/4** | 4/4 | Passed | 1,508.9 s / 2,479.8 MiB | Tested long-distance facts were absent from the bounded cache after query prefill. |

The diagnostics used isolated instances and did not resume or modify the
100K run. Their machine-readable results, individual questions/answers,
positions, cache records, timings, memory, and controls are in the
[combined results](ZETA_FINAL_SHORT_RETRIEVAL_DIAGNOSTICS.json), the
[diagnostic report](ZETA_FINAL_SHORT_RETRIEVAL_DIAGNOSTICS.md), and the
[per-test evidence directory](research/zeta-final-short-retrieval-diagnostics/).
The original 0/8 result remains an audit of saved evidence, not a fresh
inference reproduction. Cache eviction correlates with the short-test misses;
the evidence does not establish that eviction alone caused them. These
retrieval results do not indicate failure of stream processing or MathMesh
delivery.

No larger-model, other-architecture, or broad natural-language retrieval
claim is supported by this evidence. The results support an engineering
demonstration for one configuration, not a general retrieval capability,
research novelty claim, or end-to-end speedup claim.

## Validation and repository status

- Local full test suite: **681 passed, 6 skipped**, 96.73% package coverage,
  Python 3.14.2. Ruff lint and formatting passed. The supported-runtime CI
  matrix also checks strict mypy.
- JSON/JSONL syntax, report metric consistency, and targeted secret checks
  passed for the committed research evidence. Local Markdown links in this
  final report and the updated release notes were checked and passed.
- Existing PR: [#11](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/pull/11),
  open, mergeable, not merged, with no review decision recorded.
- Branch: `copilot/experiment-infrastructure-phase-2`. The release-candidate
  source commit is `a866493b40d1d0275ab8857bc2439433d1a332de`. Supplemental
  retrieval records, checkpoint manifests, token streams, and support audit
  snapshots were committed separately in evidence-only commit
  `ac9cfef5aa499c7b1930863f72b7fef4193165d9`. The branch and remote were
  synchronized at that commit before this final documentation update. Those
  evidence files are not included in the 0.5.1 distributions; the worktree was
  clean after the evidence push.
- The final release-status documentation update was committed as
  `404c3e91b369fd38bbb86659f26ce4c82d8a2553`, synchronized to the remote, and
  passed quality plus all seven Python/PyTorch matrix jobs. The `v0.5.1` tag
  points to the package source commit listed above.

### Current CI status

The first CI run exposed two compatibility issues: the monitor used
`hashlib.file_digest`, unavailable before Python 3.11, and PyTorch 2.0 emitted
a `TypedStorage` deprecation warning in three serialization tests while
warnings are configured as errors. The monitor now hashes in chunks, and the
three tests ignore only that specific PyTorch warning; their assertions
remain enabled. The full local suite passes with these changes.

The release-candidate source commit passed quality and all seven Python/PyTorch
runtime matrix jobs in
[run 44223265834](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions/runs/44223265834).
The report-only follow-up commit `f73634189f5e9e017df843e5953ac69902cd8f4d`
also passed quality and all seven matrix jobs in
[run 44223685764](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions/runs/44223685764).
The supplemental evidence commit `ac9cfef5aa499c7b1930863f72b7fef4193165d9`
passed quality and all seven Python/PyTorch matrix jobs in
[run 44225318674](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions/runs/44225318674).
Mutation and nightly jobs were skipped by workflow conditions. The PR remains
open, mergeable, and unreviewed; it was not merged.

## PyPI audit and v0.5.1 release

PyPI still publishes **0.5.0** (wheel SHA-256
`9db3a62d20e447fe894ce0ffc51ce005035a499850e5c01b467f01250d14bb65`;
sdist SHA-256
`fa50983829d74497a8d2d981ac7ea82f2e48c699a2bfe818aaa8f6cfa63393e3`).
The published distributions lack the current `checkpoint.py` and
`provenance.py` modules. Repository metadata is 0.5.1, so a 0.5.1 update is
needed. An upload of the approved 0.5.1 distributions was attempted, but
PyPI returned HTTP 403; the post-attempt PyPI JSON still reports 0.5.0 and
contains no 0.5.1 files. The configured upload credentials therefore need
their PyPI authorization checked before another upload attempt.

Candidate **0.5.1**, built from source commit
`a866493b40d1d0275ab8857bc2439433d1a332de`:

| Artifact | Size | SHA-256 |
|---|---:|---|
| `zeta_scaling_toolkit-0.5.1-py3-none-any.whl` | 28,031 bytes | `ec43b86fb14c593d35e448d9fee44125db512a250274a0809dba841eb31eb62a` |
| `zeta_scaling_toolkit-0.5.1.tar.gz` | 90,460 bytes | `203953e707e32ee78fd79e21baa8546cf519ebb49a4a63041ebc82094c321ddb` |

Both distributions pass `twine check`. Archive inspection confirmed the wheel
contains `checkpoint.py`, `provenance.py`, and `py.typed`; the 48-entry sdist
contains the required package files and no local experiment payloads. Both
artifacts installed into separate virtual environments and passed version,
public-API import, and short finite forward-pass smoke checks. To reuse the
already-installed PyTorch 2.15.0, those environments exposed system site
packages and installed the candidate with `--no-deps`; installation with a
fully empty environment and dependency download was not performed.

The candidate retains the runtime dependency `torch>=2.0.0`; NumPy and
Safetensors are test/development extras. The build succeeded with non-blocking
Setuptools deprecation warnings for the existing TOML-table license field
and MIT license classifier.

GitHub Release [v0.5.1](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/releases/tag/v0.5.1)
was created at the exact package source commit above. Its attached wheel and
sdist sizes and SHA-256 digests match the verified local artifacts. PyPI
publication has **not** succeeded.

## Remaining approvals and blockers

1. PyPI publication is blocked by the HTTP 403 response. Check or replace the
   configured PyPI upload authorization, then upload the exact approved
   artifacts and verify their PyPI hashes.
2. PR #11 remains open and unmerged. Do not merge it without separate
   authorization.
