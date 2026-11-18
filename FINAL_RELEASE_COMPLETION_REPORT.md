# Final Research Completion and Release Audit

**Status:** Research evidence documented; candidate 0.5.1 prepared but not
published. No experiment or long benchmark was started during this
finalization.

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
- JSON/JSONL syntax, report metric consistency, local Markdown links, and
  targeted secret checks passed for the previously committed evidence and
  documentation. The final report is subject to the same local-link check
  before commit.
- Existing PR: [#11](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/pull/11),
  open, mergeable, not merged, with no review decision recorded.
- Branch: `copilot/experiment-infrastructure-phase-2`. The release-candidate
  source commit is `a866493b40d1d0275ab8857bc2439433d1a332de`; the remote branch
  matched this SHA when checked. This report is a documentation-only follow-up
  to that package source commit.
- Pre-existing unrelated working-tree changes and local experiment payloads
  remain unstaged and uncommitted; they were not included in the release
  candidate or the commits described here.

### Current CI status

The first CI run exposed two compatibility issues: the monitor used
`hashlib.file_digest`, unavailable before Python 3.11, and PyTorch 2.0 emitted
a `TypedStorage` deprecation warning in three serialization tests while
warnings are configured as errors. The monitor now hashes in chunks, and the
three tests ignore only that specific PyTorch warning; their assertions
remain enabled. The full local suite passes with these changes.

The latest pushed-commit workflow is
[run 44223265834](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions/runs/44223265834).
Quality and all seven Python/PyTorch runtime matrix jobs passed. Mutation and
nightly jobs were skipped by workflow conditions. The PR remains open,
mergeable, and unreviewed; it was not merged.

## PyPI audit and 0.5.1 candidate

PyPI currently publishes **0.5.0** (wheel SHA-256
`9db3a62d20e447fe894ce0ffc51ce005035a499850e5c01b467f01250d14bb65`;
sdist SHA-256
`fa50983829d74497a8d2d981ac7ea82f2e48c699a2bfe818aaa8f6cfa63393e3`).
The published distributions lack the current `checkpoint.py` and
`provenance.py` modules. Repository metadata is 0.5.1, so a 0.5.1 update is
needed. The latest GitHub Release is `v0.1.0`; there is no GitHub release
corresponding to this candidate.

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
and MIT license classifier. PyPI publication has **not** occurred.

## Remaining approvals and blockers

1. Wait for the pending CI runtime matrix jobs and record their final result.
2. PyPI publication requires explicit approval of **version 0.5.1**, source
   commit `a866493b40d1d0275ab8857bc2439433d1a332de`, and the two artifact
   hashes above. Nothing has been uploaded.
3. If publication is approved, publish these exact artifacts, verify their
   PyPI hashes, and create/update a matching GitHub Release. Do not merge PR
   #11 without separate authorization.
