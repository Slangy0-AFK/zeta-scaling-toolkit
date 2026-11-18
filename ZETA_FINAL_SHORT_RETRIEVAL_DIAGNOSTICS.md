# Zeta Final Short Retrieval Diagnostics

**Run date:** 2026-11-18
**Diagnostics run ID:** `zeta-short-retrieval-diagnostics-20261118-0119-936e6861`

## Scope and safeguards

Two isolated inference diagnostics processed 8,192- and 16,384-token prefixes
of the verified original stream. They used new model and cache instances and
new output files; the completed 100K run, its checkpoints, reports, and
provenance were not resumed or modified. No production inference or cache code
was changed. The 100K test was not rerun.

The original runner is fixed to a 100,000-token target, and its 0/8 retrieval
queries were issued after the full stream. An exact fresh rerun would violate
the no-new-100K-run instruction; loading its final checkpoint would violate
the no-resume instruction. The third diagnostic therefore audits the saved
original prompts, outputs, scoring, and cache-position records without model
inference or checkpoint loading. The audit details its reconstruction limits
below.

MathMesh registered and completed the diagnostics at run level, outside the
token-processing path. Its run ID, configuration hash, and admission/completion
records are in
[`mathmesh-run.json`](research/zeta-final-short-retrieval-diagnostics/mathmesh-run.json).
The isolated wrapper is
[`run_short_diagnostics.py`](research/zeta-final-short-retrieval-diagnostics/run_short_diagnostics.py);
it imports the existing experiment helpers without editing them.

## Pinned inputs and configuration

- **Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Model revision:** `a10cc1512eabd3dde888204e902eca88bddb4951`
- **Tokenizer revision:** `a10cc1512eabd3dde888204e902eca88bddb4951`
- **Weights SHA-256:** `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`
- **Tokenizer SHA-256:** `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`
- **Model config SHA-256:** `224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee`
- **Fact corpus SHA-256:** `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- **Seed:** `20261117`
- **Input:** read-only prefix of the original token stream, SHA-256
  `c9c43fb35050bc31f8d60e2a9bd600ae960f96d7a7ac1618b41505b2955da322`.
- **Source stream file SHA-256:**
  `6c7337e124b9f7e4c9415fd2baeb49e47af36891a5611991cc4d06348099ca34`.
- **Execution:** CPU, float32, eager attention, 64 sink positions plus 64
  recent positions, 64-token chunks, and the existing experimental adapter
  with native unscaled RoPE. The model's native context limit remains 8,192;
  the 16,384 test uses the same position-extension mechanism as the original
  experiment.

The original question template was preserved:
`What is the exact value assigned to fact identifier <ID>? Reply with only the value.`
The exact per-question prompts and outputs are in the machine-readable results
file. Exact-match scoring uses the original digit-boundary rule.

## Results

| Diagnostic | Facts queried | Exact retrieval | Short-prompt positive controls | Negative control | Total wall time | Peak process RSS |
|---|---:|---:|---:|---|---:|---:|
| 8,192-token prefix | 3 | **0/3** | 3/3 | Passed | 767.4 s | 2,479.4 MiB |
| 16,384-token prefix | 4 | **0/4** | 4/4 | Passed | 1,508.9 s | 2,479.8 MiB |
| Original 100K result audit | 8 | **0/8** (saved result) | 8/8 (saved result) | Not part of original test | 45.2 s query evaluation in the original run | 3,400.3 MiB peak for the entire original runner |

### 8,192-token prefix: individual queries

The query starts at position 8,197. Cache positions shown are the recent
64-position range after the question prompt; sink positions 0–63 were also
retained for each query.

| Fact ID | Fact span `[start,end)` | Distance from fact end to query start | Recent cache after question | Exact match | Generated answer |
|---|---:|---:|---:|:---:|---|
| `KBQNPXVP` | `[128,174)` | 8,023 | 8,162–8,225 | No | `The exact value assigned to fact identifier KBQNPXV` |
| `DAYUU8H3` | `[2047,2095)` | 6,102 | 8,163–8,226 | No | `The exact value assigned to fact identifier DAYUU8` |
| `JVJRLNGJ` | `[8002,8048)` | 149 | 8,162–8,225 | No | `The exact value assigned to fact identifier JVJRLNG` |

No token from any queried fact span remained in the cache after its question
was prefetched. All three direct short-prompt controls returned the exact
values. The absent-identifier control asked for `ZZZZZZZZ`; its answer was
`ZZZZZZZZ` and did not match a planted value.

### 16,384-token prefix: individual queries

The query starts at position 16,389. Cache positions shown are the recent
64-position range after the question prompt; sink positions 0–63 were also
retained for each query.

| Fact ID | Fact span `[start,end)` | Distance from fact end to query start | Recent cache after question | Exact match | Generated answer |
|---|---:|---:|---:|:---:|---|
| `KBQNPXVP` | `[128,174)` | 16,215 | 16,354–16,417 | No | `Pixel\n\nvalue of the value of Pixel\n` |
| `DAYUU8H3` | `[2047,2095)` | 14,294 | 16,355–16,418 | No | `\n\n\nvalue of the value of\nvalue of the` |
| `JVJRLNGJ` | `[8002,8048)` | 8,341 | 16,354–16,417 | No | `of the value of the value of the value of the value` |
| `EFHJMHUL` | `[15998,16042)` | 347 | 16,353–16,416 | No | `of the world of the world of the world of world of` |

No token from any queried fact span remained in the cache after its question
was prefetched. All four direct short-prompt controls returned the exact
values. The absent-identifier control asked for `ZZZZZZZZ`; its generated
answer contained none of the planted values.

No errors or numerical anomalies were recorded in either short diagnostic;
all query logits were finite.

The full cache position lists, query suffix lengths, short-control outputs,
answers, token IDs, per-query timings, and prefix integrity measurements are
in [`ZETA_FINAL_SHORT_RETRIEVAL_DIAGNOSTICS.json`](ZETA_FINAL_SHORT_RETRIEVAL_DIAGNOSTICS.json).
Per-test records are also available as
[`8192.json`](research/zeta-final-short-retrieval-diagnostics/8192.json),
[`16384.json`](research/zeta-final-short-retrieval-diagnostics/16384.json),
and
[`original-0-of-8-audit.json`](research/zeta-final-short-retrieval-diagnostics/original-0-of-8-audit.json);
prefix progress records are stored alongside them.

## Original 0/8 result audit

The saved original result contains eight planted facts at starts
128, 2,047, 8,002, 15,998, 24,999, 49,999, 75,001, and 90,000. The eight saved
answers all fail the original exact-match rule; recomputing that rule agrees
with all eight recorded decisions. The eight short-prompt controls pass.
Question suffix tokenization and the runner's cache-position invariant
reconstruct query cache positions consistent with each recorded final logical
cache position. The original rows record no retained positions for the queried
facts.

This is an audit of existing evidence, **not an independent inference
reproduction**. The original protocol used a 100,000-token stream and queried
at position 100,005. The exact test had no negative control; none was added to
the saved original test. Query cache position lists are reconstructed from the
pinned tokenizer, recorded prompt and generation lengths, and the runner's
`expected_positions` rule; they were not separately saved per query.

## Analysis

- **Does retrieval work at 8,192 tokens?** Not for these long-range facts:
  0/3 exact matches. The three direct short-prompt controls passed.
- **Does retrieval work at 16,384 tokens?** Not for these long-range facts:
  0/4 exact matches. The four direct short-prompt controls passed.
- **Is the original 0/8 failure reproducible?** The saved 0/8 scoring and
  positive-control records audit consistently. A fresh exact reproduction
  was not run because it would require the prohibited 100K processing or a
  prohibited checkpoint restore.
- **Does eviction correlate with retrieval failure?** Yes in these tests:
  every queried fact was absent from the bounded cache after question
  prefill, and every corresponding long-range exact-match query failed.
  Short-prompt controls passed when the fact text was present.
- **What limitation is indicated?** The configuration provides only the
  retained sink and recent cache positions to the query; the tested fact spans
  were evicted. This is consistent with a cache-retention limitation for these
  queries. It does not prove cache eviction alone caused the failures: no
  long-context query with the fact still retained was tested, and model or
  prompt limitations are not ruled out.

No further diagnostics were run. The two short tests and the original-result
audit are local, uncommitted outputs; nothing was staged, committed, pushed, or
added to the open pull request.
