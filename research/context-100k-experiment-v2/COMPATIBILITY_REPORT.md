# Compatibility and security report

## Transformers 5 port

`compat_adapter.py` adapts the experimental cache to the Transformers 5.23
`Cache`/layer API while keeping the original adapter unchanged. It handles
explicit positions and native default RoPE configuration. The native context
limit remains the default; extension requires explicit configuration.

Focused adapter and resume tests pass under the locked runtime. The refreshed
actual-model preflight also passed. Its detailed numerical results and the
pending contiguous-run gate are recorded in `CORRECTNESS_RESULTS.json`.

## Security checks

- The pinned runtime uses Transformers 5.23.0. GitHub's reviewed advisories
  list CVE-2025-6051 as fixed in 4.53.0 and CVE-2026-4372 as fixed in 5.3.0;
  the installed version is later than both fixed versions.
- The CVE-2025-14920 advisory from ZDI/GitHub does not specify affected or
  patched version ranges. Snyk recommends Transformers 5.0.0rc1 or later;
  5.23.0 exceeds that recommendation, but the upstream advisory does not
  independently confirm the patch boundary. The issue concerns Perceiver model
  deserialization; this run uses a local Llama safetensors snapshot and does
  not load Perceiver files. This is a scoped applicability assessment, not a
  claim that the advisory's version range is authoritatively resolved.
- `pip-audit` found no known vulnerabilities among the lock entries it could
  audit, but skipped the `torch==2.15.0+cpu` distribution because it is not
  indexed on PyPI. The normalized PyPI release `torch 2.15.0` had no advisories
  in the separate OSV query; the exact CPU wheel remains outside pip-audit's
  direct coverage.
- Model loading is pinned to a verified revision, offline, with
  `trust_remote_code=False` and `use_safetensors=True`.

Advisory references:

- [CVE-2025-6051 / GHSA-rcv9-qm8p-9p6j](https://github.com/advisories/GHSA-rcv9-qm8p-9p6j)
- [CVE-2026-4372 / GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72)
- [CVE-2025-14920 / GHSA-7hhx-w23w-fg5v](https://github.com/advisories/GHSA-7hhx-w23w-fg5v)
- [Snyk CVE-2025-14920 entry](https://security.snyk.io/vuln/SNYK-PYTHON-TRANSFORMERS-14564337)
