# Runtime environment

## Pinned experiment runtime

The runtime was installed from `requirements-lock.txt` with
`--require-hashes`. `pip check` reported no broken requirements.

| Component | Version |
|---|---|
| Python | 3.14.2 |
| PyTorch | 2.15.0+cpu |
| Transformers | 5.23.0 |
| Tokenizers | 0.23.3 |
| Safetensors | 0.8.0 |
| Hugging Face Hub | 2.4.0 |

CUDA is unavailable; the actual-model gates and experiment run on CPU with
float32 weights. Run-time environment measurements are captured in
`resource-results.json`.

## Dependency audit and limits

`dependency-audit.json` was produced with pip-audit 2.10.1 against the hashed
runtime lock. It reports no known vulnerabilities for packages it could audit.
pip-audit skipped `torch==2.15.0+cpu` because that CPU build is not available
from PyPI. A separate OSV query for the normalized PyPI version `torch 2.15.0`
returned no advisories; this does not constitute a package-database audit of
the CPU wheel itself.

The Transformers-specific CVE checks and their source limitations are documented
in `COMPATIBILITY_REPORT.md`. The experiment loads only the pinned local model,
sets `trust_remote_code=False`, and uses `use_safetensors=True`.

No pip cache cleanup was performed.
