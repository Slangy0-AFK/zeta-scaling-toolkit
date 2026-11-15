# Contributing

Contributions should preserve the existing public API and include focused tests
for behavior changes. Numerical behavior is experimental; do not describe it as
a training benefit without matched-seed evidence and paired confidence
intervals.

## Development setup

```bash
git switch -c feature/short-description
python -m pip install -e ".[dev]"
```

The package supports Python 3.9+ and PyTorch 2.0+. Runtime dependencies are
limited to PyTorch. The `dev` extra contains pytest, pytest-cov, Hypothesis,
Ruff, and mypy.

## Checks

```bash
pytest
pytest --cov=zeta_scaling_toolkit --cov-report=term-missing --cov-fail-under=95
ruff check zeta_scaling_toolkit tests
mypy --strict zeta_scaling_toolkit
```

See [TESTING.md](./TESTING.md) for the deterministic Hypothesis profile and CI
details. The package uses four-space indentation, 100-character Ruff
formatting, and full type annotations in its public API.

## Pull requests

Keep commits focused, describe user-visible behavior changes, and include the
real output of tests you ran. State benchmark results with the exact command,
seeds, hardware, software versions, and paired confidence intervals. If a
benchmark was not run, say so and do not imply an improvement.
