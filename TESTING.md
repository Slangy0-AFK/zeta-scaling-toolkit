# Testing

Install the supported development tools and run the test suite:

```bash
python -m pip install -e ".[dev]"
pytest
```

The default Hypothesis profile is deterministic (`derandomize=True`) with up to
200 examples per property. To run the local coverage gate and quality checks:

```bash
pytest --cov=zeta_scaling_toolkit --cov-report=term-missing --cov-fail-under=95
ruff check zeta_scaling_toolkit tests
mypy --strict zeta_scaling_toolkit
```

`tests/test_hardening_regressions.py` pins known numerical and compatibility
failures; `tests/test_properties.py` covers generated shapes, dtypes,
coordinates, schedules, gradients, and deep-stack initialization;
`tests/test_edge_cases.py` checks invalid inputs and failure handling.

`pytest` also includes the toy benchmark setup tests. For the separate
matched-seed char-level evaluation, see
[benchmarks/RESULTS.md](./benchmarks/RESULTS.md) and run the command recorded
there. The benchmark downloads tiny Shakespeare on first use and requires
network access unless `--data-path` points to a local copy.

GitHub Actions runs CPU tests, lint, strict typing, and a nightly extended
Hypothesis profile. CUDA-only tests are marked and skipped when CUDA is not
available. See [.github/workflows/test.yml](./.github/workflows/test.yml).
