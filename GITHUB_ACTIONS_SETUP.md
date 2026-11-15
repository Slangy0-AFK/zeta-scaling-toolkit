# GitHub Actions

The repository's workflows are defined in `.github/workflows/`:

- `test.yml` runs CPU tests with coverage, Ruff, and strict mypy on pull requests
  and pushes to `main` or `develop`. A scheduled run uses the extended
  Hypothesis profile.
- `publish.yml` builds and publishes a package when a GitHub release is
  published.

The publish workflow requires a repository Actions secret named
`PYPI_API_TOKEN`. Keep the token in GitHub Secrets; never place it in a workflow
file, command history, issue, or pull request. Publishing is separate from the
test workflow and is not required to run CI.
