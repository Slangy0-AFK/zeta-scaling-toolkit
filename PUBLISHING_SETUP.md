# PyPI Publishing Setup

The `Publish to PyPI` workflow builds distributions and publishes them using
PyPI Trusted Publishing (GitHub Actions OIDC). No PyPI API token or
`PYPI_TOKEN` GitHub secret is required.

## Configure PyPI Trusted Publishing

In the PyPI project settings, add a trusted publisher with:

| Field | Value |
|-------|-------|
| PyPI project name | `zeta-scaling-toolkit` |
| Owner | `Slangy0-AFK` |
| Repository name | `zeta-scaling-toolkit` |
| Workflow name | `publish.yml` |
| Environment name | Leave blank |

The environment must be left blank because the publishing job does not declare
a GitHub environment. See
[GitHub Actions PyPI Publishing Setup](GITHUB_ACTIONS_PYPI_SETUP.md) for more
details.

## Publish a Release

Push a version tag to start publishing:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The workflow builds the source distribution and wheel in a job without PyPI
credentials, then publishes those artifacts in a separate OIDC-enabled job.
After a successful upload, it installs the published version and verifies that
the package imports.

The release tag version must match `version` in `pyproject.toml`. PyPI does not
allow replacing an already-published release; `skip-existing` lets retries
skip files that were uploaded successfully before a failure.

## Troubleshooting

- **Trusted publisher not found:** Confirm the owner, repository, workflow
  filename, and blank environment match the PyPI publisher configuration.
- **Package version already exists:** Bump the version in `pyproject.toml` and
  push a new version tag.
- **Verification cannot find the package:** PyPI indexing can take time; check
  the project page and rerun verification after it finishes indexing.
