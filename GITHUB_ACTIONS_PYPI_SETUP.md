# GitHub Actions PyPI Publishing Setup (OIDC)

This document explains how to configure PyPI publishing using GitHub Actions with OpenID Connect (OIDC) security.

## Overview

The repository includes a GitHub Actions workflow (`.github/workflows/publish.yml`) that automatically publishes tagged releases to PyPI using OIDC for secure, keyless authentication.

**Benefits of OIDC:**
- ✅ No static API tokens needed
- ✅ Automatic token generation per publish
- ✅ Tokens expire after use
- ✅ Fine-grained permission control
- ✅ Audit trail in GitHub Actions logs

## Configuration Steps

### Step 1: Create PyPI Trusted Publisher

1. Visit: https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in the form:

   | Field | Value |
   |-------|-------|
   | **PyPI Project Name** | `zeta-scaling-toolkit` |
   | **Owner** | `Slangy0-AFK` (your GitHub username) |
   | **Repository name** | `zeta-scaling-toolkit` |
   | **Workflow name** | `publish.yml` |
   | **Environment name** | `publish` (optional but recommended) |

4. Click "Add publisher"
5. PyPI will show the pending publisher - it will activate once the first successful workflow run completes

### Step 2: Create GitHub Environment (Optional but Recommended)

For additional security, create a dedicated environment for publishing:

1. Go to: https://github.com/Slangy0-AFK/zeta-scaling-toolkit/settings/environments
2. Click "New environment"
3. Name: `publish`
4. (Optional) Add required reviewers who must approve each publish
5. Click "Configure environment"

### Step 3: Verify Workflow File

The workflow file is already in place:
```
.github/workflows/publish.yml
```

Key configuration:
- **Trigger:** Automatically runs when tags matching `v*` are pushed
- **Environment:** Uses the `publish` environment (if configured)
- **Permissions:** Requests `id-token: write` for OIDC
- **Action:** Uses `pypa/gh-action-pypi-publish@release/v1`

### Step 4: Test the Setup

Trigger a publish by pushing a tag:

```bash
# Already done for v0.5.0:
git tag v0.5.0
git push origin v0.5.0
```

### Step 5: Monitor the Workflow

1. Go to: https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions
2. Find the workflow run for your tag
3. Watch the "Publish to PyPI" job
4. Check the logs for success or errors

## Workflow Details

### Trigger
```yaml
on:
  push:
    tags:
      - 'v*'
```
Runs when any tag starting with `v` is pushed.

### Permissions
```yaml
permissions:
  id-token: write    # For OIDC token generation
  contents: read     # To read repository content
```

### Steps
1. **Checkout** - Gets the repository code
2. **Python Setup** - Installs Python 3.11
3. **Dependencies** - Installs build tools
4. **Build** - Creates wheel and source distributions
5. **Publish** - Uses OIDC to authenticate and publish to PyPI
6. **Verify** - Tests the published package

## PyPI Trusted Publisher Links

- **Learn more:** https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect
- **PyPI Publishers:** https://pypi.org/manage/account/publishing/
- **Setup Guide:** https://docs.pypi.org/trusted-publishers/

## Troubleshooting

### Workflow shows "Pending publisher"
- The PyPI publisher is awaiting the first successful workflow run
- Once the workflow succeeds once, the publisher activates
- Subsequent tags will work automatically

### 401 Unauthorized
- Check that the PyPI trusted publisher is properly configured
- Verify the repository name, workflow name, and environment match exactly
- Check that `id-token: write` permission is set in the workflow

### Package already exists
- The workflow includes `skip-existing: false` to overwrite if needed
- Change to `skip-existing: true` to skip re-publishing the same version

### Verification fails
- PyPI indexing can take a few seconds
- The workflow includes a 10-second wait
- Check https://pypi.org/project/zeta-scaling-toolkit/ manually

## Manual Testing (Backup)

If automated publishing is needed immediately, you can also upload manually:

```bash
# Set your PyPI token
export PYPI_TOKEN="pypi-AgEIc..."

# Upload
cd /workspaces/zeta-scaling-toolkit
twine upload dist/zeta_scaling_toolkit-0.5.0*
```

## Future Releases

For v0.5.1 and beyond:

```bash
# Update version in pyproject.toml
# Commit changes
git add pyproject.toml
git commit -m "Bump version to 0.5.1"

# Tag the release
git tag v0.5.1

# Push both commit and tag
git push origin main
git push origin v0.5.1

# GitHub Actions automatically runs publish workflow
```

## Environment Variables in Workflow

The workflow respects your `pyproject.toml` configuration:
- Version is read from `pyproject.toml`
- Package name is read from `pyproject.toml`
- No hardcoded values - single source of truth

## Security Best Practices

✅ **Implemented:**
- OIDC authentication (no static tokens)
- GitHub environment for approvals (optional)
- Automatic token expiration
- Audit trail in Actions logs
- Read-only repository access during publish

✅ **Recommended:**
- Configure the `publish` environment with required reviewers
- Review workflow logs before publishing to production
- Monitor PyPI project activity
- Keep Python and dependencies updated in workflow

## Support

- GitHub Actions Documentation: https://docs.github.com/en/actions
- PyPI Trusted Publishers: https://docs.pypi.org/trusted-publishers/
- OpenID Connect: https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect

---

**Status:** Ready to use. Just configure the PyPI trusted publisher as described above.
