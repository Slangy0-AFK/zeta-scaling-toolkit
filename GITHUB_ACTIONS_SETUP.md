# GitHub Actions CI/CD Setup Guide

This guide shows how to configure GitHub Actions with your PyPI API key for automated publishing.

## Overview

The project includes two automated workflows:
1. **test.yml** - Runs tests on every commit and PR
2. **publish.yml** - Publishes to PyPI on every release

## Setting Up PyPI Publishing

### Step 1: Create a PyPI API Token

1. Go to https://pypi.org/account/
2. Log in to your PyPI account
3. Navigate to **Account settings** → **API tokens**
4. Click **Add API token**
5. Give it a name (e.g., "zeta-scaling-toolkit-github-actions")
6. Scope: Select **Entire account** or **This project**
7. Copy the token (you won't see it again!)

### Step 2: Add the Secret to GitHub

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. **Name:** `PYPI_API_TOKEN`
5. **Value:** Paste your PyPI API token (the one starting with `pypi-...`)
6. Click **Add secret**

**⚠️ Security Notes:**
- Never commit your API token to code
- Never paste it in issues or pull requests
- If you accidentally expose it, revoke it immediately on PyPI
- The secret is automatically hidden in GitHub logs

### Step 3: Test the Setup

1. Create a new release on GitHub:
   - Go to **Releases** → **Draft a new release**
   - Tag version: `v0.1.0` (or next version)
   - Release title: `Version 0.1.0`
   - Click **Publish release**

2. Watch the workflow execute:
   - Go to **Actions** tab
   - Click the "Publish to PyPI" workflow run
   - Monitor the build and publish steps

3. Verify on PyPI:
   - Visit https://pypi.org/project/zeta-scaling-toolkit/
   - Your new version should appear

## Workflow Details

### Test Workflow (`test.yml`)

**Triggers:**
- Every push to `main` or `develop` branches
- Every pull request to `main` or `develop` branches

**What it does:**
1. Tests on Python 3.9, 3.10, 3.11, 3.12
2. Runs full test suite with coverage
3. Uploads coverage to Codecov
4. Runs linting (Ruff, Black, mypy)

**Status badge:**
```markdown
[![Tests](https://github.com/YOUR-USERNAME/zeta-scaling-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR-USERNAME/zeta-scaling-toolkit/actions)
```

### Publish Workflow (`publish.yml`)

**Triggers:**
- When you publish a release on GitHub

**What it does:**
1. Checks out the code
2. Builds the distribution packages
3. Publishes to PyPI using your `PYPI_API_TOKEN` secret

**Automatic version management:**
- Release version is read from git tag
- Update `pyproject.toml` version before creating a release

## Manual Publishing (If Needed)

If you prefer to publish manually:

```bash
# Build distribution
python -m build

# Upload to PyPI (requires your API token)
python -m twine upload dist/* --repository pypi
```

When prompted for credentials:
- Username: `__token__`
- Password: `your-api-token-here`

## Troubleshooting

### Workflow Not Running
- [ ] Check that the secret `PYPI_API_TOKEN` is set
- [ ] Verify you published a release (not just a draft)
- [ ] Check **Actions** tab for workflow runs
- [ ] Look at workflow logs for errors

### Publish Fails with 403 Forbidden
- [ ] Token may be expired - create a new one
- [ ] Token scope may be too limited - ensure "Entire account"
- [ ] Check PyPI for existing version number

### Upload Succeeds but PyPI Not Updated
- [ ] Refresh PyPI page (might be cached)
- [ ] Check project history: https://pypi.org/project/zeta-scaling-toolkit/#history

## Environment Variables in Workflows

### Available in CI Environment
- `GITHUB_TOKEN` - GitHub's built-in token (pre-configured)
- `PYPI_API_TOKEN` - Your PyPI token (secret you configured)

### Using Secrets in Workflows
```yaml
- name: Example step using secret
  env:
    MY_SECRET: ${{ secrets.PYPI_API_TOKEN }}
  run: echo "Token is configured (length: ${#MY_SECRET})"
```

## Security Best Practices

### For Secret Management
1. ✅ Store tokens only in GitHub Secrets
2. ✅ Use minimal scopes (project-specific tokens if available)
3. ✅ Rotate tokens periodically
4. ✅ Never commit secrets to git
5. ✅ Revoke compromised tokens immediately

### For Repository Access
1. ✅ Limit who can trigger releases
2. ✅ Protect main branch with branch rules
3. ✅ Require PR reviews before merge
4. ✅ Enable branch protection rules

### Workflow Security
1. ✅ Pin action versions (e.g., `actions/setup-python@v4`)
2. ✅ Review external actions
3. ✅ Audit workflow logs for exposed secrets
4. ✅ Keep dependencies updated

## Monitoring Workflows

### View Workflow Runs
1. Go to **Actions** tab
2. Select workflow from the list
3. View individual run details
4. Check logs for each step

### Workflow Badges
Add to README.md:
```markdown
[![Tests](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions)
```

### Subscribe to Notifications
- Go to **Watching** settings
- Choose notification frequency
- Get alerts for workflow failures

## Versioning Strategy

### Semantic Versioning
Follow SemVer: `MAJOR.MINOR.PATCH`

Examples:
- `v0.1.0` - First release
- `v0.2.0` - New features
- `v0.2.1` - Bug fix
- `v1.0.0` - First stable release

### Release Process
1. Update `version` in `pyproject.toml`
2. Create release notes in GitHub
3. Tag as release (e.g., `v0.1.0`)
4. Workflow automatically publishes to PyPI

## Advanced Configuration

### Conditional Publishing
To only publish on specific conditions:

```yaml
jobs:
  deploy:
    if: github.event.release.target_commitish == 'main'
    # ... rest of job
```

### Multi-Environment Publishing
To publish to both PyPI and TestPyPI:

```yaml
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }}

- name: Publish to TestPyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    repository-url: https://test.pypi.org/legacy/
    password: ${{ secrets.TEST_PYPI_API_TOKEN }}
```

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI API Tokens](https://pypi.org/help/#apitoken)
- [Publishing Python Packages](https://packaging.python.org/tutorials/packaging-projects/)
- [Protecting Secrets in GitHub Actions](https://docs.github.com/en/actions/security-guides/encrypted-secrets)

## Questions or Issues?

- Check workflow logs in GitHub Actions tab
- Review GitHub's security guides
- See SECURITY.md for general security practices
