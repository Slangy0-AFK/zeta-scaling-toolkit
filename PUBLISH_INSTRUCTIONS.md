# v0.5.0 Publishing Instructions

## Status

- ✅ **Version bumped to 0.5.0** in `pyproject.toml`
- ✅ **Git tag v0.5.0 created and pushed**
- ✅ **Comprehensive audit completed and all critical bugs fixed**
- ✅ **All tests passing**
- ❌ **Publishing workflow not yet triggered** (awaiting manual action)

## Why Manual Trigger is Required

The v0.5.0 tag was created **before** the OIDC-based `publish.yml` workflow was implemented. Therefore:

1. The automatic tag-push trigger did not activate the workflow
2. Codespaces token lacks permissions for `workflow_dispatch` API calls
3. Manual GitHub UI trigger is required

## How to Complete Publishing

### Option 1: GitHub UI (RECOMMENDED - Easiest)

1. Go to GitHub Actions: https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions
2. Click the **"Publish to PyPI"** workflow (left sidebar)
3. Click the **"Run workflow"** button
4. Select branch: **main**
5. Leave version_tag empty (default)
6. Click **"Run workflow"**

Expected timeline: 3-8 minutes for the workflow to complete

### Option 2: GitHub CLI (from local machine with write permissions)

```bash
gh workflow run publish.yml \
  --repo Slangy0-AFK/zeta-scaling-toolkit \
  --ref main
```

### Option 3: Force Retrigger via Tag (if needed)

If the tag somehow got lost, recreate it:

```bash
git tag -d v0.5.0
git push origin :refs/tags/v0.5.0
git tag v0.5.0
git push origin v0.5.0
```

This will trigger the workflow automatically via the `push.tags` configuration.

## Monitoring the Workflow

Once triggered, monitor progress here:
- **Actions tab**: https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions/workflows/publish.yml
- **PyPI release page**: https://pypi.org/project/zeta-scaling-toolkit/

Expected workflow steps:
1. Checkout code
2. Set up Python 3.11
3. Install build dependencies
4. Build distribution (wheel + sdist)
5. Publish to PyPI via Trusted Publishing (OIDC)
6. Verify package installation

## What's Being Published

**Package**: zeta-scaling-toolkit v0.5.0
**Critical fixes included**:
- Temperature initialization protection (no divide-by-zero)
- Atomic calibration (all-or-nothing guarantee)
- Sigmoid gradient hardening (recovery from extreme values)
- RoPE overflow protection (log-space computation)
- Comprehensive regression tests

**PyPI location after publishing**:
https://pypi.org/project/zeta-scaling-toolkit/0.5.0/

## After Publishing

Once PyPI confirms publication (1-5 minutes typically):

```bash
# Verify
pip install --upgrade zeta-scaling-toolkit==0.5.0
python -c "from zeta_scaling_toolkit import __version__; print(f'v{__version__}')"
```

Expected output: `v0.5.0`

## Troubleshooting

**Workflow doesn't appear after clicking "Run workflow"?**
- Wait 10-15 seconds and refresh
- Check that you're viewing the correct workflow (publish.yml, not test.yml)

**Workflow completes but PyPI page doesn't update?**
- PyPI indexing can take 5-10 minutes
- Clear pip cache: `pip cache purge`
- Try installing again

**Publishing fails with permissions error?**
- Verify OIDC is configured in PyPI project settings
- Check GitHub Actions environment has `publish` environment configured
- Review workflow logs at: https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions/workflows/publish.yml

## Security Notes

- OIDC authentication is used (no API tokens stored in GitHub)
- Workflow runs with minimal permissions (only id-token and contents:read)
- All secrets are managed through PyPI trusted publishing configuration

## Support

For questions about the publishing process, refer to:
- [GitHub Actions OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [PyPI Trusted Publishing](https://docs.pypi.org/en/latest/trusted-publishers/)
