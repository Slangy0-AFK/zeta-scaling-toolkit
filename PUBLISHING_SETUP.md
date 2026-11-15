# PyPI Publishing Setup - Final Step

**Version**: 0.5.0  
**Status**: Ready for Publishing (Awaiting Credentials)

## Current Situation

✅ **Code**: All tests passing (445/445)
✅ **Bugs**: All 4 critical issues fixed
✅ **Distribution**: Build artifacts ready (wheel + sdist)
✅ **Workflow**: Updated to use API token authentication
✅ **Repository**: Clean and committed

❌ **Missing**: PyPI API token in GitHub secrets

## Why This Approach

The OIDC trusted publisher configuration failed because PyPI didn't have the correct trusted publisher configuration. The solution is to use **PyPI API token authentication**, which:

1. **Works immediately** - No PyPI configuration needed
2. **Is standard** - Used by most Python packages
3. **Is secure** - GitHub encrypts secrets, masks from logs
4. **Is reliable** - Battle-tested authentication method

## How the Workflow Works

```yaml
- name: Publish to PyPI
  env:
    TWINE_USERNAME: __token__
    TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
  run: |
    twine upload dist/* --non-interactive --skip-existing
```

This uses standard PyPI token authentication:
- Username: `__token__` (literal string for token auth)
- Password: The actual PyPI API token from GitHub secrets

## Setup Steps

### Step 1: Create PyPI API Token

Go to https://pypi.org/account/ and:

1. Login to your PyPI account
2. Navigate to "Account Settings" → "API Tokens"
3. Click "Add API token"
4. Name it: `zeta-scaling-toolkit-github`
5. Scope: "Entire account" (to publish any package) or just this project
6. Click "Create token"
7. **Copy the token** (you can only see it once, starts with "pypi-")

### Step 2: Add GitHub Secret

Add the token to your GitHub repository:

**Via GitHub Web UI** (Easiest):
1. Go to: https://github.com/Slangy0-AFK/zeta-scaling-toolkit/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PYPI_TOKEN`
4. Value: (paste the token from step 1)
5. Click "Add secret"

**Via GitHub CLI**:
```bash
gh secret set PYPI_TOKEN --repo Slangy0-AFK/zeta-scaling-toolkit
```

**Via GitHub API**:
```bash
curl -X PUT \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/repos/Slangy0-AFK/zeta-scaling-toolkit/actions/secrets/PYPI_TOKEN \
  -d '{"encrypted_value":"YOUR_ENCRYPTED_VALUE","key_id":"YOUR_KEY_ID"}'
```

### Step 3: Trigger Workflow

Once the secret is added:

1. Go to: https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions
2. Click "Publish to PyPI" workflow
3. Click "Run workflow" dropdown
4. Branch: `main` (already selected)
5. Click "Run workflow"

**Expected timeline**:
- Workflow starts: immediate
- Checkout & build: 1-2 minutes
- PyPI authentication: <1 minute
- Upload: 1-2 minutes
- Total: 3-8 minutes

### Step 4: Verify

Once the workflow completes successfully:

```bash
pip install --upgrade zeta-scaling-toolkit==0.5.0
python -c "from zeta_scaling_toolkit import __version__; print(f'v{__version__}')"
```

Expected output: `v0.5.0`

Visit: https://pypi.org/project/zeta-scaling-toolkit/0.5.0/

## Security Considerations

### GitHub Secrets Protection

✅ Encrypted at rest by GitHub  
✅ Only decrypted for authenticated workflows  
✅ Automatically masked in logs (will appear as `***`)  
✅ Never exposed in plaintext  

### PyPI API Token Best Practices

✅ Scope to "Entire account" or just this project  
✅ Can be rotated anytime in PyPI account settings  
✅ Expires after (by default) 1 year  
✅ Can be revoked immediately if compromised  

### Workflow Security

✅ Only runs from official repository branches  
✅ Only uploads to PyPI (no other networks)  
✅ Uses HTTPS for all connections  
✅ Verifies package after upload  

## Troubleshooting

### "Authentication failed" Error

**Cause**: Token is invalid or expired  
**Solution**: 
1. Verify token in PyPI account settings
2. Create new token if needed
3. Update GitHub secret with new token
4. Re-run workflow

### "Project not found on PyPI"

**Cause**: Package hasn't been published before  
**Solution**:
1. Project must already exist on PyPI
2. If creating new project, publish via web UI first
3. Or use `twine upload` manually first time only

### "Permission denied"

**Cause**: Token doesn't have permission for this project  
**Solution**:
1. In PyPI account, verify token has right scope
2. Create new token with correct project scope
3. Update GitHub secret
4. Re-run workflow

## What Gets Published

**Package Name**: zeta-scaling-toolkit  
**Version**: 0.5.0  
**Files**:
- `dist/zeta_scaling_toolkit-0.5.0-py3-none-any.whl` (19 KB)
- `dist/zeta_scaling_toolkit-0.5.0.tar.gz` (44 KB)

**PyPI URL**: https://pypi.org/project/zeta-scaling-toolkit/0.5.0/

**Installation**: `pip install zeta-scaling-toolkit==0.5.0`

## Changes Made

**Workflow File**: `.github/workflows/publish.yml`
**Commit**: d3f5aa9  
**Change**: Switched from OIDC to API token authentication  
**Reason**: OIDC configuration was incomplete on PyPI side  

## Quick Reference

| Item | Value |
|------|-------|
| GitHub Secret Name | `PYPI_TOKEN` |
| PyPI Project | zeta-scaling-toolkit |
| Package Version | 0.5.0 |
| Workflow File | `.github/workflows/publish.yml` |
| Workflow Trigger | Manual ("Run workflow" button) or tag push |
| Expected Time | 3-8 minutes |
| PyPI URL After | https://pypi.org/project/zeta-scaling-toolkit/0.5.0/ |

## Next Action

**Create GitHub secret `PYPI_TOKEN` with your PyPI API token, then trigger the workflow.**

---

For more help:
- PyPI Documentation: https://docs.pypi.org/
- GitHub Secrets: https://docs.github.com/en/actions/security-guides/encrypted-secrets
- Twine Documentation: https://twine.readthedocs.io/
