# PyPI Upload Guide for zeta-scaling-toolkit v0.5.0

## Current Status ✅

The release package is **fully built and ready**:

```
dist/zeta_scaling_toolkit-0.5.0.tar.gz       (44 KB)
dist/zeta_scaling_toolkit-0.5.0-py3-none-any.whl (19 KB)
```

- Git tag: `v0.5.0` ✅ Created and pushed
- All tests: `445/445` ✅ Passing
- Build artifacts: ✅ Verified

## What Needs to be Done

Upload to PyPI using valid credentials.

## Authentication Error

Current attempt failed with **403 Forbidden**:
- The `~/.pypirc` file references `PYPI_TOKEN` environment variable
- Either the token is not set, or the token is invalid/expired

## Solution: Set Up Valid PyPI Credentials

### Option 1: Using API Token (Recommended) ⭐

1. **Get Your Token:**
   - Visit: https://pypi.org/manage/account/
   - Login with your PyPI account
   - Scroll to "API tokens"
   - Click "Add API token"
   - Name: "zeta-scaling-toolkit" (or any name)
   - Scope: "Entire account"
   - Copy the token (format: `pypi-AgEIc...`)

2. **Export the Token:**
   ```bash
   export PYPI_TOKEN="pypi-AgEIc... [paste your full token here]"
   ```

3. **Upload the Package:**
   ```bash
   cd /workspaces/zeta-scaling-toolkit
   twine upload dist/zeta_scaling_toolkit-0.5.0* --verbose
   ```

### Option 2: Using Username & Password

1. **Edit ~/.pypirc:**
   ```ini
   [pypi]
   repository = https://upload.pypi.org/legacy/
   username = your_pypi_username
   password = your_pypi_password
   ```

2. **Upload the Package:**
   ```bash
   cd /workspaces/zeta-scaling-toolkit
   twine upload dist/zeta_scaling_toolkit-0.5.0*
   ```

### Option 3: Command-line Credentials

```bash
twine upload dist/zeta_scaling_toolkit-0.5.0* \
  --username __token__ \
  --password "pypi-AgEIc... [your token]"
```

## Verification After Upload

Once successfully uploaded to PyPI:

1. **Check PyPI:**
   ```
   https://pypi.org/project/zeta-scaling-toolkit/0.5.0/
   ```

2. **Test Installation:**
   ```bash
   pip install zeta-scaling-toolkit==0.5.0 --upgrade
   python -c "import zeta_scaling_toolkit; print(zeta_scaling_toolkit.__version__)"
   ```

3. **Create GitHub Release:**
   - Visit: https://github.com/Slangy0-AFK/zeta-scaling-toolkit/releases/tag/v0.5.0
   - Click "Edit"
   - Set as release (not just a tag)
   - Upload: `dist/zeta_scaling_toolkit-0.5.0.tar.gz`
   - Upload: `dist/zeta_scaling_toolkit-0.5.0-py3-none-any.whl`
   - Use content from `RELEASE_NOTES_v0.5.0.md` as description

## If Token is Expired

If you previously set a PyPI token that's now expired:

1. Visit: https://pypi.org/manage/account/
2. Find the expired token in "API tokens"
3. Click the trash icon to delete it
4. Create a new token (follow "Option 1" above)
5. Re-export the environment variable and retry

## Build Artifacts Ready ✅

All build artifacts are already prepared in:
```
/workspaces/zeta-scaling-toolkit/dist/
```

Files are verified and ready to upload immediately once credentials are provided.

## Timeline

- **Released:** v0.5.0 on November 15, 2026
- **Fixes:** 4 critical bugs + 30 regression tests
- **Tests:** All 445 tests passing
- **Status:** Production-ready, awaiting PyPI upload

---

**Note:** The release itself is complete. Only the PyPI upload step requires valid credentials.
