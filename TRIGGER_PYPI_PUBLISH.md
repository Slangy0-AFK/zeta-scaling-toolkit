# Trigger v0.5.0 Publishing to PyPI

## ⚠️ Current Situation

The v0.5.0 tag was **created before** the GitHub Actions OIDC publishing workflow existed. This means:
- ✅ Tag v0.5.0 exists locally and on GitHub
- ✅ Build artifacts are ready
- ✅ PyPI Trusted Publisher is configured
- ❌ Workflow has NOT been triggered yet
- ⏳ Need to manually trigger the workflow

---

## ✅ Solution: Manual Workflow Dispatch

The publish workflow now supports `workflow_dispatch` for manual triggering.

### Method 1: GitHub UI (Simplest) ⭐ RECOMMENDED

1. **Visit GitHub Actions:**
   https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions

2. **Select "Publish to PyPI" workflow** (left sidebar)

3. **Click "Run workflow"** (blue button, top right)

4. **Keep defaults** (branch: main, tag: v0.5.0)

5. **Click "Run workflow"** (confirm)

6. **Monitor the run** in real-time

**Expected time to complete:** 3-8 minutes

---

### Method 2: GitHub CLI

```bash
# Trigger workflow from command line
gh workflow run publish.yml --ref main

# Or specify version tag in workflow input (if using inputs)
gh workflow run publish.yml \
  -f version_tag=v0.5.0 \
  --ref main

# Monitor the workflow
gh run list --workflow publish.yml --limit 1
```

---

### Method 3: Re-Push Tag (Forces Workflow Trigger)

If you want to retrigger via tag push:

```bash
cd /workspaces/zeta-scaling-toolkit

# Delete local tag
git tag -d v0.5.0

# Delete remote tag
git push origin :v0.5.0

# Recreate and push tag
git tag v0.5.0 df6e975  # Points to same commit
git push origin v0.5.0

# Workflow should trigger automatically
```

**Note:** This is safe because:
- Same commit (df6e975) is tagged
- Package version doesn't change
- PyPI allows re-publishing with `skip-existing: false`

---

## 🎯 Recommended Path Forward

**Use Method 1 (GitHub UI) - it's the simplest:**

1. Go to: https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions
2. Click: "Publish to PyPI"
3. Click: "Run workflow"
4. Confirm the run
5. Watch the workflow execute

---

## 📊 What Will Happen When Workflow Runs

```
Workflow Start
    ↓
Checkout code at tag v0.5.0
    ↓
Setup Python 3.11
    ↓
Install build tools (build, twine)
    ↓
Build distributions:
  - .whl (wheel)
  - .tar.gz (source)
    ↓
Request OIDC token from GitHub
    ↓
PyPI validates trusted publisher config
    ↓
Upload to PyPI using OIDC token
    ↓
Verification: pip install zeta-scaling-toolkit==0.5.0
    ↓
✅ SUCCESS
```

---

## 🔍 Monitoring the Workflow

### Real-Time Monitoring
1. **Go to:** https://github.com/Slangy0-AFK/zeta-scaling-toolkit/actions
2. **Select:** "Publish to PyPI" (or find your run)
3. **Watch:** Each step in real-time
4. **Check logs:** Click each step for detailed output

### After Completion
1. **Check PyPI:**
   https://pypi.org/project/zeta-scaling-toolkit/0.5.0/

2. **Test Installation:**
   ```bash
   pip install --upgrade zeta-scaling-toolkit==0.5.0
   python -c "import zeta_scaling_toolkit; print('Success!')"
   ```

3. **Verify Version:**
   ```bash
   pip show zeta-scaling-toolkit
   ```

---

## ⚠️ Troubleshooting

### Workflow Doesn't Start
- Refresh the GitHub Actions page
- Check you clicked "Run workflow" on the right workflow
- Verify you're logged in and have write access

### Workflow Fails: 403 Forbidden
- **Cause:** PyPI trusted publisher not properly configured
- **Check:** https://pypi.org/manage/account/publishing/
- **Verify:** Project name, repository, workflow name match exactly
- **Fix:** May need to delete and recreate publisher config

### Workflow Fails: Package Already Exists
- **Not a failure!** Workflow includes `skip-existing: false`
- This intentionally allows republishing
- Can be changed to `skip-existing: true` if preferred

### Workflow Fails: Build Error
- Check Python version (3.11)
- Check dependencies installed correctly
- Review build error in workflow logs
- Fix in code if needed and retry

### Verification Step Fails
- PyPI indexing can take a few seconds
- Verification waits 10 seconds
- Check manually: `pip install zeta-scaling-toolkit==0.5.0`

---

## ✅ Success Indicators

✅ **Workflow runs without errors**  
✅ **All steps complete (green checkmarks)**  
✅ **Package appears on PyPI**  
✅ **pip install works**  
✅ **Trusted publisher shows as VERIFIED**  

---

## 📝 Timeline

| Action | Time |
|--------|------|
| Trigger workflow | Now |
| Build package | ~1-2 min |
| Request OIDC token | ~5-10 sec |
| Upload to PyPI | ~1-2 min |
| Verify installation | ~1 min |
| **Total** | **~5-8 minutes** |

---

## 🎓 Why This Happened

1. v0.5.0 tag was created: **Before** workflow existed
2. OIDC workflow added later: `.github/workflows/publish.yml`
3. Tag push events only trigger on new pushes
4. Old tag doesn't retrigger the new workflow
5. **Solution:** Manual dispatch or re-push tag

---

## 🔐 Security Note

✅ **This is secure:**
- OIDC handles authentication
- No credentials exposed
- Tokens expire immediately
- Complete audit trail logged
- Rerunning same commit is safe

---

## Next Steps

1. **Choose your trigger method** (GitHub UI recommended)
2. **Trigger the workflow**
3. **Monitor the execution** (~5-8 minutes)
4. **Verify on PyPI** after completion
5. **Celebrate! 🎉**

---

## Support

- **GitHub Actions Docs:** https://docs.github.com/en/actions
- **Workflow File:** `.github/workflows/publish.yml`
- **PyPI Project:** https://pypi.org/project/zeta-scaling-toolkit/

---

**Ready to publish? Go to GitHub Actions and click "Run workflow"!**
