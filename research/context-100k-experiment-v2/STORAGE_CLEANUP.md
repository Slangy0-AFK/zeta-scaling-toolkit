# Post-Experiment Disk Storage Cleanup Report

**Date:** November 17, 2026  
**Time:** 02:35 UTC  
**Workspace:** /workspaces/zeta-scaling-toolkit  
**Branch:** copilot/experiment-infrastructure-phase-2 (commit ee3a7c8)

## Initial Disk State

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/loop4       32G   28G  2.3G  93% /workspaces
```

**Critical Status:** Only 2.3G (7%) free disk space remaining

### Disk Usage by Major Directory
- benchmarks/ — 124M (experiment results, JSONL, archives, checkpoints)
- research/ — 1.4M (exploratory studies and protocols)
- tests/ — 1.6M (test suites with cached artifacts)
- Python pip cache — 3.4GB (most critical space consumer)
- Pip-tools cache (~/.cache) — 4.4GB
- zeta-smollm-venv (~/.cache) — 1.1GB (required)
- zeta-smollm-hf model (~/.cache) — 694MB (required)

## Cleanup Actions

### 1. Clear pip dependency cache (3.4GB freed)
- **Path:** /home/codespace/.cache/pip/
- **Command:** `python -m pip cache purge`
- **Files Removed:** 604 HTTP cache files
- **Directories Removed:** 1,087 entries
- **Status:** ✓ COMPLETE
- **Recovered:** ~3,387 MB

### 2. Clear Python __pycache__ directories (~800KB freed)
- **Action:** Remove all __pycache__ in repository and subdirectories
- **Locations Affected:**
  - ./benchmarks/__pycache__ (488KB)
  - ./tests/__pycache__ (1.3MB)
  - ./zeta_scaling_toolkit/__pycache__ (180KB)
  - ./research/**/__pycache__ (258KB across subdirectories)
  - ./tools/__pycache__ (12KB)
- **Status:** ✓ COMPLETE
- **Verification:** `find . -type d -name __pycache__` returns 0 results
- **Recovered:** ~2.3 MB

### 3. Clear pytest cache (100KB freed)
- **Path:** .pytest_cache/
- **Status:** ✓ REMOVED
- **Recovered:** ~100 KB

### 4. Clear ruff cache (60KB freed)
- **Path:** .ruff_cache/
- **Status:** ✓ REMOVED
- **Recovered:** ~60 KB

### 5. Clear mypy type-checking cache (not measured but minimal)
- **Path:** .mypy_cache/
- **Status:** ✓ REMOVED

### 6. Remove pip-tools HTTP cache (4.4GB freed)
- **Path:** /home/codespace/.cache/pip-tools/
- **Status:** ✓ REMOVED
- **Recovered:** ~4,400 MB
- **Note:** Pip-tools will automatically regenerate this cache on next dependency resolution

### 7. Remove build artifacts (~200KB freed)
- **Removed:** build/, dist/, *.egg-info
- **Status:** ✓ REMOVED
- **Recovered:** ~200 KB

## Preserved Research Data

✓ All experiment results retained (100% integrity)

### Preserved Directories and Files
1. **benchmarks/** (124M total)
   - context-ability-*.jsonl (experiment records)
   - context-retention-500-analysis.* (analysis report + CSV + SVG)
   - CONTEXT_RETRIEVAL_PROTOCOL.md (protocol spec)
   - context-extrapolation-*.tar.gz (4 archives × ~11-32MB each = 75MB)
   - context-extrapolation-resume/ (41M checkpoints)
   - context-retrieval-v1-checkpoints/ (incomplete sanity run)
   - All .py benchmark scripts

2. **research/** (1.4M)
   - context-100k-experiment-v2/ (exploratory design)
   - independent-retrieval-v1/ (retrieval runner)
   - model-integration-feasibility/ (audit documents)
   - smollm2-integration-v1/ (SmolLM2 integration work)
   - 12 markdown documentation files

3. **Model Weights & Tokenizers** (~1.4GB)
   - ~/.cache/zeta-smollm-hf/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct (694MB)
     - model.safetensors (pinned revision a10cc1512eabd3dde888204e902eca88bddb4951)
     - tokenizer.json
     - config.json
     - All ancillary files
   - ~/.cache/zeta-smollm-venv (1.1GB) — Python environment backup

4. **Git History & Branches** (all intact)
   - Main branch history preserved
   - 15 feature branches retained
   - Current branch: copilot/experiment-infrastructure-phase-2
   - Uncommitted changes preserved:
     - Modified: tests/test_smollm2_positions.py
     - Untracked: 7 research/experiment files
     - Untracked: 3 test files

## Preservation Verification Checklist

✓ All experiment JSONL records present and readable  
✓ All tar.gz archives present and checksummed  
✓ SmolLM2 model weights present and intact  
✓ Tokenizer files present  
✓ All checkpoints (including incomplete) preserved  
✓ Git history and branches untouched  
✓ Research markdown documentation preserved  
✓ Bench mark Python scripts present  
✓ Lock files preserved as experiment artifacts  

## Final Disk State

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/loop4       32G   20G  9.9G  67% /workspaces
```

**Disk Space Recovered:** 7.7GB (from 2.3GB free → 9.9GB free)  
**Utilization Reduction:** 93% → 67% (27-point improvement)

## Environment Verification

✓ **Core imports functional:**
  - zeta_scaling_toolkit (AsymptoticLRScheduler, ExperimentManifest, compute_source_digest)
  - torch
  - benchmarks.context_successor (8 methods available)

✓ **Model loading path intact:**
  - SmolLM2 model accessible at ~/.cache/zeta-smollm-hf/
  - Tokenizer present and loadable
  - Revision hash a10cc1512eabd3dde888204e902eca88bddb4951 matches configuration

✓ **Test suite ready:**
  - pytest environment functional
  - All test files preserved
  - Cache will rebuild on next test run

## Safety Notes

- **Lock files retained:** benchmarks/context-retrieval-*.lock (empty; kept as experiment artifacts)
- **Example JSONL files retained:** context-retrieval-*.examples.jsonl (experiment metadata; not regenerated if deleted)
- **Experiment status files retained:** context-retrieval-*.experiment.json (run state; needed for resume)
- **No git reset or clean performed:** Working directory history and untracked files preserved

## Remaining Storage Risks

✓ **Low-risk caches (safe to clear):** pip-tools cache (already cleared)  
✓ **Model dependencies:** SmolLM2 cache (1.4GB) — required, retained  
✓ **Environment:** zeta-lock-venv (60MB) — reference environment, low priority  
⚠️ **Copilot SDK cache:** /home/codespace/.cache/copilot (518MB) — not managed here

## Conclusion

Post-experiment cleanup successful. **Disk space recovered: 7.7GB** while maintaining 100% research data integrity.

All critical model weights, experiment records, checkpoints, and git history preserved. Environment remains fully functional and ready for the next experiment phase.

**Status: SAFE TO PROCEED** ✓
