# Release Notes: zeta-scaling-toolkit v0.5.0

**Release Date:** November 15, 2026

## 🎯 Release Summary

This release includes critical bug fixes, comprehensive testing improvements, and enhanced numerical stability across the toolkit.

## 🔴 Critical Fixes

### 1. Temperature Initialization Division by Zero (CRITICAL)
- **Issue:** `NonCollapsingAttention.__init__()` crashed when `tau_min == tau_max`
- **Root Cause:** Unchecked division in sigmoid parameterization initialization
- **Fix:** Added strict validation `tau_min < tau_max` with clear error messages
- **Impact:** Prevents silent initialization failures and provides clear guidance to users

### 2. calibrate_gain Partial Mutation Bug (CRITICAL)
- **Issue:** Model parameters modified during calibration; if later layer fails, earlier layers remain changed
- **Root Cause:** No transactional guarantees; parameters modified before validation complete
- **Fix:** Implemented atomic parameter snapshots with automatic reversion on any failure
- **Impact:** Guarantees all-or-nothing calibration semantics

### 3. Temperature Documentation Inconsistency (HIGH)
- **Issue:** README described clamping; code uses sigmoid-bounded parameterization
- **Fix:** Updated all documentation to accurately reflect implementation
- **Impact:** Users now understand actual numerical behavior

### 4. RoPE NTK Overflow (HIGH)
- **Issue:** `scale_factor ** exponent` could overflow in rotary embedding with extreme scale factors
- **Fix:** Moved computation to log-space: `exp(-log_base * frequencies / dimension)`
- **Impact:** Handles scale factors up to 1e308 without overflow

## ✅ Testing Improvements

- **Test Count:** 415 → 445 (+30 regression tests)
- **Coverage:** Added comprehensive test suite `test_audit_fixes.py`
- **Test Categories:**
  - Temperature initialization validation (8 tests)
  - Temperature gradient behavior (12 stress tests)
  - calibrate_gain atomicity (5 tests)
  - Numerical stability (5 tests)

## 📊 Verification Results

| Component | Status |
|-----------|--------|
| All Tests | ✅ 445/445 passing |
| Type Checks (mypy) | ✅ Pass |
| Linting (ruff) | ✅ Pass |
| Regressions | ✅ None |
| API Compatibility | ✅ 100% backward compatible |

## 📦 Build Information

- **Python Version:** 3.9+
- **Dependencies:** torch>=2.0.0
- **Build System:** setuptools + wheel
- **Package Files:**
  - `zeta_scaling_toolkit-0.5.0.tar.gz` (43.6 KB)
  - `zeta_scaling_toolkit-0.5.0-py3-none-any.whl` (18.4 KB)

## 🔗 Installation

```bash
pip install zeta-scaling-toolkit==0.5.0
```

## 📝 Changes

### Core Changes
- `zeta_scaling_toolkit/core.py`
  - Temperature bounds validation (lines 86-98)
  - Transactional calibrate_gain (lines 604-730)

- `zeta_scaling_toolkit/positions.py`
  - NTK scaling moved to log-space (lines 56-63)

### Documentation
- `README.md` - Updated temperature parameterization and limitations

### Tests
- `tests/test_audit_fixes.py` - New comprehensive regression tests (310 lines)

## 🚀 Git Commits

1. `6dcc439` - Avoid overflow in rotary NTK scaling
2. `d6161af` - Fix temperature gradient dead-zone with bounded sigmoid parameterization
3. `df6e975` - Comprehensive audit fixes: temperature bounds validation, calibrate_gain atomicity, documentation

**Tag:** `v0.5.0` on main branch

## 📋 Known Limitations

1. **Sigmoid Saturation:** At extreme raw_tau values (e.g., -1000), sigmoid produces zero gradients. This is mathematically unavoidable but unrealistic in practice (gradients remain non-zero for raw_tau ≥ -20).

2. **Batch-Level Calibration:** calibrate_gain operates at batch norm level, not per-sample. Deep networks without layer norm can show median collapse while batch norm is preserved.

3. **CUDA Testing:** GPU-based tests require CUDA hardware. CI matrix tests all platforms.

## 🔒 Security

- No security regressions introduced
- All numerical edge cases hardened against overflow/underflow
- Input validation strengthened across the toolkit

## 📧 Support

For issues or questions about this release:
- GitHub Issues: https://github.com/Slangy0-AFK/zeta-scaling-toolkit/issues
- Documentation: https://github.com/Slangy0-AFK/zeta-scaling-toolkit#readme

---

**Quality:** Production-ready with comprehensive testing  
**Breaking Changes:** None  
**Migration Path:** Drop-in replacement for 0.4.0
