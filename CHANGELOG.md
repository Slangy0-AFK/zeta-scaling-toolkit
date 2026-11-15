# Changelog

## 0.3.0

- Added a lower bound for learnable attention temperature and made fully masked
  rows return zeros.
- Added attention input/shape/device validation and an optional finite-input
  check; empty key sequences now raise while empty query sequences and batches
  are supported.
- Added `calibrate_gain` for sequential, data-dependent activation-norm
  calibration. `make_corrected_linear` rejects gains not supported by PyTorch;
  GELU/SiLU stacks can be calibrated from representative data.
- Added deterministic Hypothesis properties, strict typing and lint checks,
  package typing metadata, coverage-gated CI, and LF line-ending policy.
- Reworked documentation and added a matched-seed char-level benchmark.
- Behavior changes: attention temperature now has default `tau_min=1.0`;
  fully masked rows return zeros instead of NaNs; malformed inputs raise
  explicit errors. Existing positional constructor arguments and deprecated
  no-op arguments remain compatible.
- No training improvement is claimed; see
  [benchmark results](./benchmarks/RESULTS.md) for the measured comparison.
