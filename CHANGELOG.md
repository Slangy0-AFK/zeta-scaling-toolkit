# Changelog

## Unreleased

- Add versioned mid-trial checkpoint/resume infrastructure:
  - `TrialCheckpoint`: Atomic serialization of model, optimizer, scheduler, and RNG state (Python, PyTorch CPU, CUDA)
  - Checkpoint recovery enables resumption from exact next training step without divergence
  - `save_checkpoint_atomic`: Atomic writes via temporary file + replace to prevent corruption
  - Format versioning for checkpoint compatibility across package versions
  - Comprehensive checkpoint/restore and RNG restoration tests
- Add experiment provenance tracking:
  - `ExperimentManifest`: Immutable environment and configuration snapshot
  - Records: Git commit SHA, complete source fingerprints (toolkit + benchmarks), Python/PyTorch versions, OS/CPU/GPU details, precision settings, random seeds, experiment config
  - Manifest compatibility validation: strict (all fields must match) and non-strict (core requirements only)
  - Deterministic source digest computation for reproducibility verification
  - Protocol versioning for manifest schema evolution
- Extend public API: exports `TrialCheckpoint`, `load_checkpoint`, `save_checkpoint_atomic`, `ExperimentManifest`, `compute_source_digest`, `compute_toolkit_source_digest`
- Tests: 39 new tests covering checkpoint capture/restore, RNG state restoration, atomic writes, provenance creation, manifest compatibility, and source digests
- Validation: 681 passed, 6 skipped; 96.73% coverage, above the 95% gate
- Quality: ruff lint, ruff format, and strict mypy all passing
- Set the development version to 0.5.1 for changes after the 0.5.0 release.
- Add resumable context-use experiments and paired reporting, with a
  training-length sanity gate for planted-needle retrieval.
- Add a batched CPU streaming-attention path with output and gradient checks.
- Add NumPy and safetensors to the optional development/test extras required
  by the checkpoint-monitor tests; these are not runtime dependencies.

## 0.5.0

- Published `NonCollapsingAttention.forward_streaming` as a public exact
  sink/window attention path with optional ALiBi and log-length scaling.
- Added dense-equivalence output and gradient coverage across long contexts,
  decode-style inputs, window/sink settings, ALiBi, and chunk sizes; added a
  32,768-token forward finiteness smoke test.
- Extended the optional mutation checker with sink, window-boundary, decode
  alignment, and ALiBi-sign regressions.
- Added golden-value tests for RoPE linear/NTK scaling and ALiBi slopes, plus
  an opt-in full-suite mutation checker for targeted regressions.
- Consolidated duplicate extended-validation benchmark sections and added a
  table of contents; numeric tokens were checked before consolidation.
- Tests: huge-finite-input attention (overflow rescale), softplus temperature
  parameterisation, `calibrate_gain` target and warning behaviour. Found by
  mutation testing: two previously undetected injected bugs are now caught.
- Added deep-stack `calibrate_gain` regression coverage for GELU, SiLU and ReLU
  at 10, 30 and 100 layers, including mode/RNG restoration and failure safety.
- `calibrate_gain` warns (`RuntimeWarning`) when the target norm ratio is not
  reached within 2%.
- `smoke_test.py` is lint-clean and now covered by the CI lint step.
- Documented that `calibrate_gain` equalises batch norm, not per-sample norm.

## 0.4.0

- Added opt-in log-length attention temperature scaling, rotary position
  embeddings, ALiBi biases, causal streaming masks, and a bounded inference KV
  cache with attention sinks.
- Updated the benchmark's optional corrected initialization and added a
  resumable, matched-seed length-generalization evaluation runner.
- Recorded the 500-step matched-seed result: attention-only, initialization-
  only, and the combined variant have higher loss than baseline; see
  [benchmark results](./benchmarks/RESULTS.md).
- Added measured scheduler-equivalence and distractor-mass regression tests.
- New attention length scaling and positional encodings remain opt-in; no
  length-generalization or broad training improvement is claimed.

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
