# Release notes: zeta-scaling-toolkit 0.5.1

**Status:** Prepared candidate; not published

## Summary

This candidate adds trial checkpoint recovery and experiment provenance
utilities, and improves the CPU streaming-attention path. It does not change
the package's runtime dependency on PyTorch. NumPy and safetensors are added
only to the optional development/test extras because the checkpoint-monitor
tests import them.

## Changes

- Add versioned `TrialCheckpoint` save/load support, including model,
  optimizer, scheduler, and Python/PyTorch RNG state.
- Add `ExperimentManifest` environment/configuration snapshots and deterministic
  source digest helpers.
- Export the checkpoint and provenance helpers from the public package API.
- Add a batched CPU streaming-attention implementation with output and gradient
  regression coverage.
- Add or extend tests for checkpoint recovery, RNG restoration, provenance
  compatibility, and source digests.
- Update README, benchmark, and evidence documentation with the completed
  100K run and the separate short retrieval diagnostics. These measurements
  describe research experiments, not package-wide performance or retrieval
  guarantees.

## Compatibility and limitations

- Python support remains `>=3.9`; runtime dependency remains `torch>=2.0.0`.
- Validation on Python 3.14.2 with PyTorch 2.15.0+cu132 (CPU execution; CUDA
  unavailable): 681 passed, 6 skipped, 96.73% package coverage; Ruff and
  strict mypy passed.
- No general large-model, other-architecture, or end-to-end speedup claim is
  made.
- The separate 100K run processed 100,000 tokens with verified checkpoints,
  but its saved long-distance retrieval score was 0/8. The 8,192- and
  16,384-token diagnostics scored 0/3 and 0/4, respectively. These results do
  not establish a general retrieval capability or the cause of the misses.
- The candidate has not been published to PyPI. The current published release
  remains 0.5.0 pending explicit approval of the built artifacts.
