# Changelog

## Unreleased

- Tests: huge-finite-input attention (overflow rescale), softplus temperature
  parameterisation, `calibrate_gain` target and warning behaviour. Found by
  mutation testing: two previously undetected injected bugs are now caught.
- `calibrate_gain` warns (`RuntimeWarning`) when the target norm ratio is not
  reached within 2%.
- `smoke_test.py` is lint-clean and now covered by the CI lint step.
- Documented that `calibrate_gain` equalises batch norm, not per-sample norm.

## v0.1.0

- Initial stable release of Zeta Scaling Toolkit
- NonCollapsingAttention: QK-normalised attention with learnable temperature
- AsymptoticLRScheduler: Learning rate scheduler with asymptotic behavior
- get_scale_invariant_vit_coords: Scale-invariant coordinates for vision transformers
- CorrectedVarianceLinear: Corrected variance initialization for deep networks
- Full test coverage and comprehensive documentation
