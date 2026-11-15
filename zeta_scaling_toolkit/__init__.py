"""Experimental PyTorch scaling utilities."""

from .core import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    calibrate_gain,
    get_isotropic_coords,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)

__version__ = "0.3.0"
__all__ = [
    "NonCollapsingAttention",
    "AsymptoticLRScheduler",
    "calibrate_gain",
    "get_isotropic_coords",
    "get_scale_invariant_vit_coords",
    "make_corrected_linear",
]
