"""Experimental PyTorch scaling utilities."""

from .core import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)

__version__ = "0.1.0"
__all__ = [
    "NonCollapsingAttention",
    "AsymptoticLRScheduler",
    "get_scale_invariant_vit_coords",
    "make_corrected_linear",
]
