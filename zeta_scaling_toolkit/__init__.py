"""Experimental PyTorch scaling utilities."""

from .core import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    calibrate_gain,
    get_isotropic_coords,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)
from .positions import StreamingKVCache, alibi_bias, rotary_embedding, streaming_mask

__version__ = "0.4.0"
__all__ = [
    "NonCollapsingAttention",
    "AsymptoticLRScheduler",
    "calibrate_gain",
    "get_isotropic_coords",
    "get_scale_invariant_vit_coords",
    "make_corrected_linear",
    "rotary_embedding",
    "alibi_bias",
    "streaming_mask",
    "StreamingKVCache",
]
