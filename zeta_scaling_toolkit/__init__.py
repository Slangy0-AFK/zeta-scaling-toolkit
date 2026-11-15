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
from .optimized_attention import (
    OptimizedNonCollapsingAttention,
    AdaptiveAttention,
    StreamingNonCollapsingAttention,
)

__version__ = "0.5.0"
__all__ = [
    "NonCollapsingAttention",
    "OptimizedNonCollapsingAttention",
    "AdaptiveAttention",
    "StreamingNonCollapsingAttention",
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
