"""Experimental PyTorch scaling utilities."""

from .checkpoint import TrialCheckpoint, load_checkpoint, save_checkpoint_atomic
from .core import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    calibrate_gain,
    get_isotropic_coords,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)
from .positions import StreamingKVCache, alibi_bias, rotary_embedding, streaming_mask
from .provenance import ExperimentManifest, compute_source_digest, compute_toolkit_source_digest

__version__ = "0.5.1"
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
    "TrialCheckpoint",
    "load_checkpoint",
    "save_checkpoint_atomic",
    "ExperimentManifest",
    "compute_source_digest",
    "compute_toolkit_source_digest",
]
