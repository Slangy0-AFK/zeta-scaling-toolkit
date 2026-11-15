"""
Zeta Scaling Toolkit - Self-Normalizing Scaling Utilities for AI Pipelines

Based on the mathematical framework: W(T) = 1 + A0 / ln(T)
Implements 4 core fixes for numerical stability, scaling, and initialization.
"""

from .core import (
    A0,
    AsymptoticLRScheduler,
    CorrectedVarianceLinear,
    NonCollapsingAttention,
    get_scale_invariant_vit_coords,
)

__version__ = "0.1.0"
__all__ = [
    "NonCollapsingAttention",
    "AsymptoticLRScheduler",
    "get_scale_invariant_vit_coords",
    "CorrectedVarianceLinear",
    "A0",
]
