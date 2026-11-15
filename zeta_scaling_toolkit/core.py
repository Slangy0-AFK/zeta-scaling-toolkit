"""
zeta_scaling_toolkit.core - Self-Normalizing Scaling Utilities for AI Pipelines

Based on the mathematical framework: W(T) = 1 + A0 / ln(T)
Implements 4 core fixes for numerical stability, scaling, and initialization.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# Global Utility Constant (matches the paper's A0 = 100*pi)
A0 = 100 * math.pi


# -------------------------------------------------------------------
# FIX 1: Prevents Attention Softmax Collapse (Representational Collapse)
# Math: scale = 1 / (1 + A0 / ln(N)) where N = sequence length.
# -------------------------------------------------------------------
class NonCollapsingAttention(nn.Module):
    def __init__(self, d_k: int, A0: float = A0):
        super().__init__()
        self.d_k = d_k
        self.A0 = A0

    def forward(self, Q, K, V, N: int):
        if N <= 1:
            raise ValueError("Sequence length N must be > 1.")
        ln_N = math.log(N)
        W_N = 1.0 + self.A0 / ln_N  # Always > 1 for finite N

        logits = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * W_N)
        attn_weights = F.softmax(logits, dim=-1)
        return torch.matmul(attn_weights, V)


# -------------------------------------------------------------------
# FIX 2: Prevents LR Schedulers from Hitting Zero (Premature Stagnation)
# Math: eta(T) = eta_0 * (1/ln(T) + A0/(ln(T))^2)
# -------------------------------------------------------------------
class AsymptoticLRScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, A0: float = A0, last_epoch: int = -1):
        self.A0 = A0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T = max(self.last_epoch + 1, 2)  # Ensure T > 1
        ln_T = math.log(T)
        alpha_T = (1.0 / ln_T) + (self.A0 / (ln_T**2))
        return [base_lr * alpha_T for base_lr in self.base_lrs]


# -------------------------------------------------------------------
# FIX 3: Scale-Invariant Positional Embeddings (Aspect Ratio Distortion)
# Math: r(T) = sqrt((ln(T) + A0) / pi)
# -------------------------------------------------------------------
def get_scale_invariant_vit_coords(height: int, width: int, T: float, A0: float = A0):
    if T <= 1:
        raise ValueError("Input scale T must be > 1.")

    ln_T = math.log(T)
    r_T = math.sqrt((ln_T + A0) / math.pi)

    u = torch.linspace(-0.5, 0.5, width)
    v = torch.linspace(0.0, 1.0, height)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

    theta = 2 * math.pi * grid_u
    R = r_T * torch.sqrt(grid_v)

    x_norm = R * torch.cos(theta)
    y_norm = R * torch.sin(theta)

    return torch.stack([x_norm, y_norm], dim=-1)  # Shape: (H, W, 2)


# -------------------------------------------------------------------
# FIX 4: Corrected Weight Initialization (Fixes the 63% Volume Error)
# Math: Var(w) = pi / (ln(fan_in) + A0)
# -------------------------------------------------------------------
class CorrectedVarianceLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, A0: float = A0, bias: bool = True):
        self.A0 = A0
        super().__init__(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        T = self.in_features
        if T <= 1:
            raise ValueError("fan_in must be > 1.")

        ln_T = math.log(T)
        var_w = math.pi / (ln_T + self.A0)

        with torch.no_grad():
            self.weight.data.normal_(0, math.sqrt(var_w))
            if self.bias is not None:
                self.bias.data.zero_()


# Explicit export list
__all__ = [
    "NonCollapsingAttention",
    "AsymptoticLRScheduler",
    "get_scale_invariant_vit_coords",
    "CorrectedVarianceLinear",
    "A0",
]
