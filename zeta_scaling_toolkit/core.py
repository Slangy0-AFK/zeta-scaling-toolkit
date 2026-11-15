"""Experimental scaling utilities; not validated on real training runs."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class NonCollapsingAttention(nn.Module):
    def __init__(self, d_k: int, A_attn: float = 1.0):
        super().__init__()
        self.d_k = d_k
        self.A_attn = A_attn

    def forward(self, Q, K, V):
        N = Q.shape[-2]
        if N <= 1:
            raise ValueError("Sequence length N must be > 1.")
        W_N = 1.0 + self.A_attn / math.log(N)
        logits = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * W_N)
        return torch.matmul(F.softmax(logits, dim=-1), V)


class AsymptoticLRScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, floor_ratio: float = 0.01, last_epoch: int = -1):
        self.floor_ratio = floor_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T = max(self.last_epoch + 1, 2)
        alpha = max(math.log(2) / math.log(T), self.floor_ratio)
        return [base_lr * alpha for base_lr in self.base_lrs]


def get_scale_invariant_vit_coords(height: int, width: int, T: float, A_coords: float = 0.0):
    if T <= 1:
        raise ValueError("Input scale T must be > 1.")
    r_T = math.sqrt((math.log(T) + A_coords) / math.pi)

    u = torch.linspace(-0.5, 0.5, width)
    v = torch.linspace(0.0, 1.0, height)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

    theta = 2 * math.pi * grid_u
    R = r_T * torch.sqrt(grid_v)
    x_norm = R * torch.cos(theta)
    y_norm = R * torch.sin(theta)
    return torch.stack([x_norm, y_norm], dim=-1)


def make_corrected_linear(
    in_features: int, out_features: int, bias: bool = True, nonlinearity: str = "relu"
):
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


__all__ = [
    "NonCollapsingAttention",
    "AsymptoticLRScheduler",
    "get_scale_invariant_vit_coords",
    "make_corrected_linear",
]
