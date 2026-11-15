"""Experimental scaling utilities; not validated on real training runs."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class NonCollapsingAttention(nn.Module):
    def __init__(self, d_k: int, A_attn: float = 1.0):
        super().__init__()
        self.d_k = d_k
        self.A_attn = A_attn
        self.q_scale = nn.Parameter(torch.ones(1) * (d_k ** -0.25))
        self.k_scale = nn.Parameter(torch.ones(1) * (d_k ** -0.25))

    def forward(self, q, k, v):
        q_norm = F.normalize(q, dim=-1) * self.q_scale
        k_norm = F.normalize(k, dim=-1) * self.k_scale
        attn = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


class AsymptoticLRScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        floor_ratio: float = 0.01,
        total_steps: int = 300,
        last_epoch: int = -1,
    ):
        self.floor_ratio = floor_ratio
        self.total_steps = total_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step >= self.total_steps:
            return self.floor_ratio
        progress = float(step) / float(max(1, self.total_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.floor_ratio + (1.0 - self.floor_ratio) * cosine_decay


def get_scale_invariant_vit_coords(
    height: int, width: int, T: Optional[float] = None, A_coords: float = 0.0
):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    x_norm = (x.float() / width) - 0.5
    y_norm = (y.float() / height) - 0.5
    r = torch.sqrt(x_norm**2 + y_norm**2)
    theta = torch.atan2(y_norm, x_norm)
    return torch.stack([r, theta], dim=-1)


def make_corrected_linear(
    in_features: int, out_features: int, bias: bool = True, nonlinearity: str = "relu"
):
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.orthogonal_(layer.weight)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


__all__ = [
    "NonCollapsingAttention",
    "AsymptoticLRScheduler",
    "get_scale_invariant_vit_coords",
    "make_corrected_linear",
]
