"""Experimental scaling utilities (v0.2.0).

Not validated on real training runs. Every component here is a standard
technique (QK-normalised attention, cosine-with-floor LR, isotropic coordinates,
gain-aware orthogonal init) with its failure modes closed off; see tests/.
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# Added to the squared norm: bounds the gradient at zero vectors by 1/sqrt(eps) (~1e3) and keeps
# float16 finite. Trade-off: scale invariance is exact only for ||x|| >~ 1e-2.
_NORM_EPS = 1e-6


def _unit(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise the last dim; finite value and bounded gradient at x == 0."""
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + _NORM_EPS)


class NonCollapsingAttention(nn.Module):
    """QK-normalised attention with a bounded, learnable temperature.

    logits = tau * cos(q, k), so tau_min <= tau <= tau_max. Unlike a fixed
    d_k**-0.5 bound, tau starts high enough for peaked attention and stays
    within [tau_min, tau_max]. Internals run in float32 (float16 safe); output
    is cast back to ``v.dtype``.

    Args:
        d_k: per-head key dimension (inputs must match on the last dim).
        A_attn: deprecated, kept for API compatibility. Has no effect.
        tau0: initial temperature.
        tau_min: hard floor on the temperature (default 1.0).
        tau_max: hard cap on the temperature.
    """

    def __init__(
        self, d_k: int, A_attn: float = 1.0, tau0: float = 10.0, tau_min: float = 1.0, tau_max: float = 50.0
    ):
        super().__init__()
        if d_k < 1:
            raise ValueError(f"d_k must be >= 1, got {d_k}")
        if not (0.0 < tau_min <= tau0 <= tau_max):
            raise ValueError(f"need 0 < tau_min <= tau0 <= tau_max, got tau_min={tau_min}, tau0={tau0}, tau_max={tau_max}")
        if A_attn != 1.0:
            warnings.warn("A_attn is deprecated and has no effect.", DeprecationWarning, stacklevel=2)
        self.d_k = d_k
        self.A_attn = A_attn  # unused; retained so old benchmark scripts still assign it
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.raw_tau = nn.Parameter(torch.tensor(math.log(math.expm1(tau0))))  # inverse softplus

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.raw_tau).clamp(min=self.tau_min, max=self.tau_max)

    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor] = None):
        if q.shape[-1] != self.d_k or k.shape[-1] != self.d_k:
            raise ValueError(f"expected last dim d_k={self.d_k}, got q={q.shape[-1]}, k={k.shape[-1]}")
        logits = (_unit(q.float()) @ _unit(k.float()).transpose(-2, -1)) * self.tau
        if attn_mask is not None:  # bool: True = keep; float: added to logits
            if attn_mask.dtype == torch.bool:
                logits = logits.masked_fill(~attn_mask, float("-inf"))  # fully masked row -> NaN
            else:
                logits = logits + attn_mask.float()
        return (F.softmax(logits, dim=-1) @ v.float()).to(v.dtype)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        legacy = [prefix + "q_scale", prefix + "k_scale"]
        if any(key in state_dict for key in legacy):
            warnings.warn("Dropping legacy q_scale/k_scale from checkpoint; raw_tau keeps its init.")
            for key in legacy:
                state_dict.pop(key, None)
            state_dict.setdefault(prefix + "raw_tau", self.raw_tau.detach().clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class AsymptoticLRScheduler(LambdaLR):
    """Linear warmup, then cosine decay to ``floor_ratio * base_lr``, then a hard clamp.

    This is cosine annealing with a minimum LR (not a true asymptote). Call
    ``step()`` after each optimizer update.

    Args:
        floor_ratio: final LR as a fraction of base LR, in [0, 1].
        total_steps: step at which the floor is reached (>= 1).
        warmup_steps: linear warmup length (default 0 = none). Must be < total_steps.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        floor_ratio: float = 0.01,
        total_steps: int = 300,
        last_epoch: int = -1,
        warmup_steps: int = 0,
    ):
        if not (0.0 <= floor_ratio <= 1.0):
            raise ValueError(f"floor_ratio must be in [0, 1], got {floor_ratio}")
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if not (0 <= warmup_steps < total_steps):
            raise ValueError(f"need 0 <= warmup_steps < total_steps, got {warmup_steps}")
        self.floor_ratio = floor_ratio
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            return self.floor_ratio
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.floor_ratio + (1.0 - self.floor_ratio) * cosine


def get_isotropic_coords(height: int, width: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """Pixel-centred (x, y) coords, centred on the image, divided by max(h, w).

    Equal pixel distances give equal coordinate distances in x and y, so aspect
    ratio is preserved. Shape (height, width, 2).
    """
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be >= 1, got {height}x{width}")
    scale = float(max(height, width))
    ys = (torch.arange(height, dtype=dtype, device=device) + 0.5 - height / 2) / scale
    xs = (torch.arange(width, dtype=dtype, device=device) + 0.5 - width / 2) / scale
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([x, y], dim=-1)


def get_scale_invariant_vit_coords(
    height: int,
    width: int,
    T: Optional[float] = None,
    A_coords: float = 0.0,
    smooth_angle: bool = False,
    dtype=torch.float32,
    device=None,
) -> torch.Tensor:
    """Polar coords on the isotropic grid.

    Returns (r, theta) with shape (h, w, 2), or (r, cos theta, sin theta) with
    shape (h, w, 3) if ``smooth_angle`` (no jump at +-pi). Radius <= sqrt(2)/2.
    ``T`` and ``A_coords`` are deprecated and ignored.
    """
    if T is not None or A_coords != 0.0:
        warnings.warn("T and A_coords are deprecated and have no effect.", DeprecationWarning, stacklevel=2)
    xy = get_isotropic_coords(height, width, dtype=dtype, device=device)
    x, y = xy[..., 0], xy[..., 1]
    r = torch.hypot(x, y)
    theta = torch.atan2(y, x)
    if smooth_angle:
        return torch.stack([r, torch.cos(theta), torch.sin(theta)], dim=-1)
    return torch.stack([r, theta], dim=-1)


def make_corrected_linear(
    in_features: int, out_features: int, bias: bool = True, nonlinearity: str = "relu"
) -> nn.Linear:
    """Linear layer with orthogonal weights scaled by the nonlinearity's gain.

    gain=sqrt(2) for ReLU keeps activation norms stable through deep ReLU stacks;
    pass nonlinearity="linear" for gain 1 (exact isometry on square layers).
    Rectangular layers with out < in necessarily shrink norms by ~sqrt(out/in).
    """
    gain = nn.init.calculate_gain(nonlinearity)
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.orthogonal_(layer.weight, gain=gain)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


__all__ = [
    "NonCollapsingAttention",
    "AsymptoticLRScheduler",
    "get_isotropic_coords",
    "get_scale_invariant_vit_coords",
    "make_corrected_linear",
]
