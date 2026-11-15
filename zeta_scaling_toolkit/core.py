"""Experimental scaling utilities (v0.2.0).

Not validated on real training runs. Every component here is a standard
technique (QK-normalised attention, cosine-with-floor LR, isotropic coordinates,
gain-aware orthogonal init) with its failure modes closed off; see tests/.
"""

import math
import warnings
from typing import Optional, cast, Literal

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

    **Scale invariance:** This module is scale-invariant for ||q||, ||k||, ||v|| >= 1e-2.
    Below that threshold, gradient behaviour changes due to numerical regularization
    (eps=1e-6 in the L2 norm). This is a documented trade-off: keeping activations
    bounded at zero-vectors.

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
        # Input type validation
        if not isinstance(q, torch.Tensor):
            raise TypeError(f"q must be a Tensor, got {type(q).__name__}")
        if not isinstance(k, torch.Tensor):
            raise TypeError(f"k must be a Tensor, got {type(k).__name__}")
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"v must be a Tensor, got {type(v).__name__}")

        # Input rank validation
        if q.ndim < 2:
            raise ValueError(f"q must be at least 2D, got shape {q.shape}")
        if k.ndim < 2:
            raise ValueError(f"k must be at least 2D, got shape {k.shape}")
        if v.ndim < 2:
            raise ValueError(f"v must be at least 2D, got shape {v.shape}")

        # Dimension validation
        if q.shape[-1] != self.d_k or k.shape[-1] != self.d_k:
            raise ValueError(f"expected last dim d_k={self.d_k}, got q={q.shape[-1]}, k={k.shape[-1]}")
        # Batch dims must match; k and v must match
        if q.shape[:-2] != k.shape[:-2]:
            raise ValueError(f"q and k must have same batch dims, got q.shape[:-2]={q.shape[:-2]}, k.shape[:-2]={k.shape[:-2]}")
        if k.shape[:-2] != v.shape[:-2]:
            raise ValueError(f"k and v must have same batch dims, got k.shape[:-2]={k.shape[:-2]}, v.shape[:-2]={v.shape[:-2]}")
        if k.shape[-2] != v.shape[-2]:
            raise ValueError(f"k and v must have same sequence length, got k.shape[-2]={k.shape[-2]}, v.shape[-2]={v.shape[-2]}")

        # Validate key/query sequence lengths
        if k.shape[-2] == 0:
            raise ValueError(f"key sequence length must be >= 1, got {k.shape[-2]}")
        if q.shape[-2] == 0:
            raise ValueError(f"query sequence length must be >= 1, got {q.shape[-2]}")

        # Validate mask type if provided
        if attn_mask is not None:
            if not isinstance(attn_mask, torch.Tensor):
                raise TypeError(f"attn_mask must be a Tensor, got {type(attn_mask).__name__}")

        logits = (_unit(q.float()) @ _unit(k.float()).transpose(-2, -1)) * self.tau
        if attn_mask is not None:  # bool: True = keep; float: added to logits
            if attn_mask.dtype == torch.bool:
                logits = logits.masked_fill(~attn_mask, float("-inf"))
            else:
                logits = logits + attn_mask.float()
        weights = F.softmax(logits, dim=-1)
        # Fully masked rows (all NaN) -> replace with zeros (uniform attention impossible)
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        return (weights @ v.float()).to(v.dtype)

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

    Supported nonlinearities: linear, relu, leaky_relu, tanh, sigmoid, selu,
    and conv* (which all have stable gains via torch.nn.init.calculate_gain).

    For GELU and SiLU (which lack stable gain values), use calibrate_gain() after
    stacking layers, or consider using gain=1.0 (linear init) and tuning empirically.

    gain=sqrt(2) for ReLU keeps activation norms stable through deep ReLU stacks;
    pass nonlinearity="linear" for gain 1 (exact isometry on square layers).
    Rectangular layers with out < in necessarily shrink norms by ~sqrt(out/in).

    Args:
        in_features: input size.
        out_features: output size.
        bias: whether to add a bias term.
        nonlinearity: one of the supported nonlinearities.

    Raises:
        ValueError: if nonlinearity is not supported.
    """
    # Whitelist of nonlinearities supported by torch.nn.init.calculate_gain
    supported = {"linear", "relu", "leaky_relu", "tanh", "sigmoid", "selu", "conv1d", "conv2d", "conv3d"}
    if nonlinearity not in supported:
        raise ValueError(
            f"nonlinearity={nonlinearity!r} not supported. "
            f"Supported: {sorted(supported)}. "
            "For GELU/SiLU, use calibrate_gain(model, sample_input) after stacking."
        )
    gain = nn.init.calculate_gain(cast(Literal['linear', 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'selu', 'conv1d', 'conv2d', 'conv3d'], nonlinearity))
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.orthogonal_(layer.weight, gain=gain)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


def calibrate_gain(
    model_or_layer: nn.Module,
    sample_input: torch.Tensor,
    target_ratio: float = 1.0,
    iters: int = 5,
    nonlinearity: Optional[nn.Module] = None,
) -> float:
    """LSUV-style in-place gain calibration for a model/layer stack.

    Rescales all Linear layer weights in-place so that passing sample_input through
    the model yields an output with norm ratio (output_norm / input_norm) close to
    target_ratio. Useful for GELU/SiLU stacks where calculate_gain doesn't apply.

    No gradients are computed. Runs with torch.no_grad().

    Args:
        model_or_layer: nn.Module with Linear layers to calibrate.
        sample_input: (batch, *features) tensor to drive calibration. Must not require grad.
        target_ratio: desired output_norm / input_norm (default 1.0 for isometry).
        iters: number of refinement passes (default 5).
        nonlinearity: optional activation function to apply between layers (e.g., nn.GELU()).
                      If None, no nonlinearity is applied.

    Returns:
        Final achieved norm ratio after calibration.

    Example:
        model = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        x = torch.randn(32, 64)
        ratio = calibrate_gain(model, x, target_ratio=1.0)
        print(f"Norm ratio: {ratio}")
    """
    if sample_input.requires_grad:
        raise ValueError("sample_input must not require grad")

    with torch.no_grad():
        input_norm = sample_input.norm(dim=-1).mean().item()
        if input_norm == 0:
            raise ValueError("sample_input has zero norm")

        for _ in range(iters):
            out = sample_input.clone()
            for module in model_or_layer.modules():
                if isinstance(module, nn.Linear):
                    out = module(out)
                    if nonlinearity is not None:
                        out = nonlinearity(out)

            output_norm = out.norm(dim=-1).mean().item()
            if output_norm == 0:
                break

            ratio = output_norm / input_norm
            scale = (target_ratio / ratio) ** 0.5  # scale weights by sqrt of ratio

            # Rescale all Linear layers
            for module in model_or_layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.mul_(scale)

        # Final pass to compute achieved ratio
        out = sample_input.clone()
        for module in model_or_layer.modules():
            if isinstance(module, nn.Linear):
                out = module(out)
                if nonlinearity is not None:
                    out = nonlinearity(out)

        output_norm = out.norm(dim=-1).mean().item()
        final_ratio = output_norm / input_norm if input_norm > 0 else 0.0
        return final_ratio



__all__ = [
    "NonCollapsingAttention",
    "AsymptoticLRScheduler",
    "get_isotropic_coords",
    "get_scale_invariant_vit_coords",
    "make_corrected_linear",
    "calibrate_gain",
]
