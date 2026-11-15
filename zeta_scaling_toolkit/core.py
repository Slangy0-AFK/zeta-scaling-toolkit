"""Experimental PyTorch scaling utilities; not validated on large-scale training."""

import math
import numbers
import warnings
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# Added to the squared norm to keep zero-vector gradients bounded and inputs finite.
# The relative effect becomes small for vector norms around 1e-2 and larger.
_NORM_EPS = 1e-6
GainName = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]


def _unit(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise the last dim without overflowing on large finite values."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    scaled = x / scale
    epsilon = _NORM_EPS / scale / scale
    return scaled * torch.rsqrt(scaled.square().sum(dim=-1, keepdim=True) + epsilon)


def _broadcast_shape(*shapes: torch.Size) -> Optional[torch.Size]:
    try:
        return cast(torch.Size, torch.broadcast_shapes(*shapes))  # type: ignore[no-untyped-call]
    except RuntimeError:
        return None


class NonCollapsingAttention(nn.Module):
    """QK-normalised attention with a bounded, learnable temperature.

    logits = tau * cos(q, k), so |logit| <= tau <= tau_max. Unlike a fixed
    d_k**-0.5 bound, tau starts high enough for peaked attention and can never
    grow past ``tau_max``. Internals run in float32 (float16 safe); output is
    cast back to ``v.dtype``. Scale invariance is approximate near zero: because
    ``1e-6`` is added to the squared norm, it is reliable for vector norms of
    about ``1e-2`` and larger. Fully masked query rows return zero vectors.

    Args:
        d_k: per-head key dimension (inputs must match on the last dim).
        A_attn: deprecated, kept for API compatibility. Has no effect.
        tau0: initial temperature.
        tau_max: hard cap on the temperature.
        tau_min: lower bound on the temperature.
        check_finite: raise when q, k, or v contain NaN or infinity.
        length_scaling: optional logarithmic scaling by unmasked key count.
        train_length: training key count used as the log-scaling reference.
        tau_eff_max: upper bound on the scaled effective temperature.
    """

    def __init__(
        self,
        d_k: int,
        A_attn: float = 1.0,
        tau0: float = 10.0,
        tau_max: float = 50.0,
        tau_min: float = 1.0,
        check_finite: bool = False,
        length_scaling: Literal["none", "log"] = "none",
        train_length: Optional[int] = None,
        tau_eff_max: float = 200.0,
    ):
        super().__init__()
        if isinstance(d_k, bool) or not isinstance(d_k, int) or d_k < 1:
            raise ValueError(f"d_k must be >= 1, got {d_k}")
        values = {"tau0": tau0, "tau_min": tau_min, "tau_max": tau_max}
        if any(
            not isinstance(value, numbers.Real) or not math.isfinite(value)
            for value in values.values()
        ):
            raise ValueError(f"temperature values must be finite real numbers, got {values}")
        if not (0.0 <= tau_min <= tau0 <= tau_max) or tau0 == 0.0:
            raise ValueError(
                "need 0 <= tau_min <= tau0 <= tau_max and tau0 > 0, "
                f"got tau_min={tau_min}, tau0={tau0}, tau_max={tau_max}"
            )
        if tau_min >= tau_max:
            raise ValueError(
                f"tau_min must be strictly less than tau_max, "
                f"got tau_min={tau_min}, tau_max={tau_max}"
            )
        if not isinstance(check_finite, bool):
            raise TypeError("check_finite must be a bool")
        if length_scaling not in ("none", "log"):
            raise ValueError(f"length_scaling must be 'none' or 'log', got {length_scaling!r}")
        if train_length is not None and (
            isinstance(train_length, bool) or not isinstance(train_length, int) or train_length < 2
        ):
            raise ValueError(f"train_length must be an integer >= 2, got {train_length!r}")
        if length_scaling == "log" and train_length is None:
            raise ValueError("train_length is required when length_scaling='log'")
        if (
            isinstance(tau_eff_max, bool)
            or not isinstance(tau_eff_max, numbers.Real)
            or not math.isfinite(tau_eff_max)
            or tau_eff_max <= 0
        ):
            raise ValueError(f"tau_eff_max must be positive and finite, got {tau_eff_max}")
        if A_attn != 1.0:
            warnings.warn(
                "A_attn is deprecated and has no effect.", DeprecationWarning, stacklevel=2
            )
        self.d_k: int = d_k
        self.A_attn = A_attn  # unused; retained so old benchmark scripts still assign it
        self.tau_max: float = float(tau_max)
        self.tau_min: float = float(tau_min)
        self.check_finite: bool = check_finite
        self.length_scaling: Literal["none", "log"] = length_scaling
        self.train_length: Optional[int] = train_length
        self.tau_eff_max: float = float(tau_eff_max)
        scaled_tau0 = (tau0 - tau_min) / (tau_max - tau_min)
        clipped_scaled = max(1e-7, min(1.0 - 1e-7, scaled_tau0))
        logit_tau0 = math.log(clipped_scaled / (1.0 - clipped_scaled))
        self.raw_tau: nn.Parameter = nn.Parameter(torch.tensor(logit_tau0))

    @property
    def tau(self) -> torch.Tensor:
        scaled = torch.sigmoid(self.raw_tau)
        return self.tau_min + scaled * (self.tau_max - self.tau_min)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        visibility_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend with an optional additive/bool mask and explicit boolean visibility.

        ``visibility_mask`` is broadcastable to the logits and intersects
        ``attn_mask``. Use it with finite additive biases or finite padding
        sentinels when log-length scaling must count only visible keys. Without
        it, every finite additive mask value is treated as visible.
        """
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.ndim < 2:
                raise ValueError(f"{name} must have rank >= 2, got rank {tensor.ndim}")
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
        if q.shape[-1] != self.d_k or k.shape[-1] != self.d_k:
            raise ValueError(
                f"expected last dim d_k={self.d_k}, got q={q.shape[-1]}, k={k.shape[-1]}"
            )
        if k.shape[-2] == 0:
            raise ValueError("key sequence length must be nonzero")
        if v.shape[-2] != k.shape[-2]:
            raise ValueError("key and value sequence lengths must match")
        if q.device != k.device or q.device != v.device:
            raise ValueError("q, k, and v must be on the same device")
        if self.raw_tau.device != q.device:
            raise ValueError(
                "attention parameters and inputs must be on the same device; "
                "move the module with attention.to(q.device)"
            )
        if _broadcast_shape(q.shape[:-2], k.shape[:-2], v.shape[:-2]) is None:
            raise ValueError("q, k, and v batch dimensions must be broadcastable")
        if self.check_finite and not all(torch.isfinite(tensor).all() for tensor in (q, k, v)):
            raise ValueError("q, k, and v must contain only finite values")

        dtype = (
            torch.float64
            if any(tensor.dtype == torch.float64 for tensor in (q, k, v))
            else torch.float32
        )
        batch_shape = _broadcast_shape(q.shape[:-2], k.shape[:-2], v.shape[:-2])
        assert batch_shape is not None
        query_length, key_length = q.shape[-2], k.shape[-2]
        value_length = v.shape[-1]
        q_unit = _unit(q.to(dtype)).expand(*batch_shape, query_length, self.d_k)
        k_unit = _unit(k.to(dtype)).expand(*batch_shape, key_length, self.d_k)
        values = v.to(dtype).expand(*batch_shape, key_length, value_length)
        logits_shape = (*batch_shape, query_length, key_length)
        expanded_mask: Optional[torch.Tensor] = None
        if attn_mask is not None:
            if not isinstance(attn_mask, torch.Tensor):
                raise TypeError("attn_mask must be a torch.Tensor")
            if attn_mask.device != q.device:
                raise ValueError("attn_mask must be on the same device as q, k, and v")
            if attn_mask.dtype != torch.bool and not attn_mask.is_floating_point():
                raise TypeError("attn_mask must have a bool or floating-point dtype")
            mask_shape = _broadcast_shape(attn_mask.shape, torch.Size(logits_shape))
            if mask_shape is None or tuple(mask_shape) != tuple(logits_shape):
                raise ValueError("attn_mask must be broadcastable to the attention logits")
            if attn_mask.dtype != torch.bool and (
                torch.isnan(attn_mask).any() or torch.isposinf(attn_mask).any()
            ):
                raise ValueError("floating attn_mask values must not contain NaN or +inf")
            expanded_mask = attn_mask.expand(logits_shape)
        if visibility_mask is not None:
            if not isinstance(visibility_mask, torch.Tensor) or visibility_mask.dtype != torch.bool:
                raise TypeError("visibility_mask must be a boolean torch.Tensor")
            if visibility_mask.device != q.device:
                raise ValueError("visibility_mask must be on the same device as q, k, and v")
            visibility_shape = _broadcast_shape(visibility_mask.shape, torch.Size(logits_shape))
            if visibility_shape is None or tuple(visibility_shape) != tuple(logits_shape):
                raise ValueError("visibility_mask must be broadcastable to the attention logits")
            visible = visibility_mask.expand(logits_shape)
            if expanded_mask is None:
                expanded_mask = visible
            elif expanded_mask.dtype == torch.bool:
                expanded_mask = expanded_mask & visible
            else:
                expanded_mask = expanded_mask.masked_fill(~visible, float("-inf"))
        if query_length == 0 or any(dimension == 0 for dimension in batch_shape):
            logits = (q_unit @ k_unit.transpose(-2, -1)) * self.tau.to(dtype)
            return (logits @ values).to(v.dtype)

        counts: Union[int, torch.Tensor] = key_length
        if expanded_mask is not None:
            if expanded_mask.dtype == torch.bool:
                counts = expanded_mask.sum(dim=-1, keepdim=True)
            else:
                counts = torch.isfinite(expanded_mask).sum(dim=-1, keepdim=True)

        scale_per_row = isinstance(counts, torch.Tensor) and counts.numel() > 0
        if self.length_scaling == "log":
            assert self.train_length is not None
            if isinstance(counts, torch.Tensor):
                scale_per_row = scale_per_row and bool((counts != counts.reshape(-1)[0]).any())
                length_factor: Union[float, torch.Tensor] = (
                    counts.clamp_min(1).to(dtype).log() / math.log(self.train_length)
                ).clamp_min(1.0)
            else:
                length_factor = max(1.0, math.log(counts) / math.log(self.train_length))
            tau_eff: Union[float, torch.Tensor] = (self.tau.to(dtype) * length_factor).clamp(
                max=self.tau_eff_max
            )
        else:
            tau_eff = self.tau.to(dtype)
            scale_per_row = False

        has_empty_rows = isinstance(counts, torch.Tensor) and bool((counts == 0).any())
        if not scale_per_row and not has_empty_rows:
            if isinstance(tau_eff, torch.Tensor):
                scale = tau_eff.reshape(-1)[0]
            else:
                scale = torch.tensor(tau_eff, dtype=dtype, device=q.device)
            scaled_q = q_unit * (scale * math.sqrt(self.d_k))
            sdpa_mask = expanded_mask
            if sdpa_mask is not None and sdpa_mask.dtype != torch.bool:
                sdpa_mask = sdpa_mask.to(dtype)
            output = F.scaled_dot_product_attention(
                scaled_q,
                k_unit,
                values,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
            )
            return output.to(v.dtype)

        if expanded_mask is not None and expanded_mask.dtype != torch.bool:
            expanded_mask = expanded_mask.to(dtype)
        outputs = []
        for start in range(0, query_length, 128):
            end = min(start + 128, query_length)
            logits = q_unit[..., start:end, :] @ k_unit.transpose(-2, -1)
            if isinstance(tau_eff, torch.Tensor) and tau_eff.ndim > 0:
                row_scale = tau_eff[..., start:end, :]
                logits = logits * row_scale
            else:
                logits = logits * tau_eff
            mask_chunk = expanded_mask[..., start:end, :] if expanded_mask is not None else None
            if mask_chunk is not None:
                if mask_chunk.dtype == torch.bool:
                    logits = logits.masked_fill(~mask_chunk, float("-inf"))
                else:
                    logits = logits + mask_chunk
            row_has_keys = ~torch.isneginf(logits).all(dim=-1, keepdim=True)
            safe_logits = torch.where(row_has_keys, logits, torch.zeros_like(logits))
            weights = F.softmax(safe_logits, dim=-1)
            weights = torch.where(row_has_keys, weights, torch.zeros_like(weights))
            outputs.append(weights @ values)
        output = torch.cat(outputs, dim=-2)
        return output.to(v.dtype)

    def forward_streaming(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window: int,
        sink_tokens: int = 0,
        *,
        alibi: bool = False,
        chunk_size: int = 128,
        head_axis: Optional[int] = None,
    ) -> torch.Tensor:
        """Attend exactly to sink keys and each query's recent causal window.

        Unlike a dense ``streaming_mask`` plus dense ALiBi bias, this method
        selects only the union of sink keys and recent keys for each query
        chunk. Query/key tensors use the same shapes as :meth:`forward`; for
        rank-4 inputs the third-from-last dimension is interpreted as heads
        when ``alibi=True``. If singleton head dimensions are broadcast, ALiBi
        uses the effective head count from the broadcast batch shape. Query
        positions are aligned to the final ``q_len`` key positions, matching
        ``streaming_mask`` and ``alibi_bias``.

        Set ``head_axis`` to the head dimension's index in the broadcast batch
        shape (excluding sequence and feature dimensions) for other layouts.
        Without it, fewer than two batch dimensions are treated as headless.

        The result is exact (no approximation). During the forward pass, the
        temporary attention work for each query chunk is bounded by
        ``chunk_size * (sink_tokens + window + chunk_size)`` rather than the
        full Q-by-K matrix. With autograd enabled, saved tensors from every
        chunk remain live until backward, so training activation memory grows
        with query length, roughly as
        ``q_len * (sink_tokens + window + chunk_size)``, in addition to
        full-sequence normalized inputs and output storage. This is not a
        sequence-length-independent training-memory bound. Gradients are
        supported through Q/K/V and the learnable temperature.
        """
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.ndim < 2:
                raise ValueError(f"{name} must have rank >= 2, got rank {tensor.ndim}")
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
        if q.shape[-1] != self.d_k or k.shape[-1] != self.d_k:
            raise ValueError(
                f"expected last dim d_k={self.d_k}, got q={q.shape[-1]}, k={k.shape[-1]}"
            )
        if k.shape[-2] == 0:
            raise ValueError("key sequence length must be nonzero")
        if q.shape[-2] > k.shape[-2]:
            raise ValueError("query sequence length must be <= key sequence length")
        if v.shape[-2] != k.shape[-2]:
            raise ValueError("key and value sequence lengths must match")
        if q.device != k.device or q.device != v.device:
            raise ValueError("q, k, and v must be on the same device")
        if self.raw_tau.device != q.device:
            raise ValueError(
                "attention parameters and inputs must be on the same device; "
                "move the module with attention.to(q.device)"
            )
        if _broadcast_shape(q.shape[:-2], k.shape[:-2], v.shape[:-2]) is None:
            raise ValueError("q, k, and v batch dimensions must be broadcastable")
        if self.check_finite and not all(torch.isfinite(tensor).all() for tensor in (q, k, v)):
            raise ValueError("q, k, and v must contain only finite values")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be an integer >= 1")
        if isinstance(sink_tokens, bool) or not isinstance(sink_tokens, int) or sink_tokens < 0:
            raise ValueError("sink_tokens must be a nonnegative integer")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be an integer >= 1")
        if not isinstance(alibi, bool):
            raise TypeError("alibi must be a bool")

        batch_shape = _broadcast_shape(q.shape[:-2], k.shape[:-2], v.shape[:-2])
        assert batch_shape is not None
        if head_axis is not None:
            if isinstance(head_axis, bool) or not isinstance(head_axis, int):
                raise TypeError("head_axis must be an integer")
            if not alibi:
                raise ValueError("head_axis requires alibi=True")
            if not -len(batch_shape) <= head_axis < len(batch_shape):
                raise ValueError("head_axis must index a broadcast batch dimension")
            head_axis %= len(batch_shape)
        elif len(batch_shape) >= 2:
            head_axis = len(batch_shape) - 1
        query_length, key_length = q.shape[-2], k.shape[-2]
        value_length = v.shape[-1]
        if query_length == 0 or any(dimension == 0 for dimension in batch_shape):
            zero = q.sum() * 0 + k.sum() * 0 + v.sum() * 0 + self.raw_tau.to(q.dtype) * 0
            return zero.expand(*batch_shape, query_length, value_length).to(v.dtype)
        dtype = (
            torch.float64
            if any(tensor.dtype == torch.float64 for tensor in (q, k, v))
            else torch.float32
        )
        q_unit = _unit(q.to(dtype)).expand(*batch_shape, query_length, self.d_k)
        k_unit = _unit(k.to(dtype)).expand(*batch_shape, key_length, self.d_k)
        values = v.to(dtype).expand(*batch_shape, key_length, value_length)
        query_offset = key_length - query_length
        sink_end = min(sink_tokens, key_length)
        all_sink_positions = torch.arange(sink_end, device=q.device)

        if alibi:
            head_count = batch_shape[head_axis] if head_axis is not None else 1
            head_ids = torch.arange(1, head_count + 1, dtype=dtype, device=q.device)
            slopes = torch.exp2(-8.0 * head_ids / head_count)
        else:
            slopes = None

        outputs: list[torch.Tensor] = []
        for start in range(0, query_length, chunk_size):
            end = min(start + chunk_size, query_length)
            first_query_position = query_offset + start
            final_query_position = query_offset + end
            recent_start = max(sink_end, first_query_position - window + 1)
            recent_end = max(recent_start, final_query_position)
            recent_positions = torch.arange(recent_start, recent_end, device=q.device)
            if sink_end:
                key_positions = torch.cat((all_sink_positions, recent_positions))
            else:
                key_positions = recent_positions

            query_positions = torch.arange(
                first_query_position, final_query_position, device=q.device
            )
            keep = (key_positions[None, :] <= query_positions[:, None]) & (
                (key_positions[None, :] < sink_tokens)
                | (key_positions[None, :] >= query_positions[:, None] - window + 1)
            )
            counts = keep.sum(dim=-1).clamp_min(1)
            q_chunk = q_unit[..., start:end, :]
            k_chunk = k_unit.index_select(-2, key_positions)
            v_chunk = values.index_select(-2, key_positions)

            if self.length_scaling == "log":
                assert self.train_length is not None
                length_factor = (counts.to(dtype).log() / math.log(self.train_length)).clamp_min(
                    1.0
                )
                row_scale = (self.tau.to(dtype) * length_factor).clamp(max=self.tau_eff_max)
                logits = q_chunk @ k_chunk.transpose(-2, -1)
                logits = logits * row_scale.reshape(*([1] * len(batch_shape)), end - start, 1)
                if alibi:
                    assert slopes is not None
                    distances = (query_positions[:, None] - key_positions[None, :]).abs().to(dtype)
                    if head_axis is not None:
                        slope_shape = [1] * (len(batch_shape) + 2)
                        slope_shape[head_axis] = head_count
                        slope_values: Union[torch.Tensor, float] = slopes.reshape(*slope_shape)
                    else:
                        slope_values = slopes[0]
                    bias = -slope_values * distances
                    logits = logits + bias
                logits = logits.masked_fill(~keep, float("-inf"))
                weights = F.softmax(logits, dim=-1)
                output_chunk = weights @ v_chunk
            else:
                scaled_query = q_chunk * (self.tau.to(dtype) * math.sqrt(self.d_k))
                if alibi:
                    assert slopes is not None
                    distances = (query_positions[:, None] - key_positions[None, :]).abs().to(dtype)
                    if head_axis is not None:
                        slope_shape = [1] * (len(batch_shape) + 2)
                        slope_shape[head_axis] = head_count
                        slope_values = slopes.reshape(*slope_shape)
                    else:
                        slope_values = slopes[0]
                    bias = -slope_values * distances
                    bias = bias.masked_fill(~keep, float("-inf"))
                    attention_mask = bias
                else:
                    attention_mask = keep
                output_chunk = F.scaled_dot_product_attention(
                    scaled_query,
                    k_chunk,
                    v_chunk,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                )
            outputs.append(output_chunk)
        return torch.cat(outputs, dim=-2).to(v.dtype)

    def _load_from_state_dict(
        self,
        state_dict: Any,
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: Any,
        unexpected_keys: Any,
        error_msgs: Any,
    ) -> None:
        legacy = [prefix + "q_scale", prefix + "k_scale"]
        if any(key in state_dict for key in legacy):
            warnings.warn(
                "Dropping legacy q_scale/k_scale from checkpoint; raw_tau keeps its init.",
                UserWarning,
                stacklevel=2,
            )
            state_dict = state_dict.copy()
            for key in legacy:
                state_dict.pop(key, None)
            state_dict.setdefault(prefix + "raw_tau", self.raw_tau.detach().clone())
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class AsymptoticLRScheduler(LambdaLR):
    """Linear warmup, then cosine decay to ``floor_ratio * base_lr``, then a hard clamp.

    This schedule reaches a fixed minimum LR; it does not decay indefinitely.
    Call ``step()`` after each optimizer update.
    For ``last_epoch >= 0``, restore the optimizer's original ``initial_lr``
    values first; loading both optimizer and scheduler state dicts is preferred.

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
        if (
            isinstance(floor_ratio, bool)
            or not isinstance(floor_ratio, numbers.Real)
            or not math.isfinite(floor_ratio)
        ):
            raise ValueError(f"floor_ratio must be finite, got {floor_ratio}")
        if not (0.0 <= floor_ratio <= 1.0):
            raise ValueError(f"floor_ratio must be in [0, 1], got {floor_ratio}")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if (
            isinstance(warmup_steps, bool)
            or not isinstance(warmup_steps, int)
            or not (0 <= warmup_steps < total_steps)
        ):
            raise ValueError(f"need 0 <= warmup_steps < total_steps, got {warmup_steps}")
        if last_epoch >= 0 and any("initial_lr" not in group for group in optimizer.param_groups):
            raise KeyError(
                "resuming with last_epoch >= 0 requires initial_lr in every optimizer "
                "param group; restore optimizer state or supply the original base learning rates"
            )
        self.floor_ratio: float = float(floor_ratio)
        self.total_steps: int = total_steps
        self.warmup_steps: int = warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            return self.floor_ratio
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.floor_ratio + (1.0 - self.floor_ratio) * cosine


def get_isotropic_coords(
    height: int,
    width: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Pixel-centred (x, y) coords, centred on the image, divided by max(h, w).

    Equal pixel distances give equal coordinate distances in x and y, so aspect
    ratio is preserved. Shape (height, width, 2).
    """
    if (
        isinstance(height, bool)
        or not isinstance(height, int)
        or isinstance(width, bool)
        or not isinstance(width, int)
        or height < 1
        or width < 1
    ):
        raise ValueError(f"height and width must be >= 1, got {height}x{width}")
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating-point torch.dtype")
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
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Polar coords on the isotropic grid.

    Returns (r, theta) with shape (h, w, 2), or (r, cos theta, sin theta) with
    shape (h, w, 3) if ``smooth_angle`` (no jump at +-pi). Radius <= sqrt(2)/2.
    ``T`` and ``A_coords`` are deprecated and ignored.
    """
    if T is not None or A_coords != 0.0:
        warnings.warn(
            "T and A_coords are deprecated and have no effect.", DeprecationWarning, stacklevel=2
        )
    xy = get_isotropic_coords(height, width, dtype=dtype, device=device)
    x, y = xy[..., 0], xy[..., 1]
    r = torch.hypot(x, y)
    theta = torch.atan2(y, x)
    if smooth_angle:
        return torch.stack([r, torch.cos(theta), torch.sin(theta)], dim=-1)
    return torch.stack([r, theta], dim=-1)


def make_corrected_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    nonlinearity: GainName = "relu",
) -> nn.Linear:
    """Linear layer with orthogonal weights scaled by the nonlinearity's gain.

    gain=sqrt(2) for ReLU keeps activation norms stable through deep ReLU stacks;
    pass nonlinearity="linear" for gain 1 (exact isometry on square layers).
    Rectangular layers with out < in necessarily shrink norms by ~sqrt(out/in).
    """
    supported = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "sigmoid",
        "tanh",
        "relu",
        "leaky_relu",
        "selu",
    }
    if not isinstance(nonlinearity, str) or nonlinearity not in supported:
        raise ValueError(
            f"unsupported nonlinearity {nonlinearity!r}; GELU and SiLU require "
            "data-dependent calibrate_gain(...)"
        )
    if not isinstance(bias, bool):
        raise TypeError("bias must be a bool")
    gain = nn.init.calculate_gain(nonlinearity)
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.orthogonal_(layer.weight, gain=gain)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


def calibrate_gain(
    model_or_layer_stack: Union[nn.Module, Sequence[nn.Module]],
    sample_input: torch.Tensor,
    target_ratio: float = 1.0,
    iters: int = 10,
) -> Union[nn.Module, Sequence[nn.Module]]:
    """Calibrate a sequential layer stack's activation norm using sample data.

    The stack must be an ``nn.Sequential``, ``nn.ModuleList``, or a sequence of
    modules. Each layer's whole-batch output norm is adjusted toward
    ``target_ratio`` times its whole-batch input norm, not each sample's norm.
    Measure per-sample ratios separately, especially for deep smooth-activation
    stacks. Calibration runs without gradients, preserves module
    training flags and CPU/CUDA RNG state, and mutates linear layer weights.

    Calibration is transactional: if any validation fails, no parameters are modified.
    A warning on failure to converge leaves the best-effort parameters in place.
    """
    if not isinstance(sample_input, torch.Tensor) or not sample_input.is_floating_point():
        raise TypeError("sample_input must be a floating-point torch.Tensor")
    if (
        not isinstance(target_ratio, numbers.Real)
        or not math.isfinite(target_ratio)
        or target_ratio <= 0
    ):
        raise ValueError("target_ratio must be a finite positive real number")
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise ValueError("iters must be a positive integer")
    if isinstance(model_or_layer_stack, (nn.Sequential, nn.ModuleList)):
        layers = list(model_or_layer_stack)
    elif isinstance(model_or_layer_stack, Sequence) and all(
        isinstance(layer, nn.Module) for layer in model_or_layer_stack
    ):
        layers = list(model_or_layer_stack)
    else:
        raise TypeError(
            "model_or_layer_stack must be nn.Sequential, nn.ModuleList, or a module sequence"
        )
    if not layers:
        raise ValueError("model_or_layer_stack must not be empty")
    if not any(isinstance(layer, nn.Linear) for layer in layers):
        raise ValueError("model_or_layer_stack must contain a direct nn.Linear layer")

    modules = [module for layer in layers for module in layer.modules()]
    training_flags = [module.training for module in modules]
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    linear_modules = [layer for layer in layers if isinstance(layer, nn.Linear)]
    parameter_snapshots = [
        (layer, layer.weight.detach().clone(), layer.bias.detach().clone() if layer.bias is not None else None)
        for layer in linear_modules
    ]

    activation_types = (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid)
    worst_deviation = 0.0
    try:
        for layer in layers:
            layer.eval()
        with torch.no_grad():
            for _ in range(iters):
                worst_deviation = 0.0
                values = sample_input
                index = 0
                while index < len(layers):
                    layer = layers[index]
                    layer_input = values
                    if not isinstance(layer, nn.Linear):
                        values = layer(values)
                        index += 1
                        continue

                    values = layer(values)
                    following_activation = (
                        layers[index + 1]
                        if index + 1 < len(layers)
                        and isinstance(layers[index + 1], activation_types)
                        else None
                    )
                    measured = (
                        following_activation(values) if following_activation is not None else values
                    )
                    input_norm = torch.linalg.vector_norm(layer_input.double())
                    output_norm = torch.linalg.vector_norm(measured.double())
                    if input_norm == 0 or output_norm == 0 or not torch.isfinite(output_norm):
                        raise ValueError(
                            "cannot calibrate a layer with zero or non-finite activation norm"
                        )
                    factor = float(target_ratio) * input_norm / output_norm
                    if not torch.isfinite(factor) or factor <= 0:
                        raise ValueError("calibration produced a non-finite or non-positive scale")
                    scaled_weight = layer.weight * factor.to(layer.weight.dtype)
                    if not torch.isfinite(scaled_weight).all():
                        raise ValueError("calibration would produce non-finite layer weights")
                    scaled_bias = None
                    if layer.bias is not None:
                        scaled_bias = layer.bias * factor.to(layer.bias.dtype)
                        if not torch.isfinite(scaled_bias).all():
                            raise ValueError("calibration would produce non-finite layer bias")
                    layer.weight.copy_(scaled_weight)
                    if layer.bias is not None and scaled_bias is not None:
                        layer.bias.copy_(scaled_bias)
                    values = layer(layer_input)
                    if following_activation is not None:
                        values = following_activation(values)
                        index += 1
                    achieved = torch.linalg.vector_norm(values.double()) / input_norm
                    worst_deviation = max(
                        worst_deviation, abs(float(achieved) / float(target_ratio) - 1.0)
                    )
                    index += 1
    except Exception:
        with torch.no_grad():
            for layer, orig_weight, orig_bias in parameter_snapshots:
                layer.weight.copy_(orig_weight)
                if orig_bias is not None and layer.bias is not None:
                    layer.bias.copy_(orig_bias)
        raise
    finally:
        for module, training in zip(modules, training_flags):
            module.training = training
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    if worst_deviation > 0.02:
        warnings.warn(
            f"calibrate_gain did not reach target_ratio={target_ratio} within 2% "
            f"(worst layer off by {worst_deviation:.1%}); the target may be unreachable "
            "for this activation and input scale.",
            RuntimeWarning,
            stacklevel=2,
        )
    return model_or_layer_stack


__all__ = [
    "NonCollapsingAttention",
    "AsymptoticLRScheduler",
    "get_isotropic_coords",
    "get_scale_invariant_vit_coords",
    "make_corrected_linear",
    "calibrate_gain",
]
