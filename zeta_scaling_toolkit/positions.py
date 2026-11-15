"""Position encodings and masks for sequence-length extrapolation."""

import math
import numbers
from typing import Literal, Optional, Union

import torch


def rotary_embedding(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
    scaling: Optional[Literal["linear", "ntk"]] = None,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Apply interleaved rotary position encoding along the final even dimension.

    Position angles are computed from int64 positions in float64, then the
    rotated tensor is cast back to the input dtype. ``linear`` divides positions
    by ``scale_factor``. ``ntk`` scales the frequency base using the common NTK
    interpolation rule.
    """
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating-point torch.Tensor")
    if x.ndim < 2 or x.shape[-1] < 2 or x.shape[-1] % 2:
        raise ValueError("x must have rank >= 2 and an even last dimension >= 2")
    if not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64:
        raise TypeError("positions must be an int64 torch.Tensor")
    if positions.device != x.device:
        raise ValueError("positions and x must be on the same device")
    try:
        shape = torch.broadcast_shapes(positions.shape, x.shape[:-1])  # type: ignore[no-untyped-call]
    except RuntimeError as error:
        raise ValueError("positions must be broadcastable to x.shape[:-1]") from error
    if shape != x.shape[:-1]:
        raise ValueError("positions must be broadcastable to x.shape[:-1]")
    if not isinstance(base, numbers.Real) or not math.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than 1")
    if scaling not in (None, "linear", "ntk"):
        raise ValueError("scaling must be None, 'linear', or 'ntk'")
    if (
        isinstance(scale_factor, bool)
        or not isinstance(scale_factor, numbers.Real)
        or not math.isfinite(scale_factor)
        or scale_factor < 1
    ):
        raise ValueError("scale_factor must be a finite real number >= 1")

    dimension = x.shape[-1]
    effective_base = float(base)
    position_values = positions.to(torch.float64)
    if scaling == "linear":
        position_values = position_values / scale_factor
    elif scaling == "ntk" and scale_factor > 1:
        exponent = dimension / (dimension - 2) if dimension > 2 else 1.0
        effective_base *= scale_factor**exponent

    frequencies = torch.arange(0, dimension, 2, device=x.device, dtype=torch.float64)
    inverse_frequencies = effective_base ** (-frequencies / dimension)
    angles = position_values[..., None] * inverse_frequencies
    cosine, sine = torch.cos(angles), torch.sin(angles)
    pairs = x.to(torch.float64).reshape(*x.shape[:-1], dimension // 2, 2)
    even, odd = pairs.unbind(dim=-1)
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def alibi_bias(
    num_heads: int,
    q_len: int,
    k_len: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Return negative absolute-distance ALiBi biases with shape (1, heads, Q, K)."""
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
        raise ValueError("num_heads must be an integer >= 1")
    if any(
        isinstance(length, bool) or not isinstance(length, int) or length < 0
        for length in (q_len, k_len)
    ):
        raise ValueError("q_len and k_len must be nonnegative integers")
    if q_len > k_len:
        raise ValueError("q_len must be <= k_len for aligned causal positions")
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating-point torch.dtype")
    query_positions = torch.arange(k_len - q_len, k_len, device=device)
    key_positions = torch.arange(k_len, device=device)
    distances = (query_positions[:, None] - key_positions[None, :]).abs().to(dtype)
    head_ids = torch.arange(1, num_heads + 1, device=device, dtype=dtype)
    slopes = torch.exp2(-8.0 * head_ids / num_heads)
    return -slopes[None, :, None, None] * distances[None, None, :, :]


def streaming_mask(q_len: int, k_len: int, window: int, sink_tokens: int) -> torch.Tensor:
    """Return a boolean causal window mask with permanently visible initial sink keys."""
    if any(
        isinstance(length, bool) or not isinstance(length, int) or length < 0
        for length in (q_len, k_len)
    ):
        raise ValueError("q_len and k_len must be nonnegative integers")
    if q_len > k_len:
        raise ValueError("q_len must be <= k_len for aligned causal streaming")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be an integer >= 1")
    if isinstance(sink_tokens, bool) or not isinstance(sink_tokens, int) or sink_tokens < 0:
        raise ValueError("sink_tokens must be a nonnegative integer")

    query_positions = torch.arange(k_len - q_len, k_len)
    key_positions = torch.arange(k_len)
    causal = key_positions[None, :] <= query_positions[:, None]
    in_window = key_positions[None, :] >= query_positions[:, None] - window + 1
    is_sink = key_positions[None, :] < sink_tokens
    return causal & (in_window | is_sink)


class StreamingKVCache:
    """Detached inference cache retaining initial sinks and a bounded recent window.

    Append one position at a time. Returned int64 positions are absolute and
    should be used when applying position encodings. Cached tensors are detached
    and this cache does not support backpropagation through previous tokens.
    """

    def __init__(self, window: int, sink_tokens: int) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be an integer >= 1")
        if isinstance(sink_tokens, bool) or not isinstance(sink_tokens, int) or sink_tokens < 0:
            raise ValueError("sink_tokens must be a nonnegative integer")
        self.window = window
        self.sink_tokens = sink_tokens
        self.next_position = 0
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.positions: Optional[torch.Tensor] = None

    def append(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("key and value must be tensors")
        if key.ndim < 2 or value.ndim < 2 or key.shape[-2] != 1 or value.shape[-2] != 1:
            raise ValueError("key and value must each contain one position")
        if key.shape[:-1] != value.shape[:-1] or key.device != value.device:
            raise ValueError("key and value batch shapes and devices must match")
        key, value = key.detach(), value.detach()
        position = torch.tensor([self.next_position], dtype=torch.int64, device=key.device)
        if self.keys is None:
            keys, values, positions = key, value, position
        else:
            assert self.values is not None and self.positions is not None
            if (
                key.shape[:-2] != self.keys.shape[:-2]
                or key.shape[-1] != self.keys.shape[-1]
                or value.shape[-1] != self.values.shape[-1]
                or key.device != self.keys.device
                or key.dtype != self.keys.dtype
                or value.dtype != self.values.dtype
            ):
                raise ValueError("new keys and values must match the existing cache")
            keys = torch.cat((self.keys, key), dim=-2)
            values = torch.cat((self.values, value), dim=-2)
            positions = torch.cat((self.positions, position))
        keep = (positions < self.sink_tokens) | (positions >= self.next_position - self.window + 1)
        self.keys = keys[..., keep, :]
        self.values = values[..., keep, :]
        self.positions = positions[keep]
        self.next_position += 1
        return self.keys, self.values, self.positions

    def clear(self) -> None:
        """Discard all cached state and restart positions at zero."""
        self.keys = None
        self.values = None
        self.positions = None
        self.next_position = 0
