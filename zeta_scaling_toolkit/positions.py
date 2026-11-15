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
    interpolation rule in log space to avoid intermediate overflow. Extremely
    large scale factors can still limit positional precision. Create positions
    directly as int64 (for example with ``torch.arange``); converting from a
    low-precision floating tensor cannot recover already-lost position bits.
    """
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating-point torch.Tensor")
    if x.ndim < 2 or x.shape[-1] < 2 or x.shape[-1] % 2:
        raise ValueError("x must have rank >= 2 and an even last dimension >= 2")
    if not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64:
        raise TypeError(
            "positions must be an int64 torch.Tensor; create them as integers before "
            "any floating-point conversion"
        )
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
    position_values = positions.to(torch.float64)
    log_base = math.log(float(base))
    if scaling == "linear":
        position_values = position_values / scale_factor
    elif scaling == "ntk" and scale_factor > 1:
        exponent = dimension / (dimension - 2) if dimension > 2 else 1.0
        log_base += exponent * math.log(float(scale_factor))

    frequencies = torch.arange(0, dimension, 2, device=x.device, dtype=torch.float64)
    inverse_frequencies = torch.exp(-log_base * frequencies / dimension)
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
    chunk_size: int = 128,
) -> torch.Tensor:
    """Return negative absolute-distance ALiBi biases with shape (1, heads, Q, K).

    Query rows are built in chunks to limit temporary distance and bias tensors.
    The returned bias remains dense; use ``NonCollapsingAttention.forward_streaming``
    to avoid materializing a full bias for sink/window attention.
    """
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
        raise ValueError("num_heads must be an integer >= 1")
    if any(
        isinstance(length, bool) or not isinstance(length, int) or length < 0
        for length in (q_len, k_len)
    ):
        raise ValueError("q_len and k_len must be nonnegative integers")
    if q_len > k_len:
        raise ValueError("q_len must be <= k_len for aligned causal positions")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be an integer >= 1")
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating-point torch.dtype")
    key_positions = torch.arange(k_len, device=device)
    head_ids = torch.arange(1, num_heads + 1, device=device, dtype=dtype)
    slopes = torch.exp2(-8.0 * head_ids / num_heads)
    bias = torch.empty((1, num_heads, q_len, k_len), dtype=dtype, device=device)
    query_offset = k_len - q_len
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        query_positions = torch.arange(query_offset + start, query_offset + end, device=device)
        distances = (query_positions[:, None] - key_positions[None, :]).abs().to(dtype)
        bias[:, :, start:end, :] = -slopes[None, :, None, None] * distances[None, None, :, :]
    return bias


def streaming_mask(
    q_len: int,
    k_len: int,
    window: int,
    sink_tokens: int,
    device: Optional[Union[torch.device, str]] = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Return a boolean causal window mask with permanently visible initial sink keys.

    The dense output contract is unchanged. Query chunks bound temporary mask
    construction memory; ``device`` can place the result directly with attention
    inputs. For a genuinely sparse sink/window operation, use
    ``NonCollapsingAttention.forward_streaming`` instead.
    """
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
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be an integer >= 1")

    key_positions = torch.arange(k_len, device=device)
    mask = torch.empty((q_len, k_len), dtype=torch.bool, device=device)
    query_offset = k_len - q_len
    sink_end = min(sink_tokens, k_len)
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        query_positions = torch.arange(query_offset + start, query_offset + end, device=device)
        block = key_positions[None, :] <= query_positions[:, None]
        block &= key_positions[None, :] >= query_positions[:, None] - window + 1
        if sink_end:
            block[:, :sink_end] |= key_positions[:sink_end][None, :] <= query_positions[:, None]
        mask[start:end] = block
    return mask


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
