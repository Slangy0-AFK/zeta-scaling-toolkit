"""Scaled grouped-query attention for separately projected Q, K, and V tensors."""

import math

import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    """Apply raw scaled-dot-product attention with grouped key/value heads.

    Inputs use ``[batch, heads, sequence, head_dim]`` layout: queries have
    ``num_query_heads`` heads, while keys and values have
    ``num_key_value_heads`` heads. Query heads are assigned to KV heads in
    contiguous groups (for example, query heads 0-2 use KV head 0).

    Causal queries align to the end of the key sequence, which supports
    decoding a query suffix against a longer cached key/value prefix. The
    returned attention heads are merged to ``[batch, query_length,
    num_query_heads * head_dim]``; no output projection is applied.

    Grouping is computed without physically repeating K/V heads. Attention
    logits still scale with the full query-to-key product; this implementation
    does not dispatch to a specialized fused GQA kernel.
    """

    def __init__(
        self,
        num_query_heads: int = 15,
        num_key_value_heads: int = 5,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_query_heads", num_query_heads),
            ("num_key_value_heads", num_key_value_heads),
            ("head_dim", head_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if num_query_heads % num_key_value_heads:
            raise ValueError(
                "num_query_heads must be divisible by num_key_value_heads, "
                f"got {num_query_heads} and {num_key_value_heads}"
            )
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.groups_per_key_value_head = num_query_heads // num_key_value_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Return merged attention heads, optionally applying a causal mask.

        For causal inputs with ``query_length < key_length``, query position
        ``i`` is interpreted as absolute position ``key_length - query_length + i``.
        Queries longer than the keys are rejected because this suffix alignment
        cannot represent their causal positions.
        """
        if not isinstance(is_causal, bool):
            raise TypeError("is_causal must be a bool")
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must have rank 4 with [batch, heads, sequence, head_dim] layout, "
                    f"got rank {tensor.ndim}"
                )
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")

        if query.device != key.device or query.device != value.device:
            raise ValueError("query, key, and value must be on the same device")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("query, key, and value must have the same dtype")
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            raise ValueError("query, key, and value batch sizes must match")
        if query.shape[1] != self.num_query_heads:
            raise ValueError(f"query must have {self.num_query_heads} heads, got {query.shape[1]}")
        if key.shape[1] != self.num_key_value_heads or value.shape[1] != self.num_key_value_heads:
            raise ValueError(
                "key and value must each have "
                f"{self.num_key_value_heads} heads, got {key.shape[1]} and {value.shape[1]}"
            )
        if query.shape[-1] != self.head_dim or key.shape[-1] != self.head_dim:
            raise ValueError(
                f"query and key head dimensions must equal {self.head_dim}, "
                f"got {query.shape[-1]} and {key.shape[-1]}"
            )
        if value.shape[-1] != self.head_dim:
            raise ValueError(
                f"value head dimension must equal {self.head_dim}, got {value.shape[-1]}"
            )
        if key.shape[2] == 0:
            raise ValueError("key/value sequence length must be nonzero")
        if key.shape[2] != value.shape[2]:
            raise ValueError("key and value sequence lengths must match")
        if is_causal and query.shape[2] > key.shape[2]:
            raise ValueError("causal query length must be <= key length for end-aligned positions")

        accumulation_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        batch_size, _, query_length, _ = query.shape
        key_length = key.shape[2]
        grouped_query = query.to(accumulation_dtype).reshape(
            batch_size,
            self.num_key_value_heads,
            self.groups_per_key_value_head,
            query_length,
            self.head_dim,
        )
        keys = key.to(accumulation_dtype)
        values = value.to(accumulation_dtype)
        logits = torch.einsum("bhgqd,bhkd->bhgqk", grouped_query, keys)
        logits = logits * (1.0 / math.sqrt(self.head_dim))

        if is_causal:
            query_positions = (
                key_length - query_length + torch.arange(query_length, device=query.device)
            )
            key_positions = torch.arange(key_length, device=query.device)
            causal_mask = key_positions[None, :] <= query_positions[:, None]
            logits = logits.masked_fill(~causal_mask[None, None, None, :, :], -torch.inf)

        probabilities = torch.softmax(logits, dim=-1)
        grouped_output = torch.einsum("bhgqk,bhkd->bhgqd", probabilities, values)
        output = grouped_output.reshape(
            batch_size, self.num_query_heads, query_length, self.head_dim
        )
        return (
            output.transpose(1, 2)
            .reshape(batch_size, query_length, self.num_query_heads * self.head_dim)
            .to(query.dtype)
        )
