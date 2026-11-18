"""Independent numerical acceptance tests for native grouped-query attention."""

import math

import pytest
import torch

from zeta_scaling_toolkit import GroupedQueryAttention


def _expanded_reference(query, key, value, *, is_causal=True):
    """Reference using explicit KV repetition and ordinary scaled dot products."""
    groups = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    accumulation_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    logits = torch.matmul(
        query.to(accumulation_dtype), expanded_key.to(accumulation_dtype).transpose(-2, -1)
    )
    logits = logits / math.sqrt(query.shape[-1])
    if is_causal:
        query_length, key_length = query.shape[2], key.shape[2]
        query_positions = key_length - query_length + torch.arange(query_length)
        key_positions = torch.arange(key_length)
        logits = logits.masked_fill(
            ~(key_positions[None, :] <= query_positions[:, None])[None, None, :, :],
            -torch.inf,
        )
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.matmul(probabilities, expanded_value.to(accumulation_dtype))
    return (
        output.transpose(1, 2)
        .reshape(query.shape[0], query.shape[2], query.shape[1] * value.shape[-1])
        .to(query.dtype)
    )


@pytest.mark.parametrize(
    ("batch_size", "query_length", "key_length"),
    [(1, 1, 1), (2, 4, 4), (2, 2, 5), (1, 7, 7)],
)
def test_gqa_matches_expanded_reference_for_smollm2_dimensions(
    batch_size, query_length, key_length
):
    torch.manual_seed(917 + batch_size + query_length + key_length)
    query = torch.randn(batch_size, 15, query_length, 64)
    key = torch.randn(batch_size, 5, key_length, 64)
    value = torch.randn(batch_size, 5, key_length, 64)

    actual = GroupedQueryAttention()(query, key, value)
    expected = _expanded_reference(query, key, value)

    assert actual.shape == (batch_size, query_length, 960)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_gqa_maps_each_contiguous_query_group_to_one_kv_head():
    query = torch.zeros(1, 15, 1, 64)
    key = torch.zeros(1, 5, 1, 64)
    value = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1).expand(1, 5, 1, 64)

    actual = GroupedQueryAttention()(query, key, value).reshape(1, 1, 15, 64)
    expected_heads = torch.arange(5, dtype=torch.float32).repeat_interleave(3)

    torch.testing.assert_close(actual[0, 0, :, 0], expected_heads)
    torch.testing.assert_close(actual[0, 0, :, 1:], expected_heads[:, None].expand(15, 63))


def test_gqa_causal_mask_blocks_future_key_and_value_changes():
    torch.manual_seed(23)
    query = torch.randn(1, 15, 4, 64)
    key = torch.randn(1, 5, 4, 64)
    value = torch.randn(1, 5, 4, 64)
    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[:, :, 3] += 100
    changed_value[:, :, 3] -= 100

    attention = GroupedQueryAttention()
    actual = attention(query, key, value)
    changed = attention(query, changed_key, changed_value)
    expected = _expanded_reference(query, key, value)

    torch.testing.assert_close(actual[:, :3], changed[:, :3], atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_gqa_can_disable_causal_mask_and_matches_expanded_reference():
    torch.manual_seed(41)
    query = torch.randn(1, 15, 3, 64)
    key = torch.randn(1, 5, 5, 64)
    value = torch.randn(1, 5, 5, 64)

    actual = GroupedQueryAttention()(query, key, value, is_causal=False)
    expected = _expanded_reference(query, key, value, is_causal=False)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_gqa_preserves_dtype_device_and_has_finite_gradients(dtype):
    attention = GroupedQueryAttention()
    query = torch.randn(2, 15, 3, 64, dtype=dtype, requires_grad=True)
    key = torch.randn(2, 5, 5, 64, dtype=dtype, requires_grad=True)
    value = torch.randn(2, 5, 5, 64, dtype=dtype, requires_grad=True)

    output = attention(query, key, value)
    expected = _expanded_reference(query.detach(), key.detach(), value.detach())
    output.float().square().mean().backward()

    assert output.dtype == dtype
    assert output.device == query.device
    assert torch.isfinite(output).all()
    tolerances = {
        torch.float32: (1e-5, 1e-4),
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (1e-2, 1e-2),
        torch.float64: (1e-12, 1e-12),
    }
    atol, rtol = tolerances[dtype]
    torch.testing.assert_close(output.detach(), expected, atol=atol, rtol=rtol)
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_query_heads": 0}, "num_query_heads"),
        ({"num_key_value_heads": 0}, "num_key_value_heads"),
        ({"head_dim": 0}, "head_dim"),
        ({"num_query_heads": 4, "num_key_value_heads": 3}, "divisible"),
        ({"num_query_heads": True}, "num_query_heads"),
    ],
)
def test_gqa_rejects_invalid_configurations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GroupedQueryAttention(**kwargs)


def test_gqa_rejects_invalid_shapes_and_mixed_tensor_properties():
    attention = GroupedQueryAttention()
    query = torch.randn(1, 15, 2, 64)
    key = torch.randn(1, 5, 2, 64)
    value = torch.randn(1, 5, 2, 64)

    with pytest.raises(ValueError, match="rank 4"):
        attention(query[:, :, 0], key, value)
    with pytest.raises(ValueError, match="15 heads"):
        attention(query[:, :12], key, value)
    with pytest.raises(ValueError, match="batch sizes must match"):
        attention(query, key.expand(2, -1, -1, -1), value)
    with pytest.raises(ValueError, match="key and value must each have 5 heads"):
        attention(query, key[:, :4], value[:, :4])
    with pytest.raises(ValueError, match="head dimensions must equal 64"):
        attention(query, key[..., :32], value[..., :32])
    with pytest.raises(ValueError, match="key/value sequence length"):
        attention(query, key[:, :, :0], value[:, :, :0])
    with pytest.raises(ValueError, match="key and value sequence lengths"):
        attention(query, key, value[:, :, :1])
    with pytest.raises(ValueError, match="same dtype"):
        attention(query, key.double(), value)
    with pytest.raises(TypeError, match="floating-point"):
        attention(query.long(), key, value)
    with pytest.raises(TypeError, match="is_causal"):
        attention(query, key, value, is_causal=1)
    with pytest.raises(ValueError, match="query length must be <= key length"):
        attention(query, key[:, :, :1], value[:, :, :1])
