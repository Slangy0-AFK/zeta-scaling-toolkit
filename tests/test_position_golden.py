"""Golden-value tests that pin RoPE scaling modes and ALiBi slopes (kill mutation survivors)."""

import pytest
import torch

from zeta_scaling_toolkit import alibi_bias, rotary_embedding

BASE = 10000.0


def _reference_rope(x, positions, base, position_scale=1.0):
    """Independent float64 reference: rotate (even, odd) pairs by pos/scale * base**(-2i/d)."""
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    angles = (positions.to(torch.float64) / position_scale)[..., None] * inv
    cos, sin = torch.cos(angles), torch.sin(angles)
    xd = x.to(torch.float64)
    even, odd = xd[..., 0::2], xd[..., 1::2]
    out = torch.empty_like(xd)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out.to(x.dtype)


@pytest.fixture()
def sample():
    torch.manual_seed(0)
    return torch.randn(5, 16), torch.tensor([0, 5, 1000, 2**20, 10**9], dtype=torch.int64)


def test_rope_without_scaling_matches_reference(sample):
    x, pos = sample
    torch.testing.assert_close(
        rotary_embedding(x, pos), _reference_rope(x, pos, BASE), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0, 16.0])
def test_linear_scaling_divides_positions(sample, scale):
    x, pos = sample
    actual = rotary_embedding(x, pos, scaling="linear", scale_factor=scale)
    torch.testing.assert_close(
        actual, _reference_rope(x, pos, BASE, position_scale=scale), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("scale", [2.0, 8.0, 64.0])
def test_ntk_scaling_raises_the_base_by_scale_to_d_over_d_minus_2(sample, scale):
    x, pos = sample
    d = x.shape[-1]
    expected_base = BASE * scale ** (d / (d - 2))
    actual = rotary_embedding(x, pos, scaling="ntk", scale_factor=scale)
    torch.testing.assert_close(actual, _reference_rope(x, pos, expected_base), atol=1e-6, rtol=1e-6)


def test_ntk_two_dimensional_edge_case_uses_exponent_one():
    x = torch.randn(3, 2)
    pos = torch.tensor([1, 10, 100], dtype=torch.int64)
    actual = rotary_embedding(x, pos, scaling="ntk", scale_factor=4.0)
    torch.testing.assert_close(actual, _reference_rope(x, pos, BASE * 4.0), atol=1e-6, rtol=1e-6)


def test_scale_factor_one_is_identity_for_both_modes(sample):
    x, pos = sample
    plain = rotary_embedding(x, pos)
    for mode in ("linear", "ntk"):
        torch.testing.assert_close(rotary_embedding(x, pos, scaling=mode, scale_factor=1.0), plain)


def test_scaling_modes_are_distinct_from_each_other_and_from_none(sample):
    x, pos = sample
    plain = rotary_embedding(x, pos)
    linear = rotary_embedding(x, pos, scaling="linear", scale_factor=4.0)
    ntk = rotary_embedding(x, pos, scaling="ntk", scale_factor=4.0)
    assert not torch.allclose(plain, linear, atol=1e-3)
    assert not torch.allclose(plain, ntk, atol=1e-3)
    assert not torch.allclose(linear, ntk, atol=1e-3)


@pytest.mark.parametrize("heads", [1, 4, 8])
def test_alibi_slopes_are_two_to_the_minus_eight_h_over_heads(heads):
    bias = alibi_bias(heads, 6, 6)
    distances = (torch.arange(6)[:, None] - torch.arange(6)[None, :]).abs().to(torch.float32)
    for h in range(1, heads + 1):
        expected = -(2.0 ** (-8.0 * h / heads)) * distances
        torch.testing.assert_close(bias[0, h - 1], expected, atol=0.0, rtol=1e-6)


def test_alibi_four_head_slopes_literal_values():
    bias = alibi_bias(4, 1, 2)  # one query at the last position, distance 1 to key 0
    assert bias[0, :, 0, 0].tolist() == pytest.approx([-0.25, -0.0625, -0.015625, -0.00390625])
    assert bias[0, :, 0, 1].tolist() == [0.0, 0.0, 0.0, 0.0]
