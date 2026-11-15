"""Regression tests for the corrected scaling formulas."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.optim import SGD

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)


@pytest.mark.parametrize("a_attn", [0.0, 1.0, 2.5])
def test_attention_uses_mild_temperature(a_attn):
    torch.manual_seed(0)
    Q, K, V = (torch.randn(2, 8, 16) for _ in range(3))
    actual = NonCollapsingAttention(16, A_attn=a_attn)(Q, K, V)
    logits = Q @ K.transpose(-2, -1) / (math.sqrt(16) * (1 + a_attn / math.log(8)))
    torch.testing.assert_close(actual, F.softmax(logits, dim=-1) @ V)


def test_attention_rejects_single_token():
    x = torch.randn(1, 1, 16)
    with pytest.raises(ValueError, match="Sequence length N must be > 1"):
        NonCollapsingAttention(16)(x, x, x)


def test_scheduler_starts_at_base_and_decays_to_configured_floor():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = SGD([param], lr=0.1)
    scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    for step in range(1, 5):
        optimizer.step()
        scheduler.step()
        expected = 0.1 * max(math.log(2) / math.log(max(step + 1, 2)), 0.5)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


@pytest.mark.parametrize("offset", [0.0, 2.0])
def test_coordinate_radius_and_shape(offset):
    T = 1024
    coords = get_scale_invariant_vit_coords(8, 9, T, A_coords=offset)
    assert coords.shape == (8, 9, 2)
    expected_radius = math.sqrt((math.log(T) + offset) / math.pi)
    torch.testing.assert_close(
        torch.linalg.vector_norm(coords[-1], dim=-1),
        torch.full((9,), expected_radius),
    )
    assert torch.count_nonzero(coords[0]) == 0


def test_coordinates_reject_invalid_scale():
    with pytest.raises(ValueError, match="Input scale T must be > 1"):
        get_scale_invariant_vit_coords(8, 8, 1)


@pytest.mark.parametrize("fan_in", [128, 1024])
def test_linear_uses_kaiming_variance(fan_in):
    torch.manual_seed(0)
    layer = make_corrected_linear(fan_in, 512)
    assert layer.weight.var().item() == pytest.approx(2 / fan_in, rel=0.05)
    assert torch.count_nonzero(layer.bias) == 0


def test_linear_supports_no_bias_and_other_nonlinearities():
    torch.manual_seed(0)
    layer = make_corrected_linear(128, 512, bias=False, nonlinearity="linear")
    assert layer.bias is None
    assert layer.weight.var().item() == pytest.approx(1 / 128, rel=0.05)
