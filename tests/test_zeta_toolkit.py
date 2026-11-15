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


@pytest.mark.parametrize("length", [1, 8, 4096])
def test_attention_uses_qk_normalization(length):
    torch.manual_seed(0)
    q = torch.randn(1, 1 if length == 4096 else length, 16)
    k, v = (torch.randn(1, length, 16) for _ in range(2))
    attention = NonCollapsingAttention(16, A_attn=2.5)
    actual = attention(q, k, v)
    logits = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1) / 4
    torch.testing.assert_close(actual, F.softmax(logits, dim=-1) @ v)
    assert torch.isfinite(actual).all()


def test_attention_scales_are_trainable_and_legacy_parameter_is_ignored():
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 3, 4) for _ in range(3))
    attention = NonCollapsingAttention(4, A_attn=1.0)
    output = attention(q, k, v)
    attention.A_attn = 0.0
    torch.testing.assert_close(attention(q, k, v), output)
    output.square().sum().backward()
    assert attention.q_scale.grad is not None and attention.q_scale.grad.abs().item() > 0
    assert attention.k_scale.grad is not None and attention.k_scale.grad.abs().item() > 0


def test_attention_handles_zero_vectors():
    q = torch.zeros(1, 2, 4)
    k = torch.zeros(1, 3, 4)
    v = torch.randn(1, 3, 4)
    actual = NonCollapsingAttention(4)(q, k, v)
    torch.testing.assert_close(actual, v.mean(dim=-2, keepdim=True).expand_as(actual))


def test_scheduler_starts_at_base_and_decays_to_configured_floor():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = SGD([param], lr=0.1)
    scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=0.5, total_steps=4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    for step in range(1, 7):
        optimizer.step()
        scheduler.step()
        expected = 0.1 * (0.5 + 0.5 * (1 + math.cos(math.pi * min(step, 4) / 4)) / 2)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_scheduler_zero_total_steps_starts_at_floor():
    optimizer = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    AsymptoticLRScheduler(optimizer, floor_ratio=0.25, total_steps=0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)


def test_coordinates_are_normalized_polar_and_resolution_independent():
    coords = get_scale_invariant_vit_coords(4, 6)
    assert coords.shape == (4, 6, 2)
    assert coords[0, 0, 0].item() == pytest.approx(math.sqrt(0.5))
    assert coords[0, 0, 1].item() == pytest.approx(-3 * math.pi / 4)
    torch.testing.assert_close(coords[2, 3], torch.zeros(2))
    larger = get_scale_invariant_vit_coords(8, 12, T=1, A_coords=2)
    torch.testing.assert_close(coords, larger[::2, ::2])


@pytest.mark.parametrize("in_features,out_features", [(8, 16), (16, 8)])
def test_linear_uses_orthogonal_weights(in_features, out_features):
    torch.manual_seed(0)
    layer = make_corrected_linear(in_features, out_features)
    weight = layer.weight
    gram = weight.T @ weight if out_features >= in_features else weight @ weight.T
    torch.testing.assert_close(gram, torch.eye(min(in_features, out_features)), atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(layer.bias) == 0


def test_linear_supports_no_bias_and_other_nonlinearities():
    torch.manual_seed(0)
    layer = make_corrected_linear(8, 16, bias=False, nonlinearity="linear")
    assert layer.bias is None
    torch.testing.assert_close(layer.weight.T @ layer.weight, torch.eye(8), atol=1e-6, rtol=1e-6)
