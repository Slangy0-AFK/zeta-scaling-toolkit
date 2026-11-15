import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    calibrate_gain,
    get_isotropic_coords,
    make_corrected_linear,
)


def test_attention_constructor_validates_dimension_and_flags():
    for d_k in (0, -1, True):
        with pytest.raises(ValueError, match="d_k"):
            NonCollapsingAttention(d_k)
    with pytest.raises(TypeError, match="check_finite"):
        NonCollapsingAttention(8, check_finite=1)


def test_attention_rejects_bad_tensor_dtype_and_value_sequence_length():
    attention = NonCollapsingAttention(8)
    with pytest.raises(TypeError, match="floating-point"):
        attention(
            torch.ones(2, 3, 8, dtype=torch.int64), torch.randn(2, 3, 8), torch.randn(2, 3, 8)
        )
    with pytest.raises(ValueError, match="sequence lengths"):
        attention(torch.randn(2, 3, 8), torch.randn(2, 4, 8), torch.randn(2, 3, 8))


def test_attention_float_masks_handle_fully_blocked_rows_and_reject_nan():
    attention = NonCollapsingAttention(4)
    q, k, v = (torch.randn(1, n, 4, requires_grad=True) for n in (2, 3, 3))
    mask = torch.zeros(2, 3)
    mask[0, :] = float("-inf")
    output = attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(output[:, 0], torch.zeros_like(output[:, 0]))
    assert torch.isfinite(output[:, 1]).all()
    output.sum().backward()
    assert torch.isfinite(q.grad).all()

    for invalid in (torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]])):
        with pytest.raises(ValueError, match="attn_mask"):
            attention(q.detach(), k.detach(), v.detach(), attn_mask=invalid)
    with pytest.raises(TypeError, match="bool or floating"):
        attention(q.detach(), k.detach(), v.detach(), attn_mask=torch.ones(2, 3, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_zero_vector_gradients_are_finite_and_bounded_for_all_float_dtypes(dtype):
    attention = NonCollapsingAttention(8)
    q = torch.zeros(2, 3, 8, dtype=dtype, requires_grad=True)
    k = torch.zeros(2, 4, 8, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 4, 8, dtype=dtype, requires_grad=True)
    output = attention(q, k, v)
    output.float().square().sum().backward()
    for gradient in (q.grad, k.grad, v.grad, attention.raw_tau.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max().item() < 1e5


def test_attention_rejects_mask_and_batch_broadcast_mismatches():
    attention = NonCollapsingAttention(4)
    with pytest.raises(ValueError, match="batch dimensions"):
        attention(torch.randn(2, 3, 4), torch.randn(5, 3, 4), torch.randn(5, 3, 4))
    with pytest.raises(ValueError, match="broadcastable"):
        attention(
            torch.randn(2, 3, 4), torch.randn(2, 3, 4), torch.randn(2, 3, 4), torch.ones(4, 4)
        )
    with pytest.raises(ValueError, match="same device"):
        attention(torch.randn(1, 3, 4), torch.empty(1, 3, 4, device="meta"), torch.randn(1, 3, 4))
    with pytest.raises(ValueError, match="same device"):
        attention(
            torch.randn(1, 3, 4),
            torch.randn(1, 3, 4),
            torch.randn(1, 3, 4),
            torch.empty(3, 3, dtype=torch.bool, device="meta"),
        )


def test_scheduler_rejects_boolean_and_non_numeric_arguments():
    parameter = nn.Parameter(torch.zeros(1))
    for kwargs in (
        {"total_steps": True},
        {"warmup_steps": True},
        {"floor_ratio": float("nan")},
        {"floor_ratio": float("inf")},
        {"floor_ratio": "0.5"},
    ):
        optimizer = SGD([parameter], lr=0.1)
        with pytest.raises(ValueError):
            AsymptoticLRScheduler(optimizer, **kwargs)


def test_coordinates_validate_integral_sizes_and_floating_dtype():
    for height, width in ((1.5, 4), (True, 4), (4, 0)):
        with pytest.raises(ValueError, match="height and width"):
            get_isotropic_coords(height, width)
    with pytest.raises(TypeError, match="floating-point"):
        get_isotropic_coords(4, 4, dtype=torch.int64)


def test_linear_initialization_is_reproducible_and_rejects_non_string_gain():
    torch.manual_seed(19)
    first = make_corrected_linear(8, 8).weight.detach().clone()
    torch.manual_seed(19)
    second = make_corrected_linear(8, 8).weight.detach().clone()
    torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match="unsupported nonlinearity"):
        make_corrected_linear(8, 8, nonlinearity=[])


@pytest.mark.parametrize("target_ratio", [0.0, float("nan"), float("inf")])
def test_calibrate_gain_rejects_invalid_target_ratios(target_ratio):
    with pytest.raises(ValueError, match="target_ratio"):
        calibrate_gain(nn.Sequential(nn.Linear(4, 4)), torch.randn(8, 4), target_ratio)


def test_calibrate_gain_validates_stack_input_and_preserves_training_state_on_failure():
    model = nn.Sequential(nn.Linear(4, 4))
    model.train()
    with pytest.raises(TypeError, match="sample_input"):
        calibrate_gain(model, [torch.randn(8, 4)])
    with pytest.raises(TypeError, match="model_or_layer_stack"):
        calibrate_gain(nn.ReLU(), torch.randn(8, 4))
    with pytest.raises(ValueError, match="iters"):
        calibrate_gain(model, torch.randn(8, 4), iters=0)

    empty = nn.Sequential()
    with pytest.raises(ValueError, match="must not be empty"):
        calibrate_gain(empty, torch.randn(8, 4))
    with pytest.raises(ValueError, match="nn.Linear"):
        calibrate_gain(nn.Sequential(nn.ReLU()), torch.randn(8, 4))

    zero_layer = nn.Linear(4, 4)
    nn.init.zeros_(zero_layer.weight)
    nn.init.zeros_(zero_layer.bias)
    zero_model = nn.Sequential(zero_layer)
    with pytest.raises(ValueError, match="zero or non-finite"):
        calibrate_gain(zero_model, torch.zeros(8, 4))
    assert model.training
    assert zero_model.training


def test_calibrate_gain_accepts_a_module_list():
    layers = nn.ModuleList([nn.Linear(4, 4), nn.GELU()])
    result = calibrate_gain(layers, torch.randn(8, 4), iters=2)
    assert result is layers
