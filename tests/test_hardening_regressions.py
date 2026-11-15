import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    alibi_bias,
    calibrate_gain,
    make_corrected_linear,
    streaming_mask,
)
from zeta_scaling_toolkit.core import _unit


def test_temperature_stays_within_bounds_under_aggressive_optimization():
    attention = NonCollapsingAttention(4, tau_min=2.0)
    optimizer = Adam([attention.raw_tau], lr=0.5)

    for _ in range(1000):
        optimizer.zero_grad()
        attention.tau.backward()
        optimizer.step()
        assert 2.0 <= attention.tau.item() <= attention.tau_max


@pytest.mark.parametrize("activation", [nn.GELU, nn.SiLU])
def test_calibrate_gain_preserves_deep_smooth_activation_norms(activation):
    torch.manual_seed(0)
    modules = []
    for _ in range(30):
        modules.extend((nn.Linear(32, 32), activation()))
    model = nn.Sequential(*modules)
    sample = torch.randn(128, 32)
    rng_state = torch.random.get_rng_state()

    calibrate_gain(model, sample, iters=20)

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    with torch.no_grad():
        ratio = model(sample).norm() / sample.norm()
    assert 0.5 <= ratio.item() <= 2.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_calculate_gain_error_explains_smooth_activation_calibration():
    with pytest.raises(ValueError, match="calibrate_gain"):
        make_corrected_linear(8, 8, nonlinearity="gelu")


def test_fully_masked_rows_return_zero_with_finite_gradients():
    attention = NonCollapsingAttention(8)
    q, k, v = (torch.randn(2, n, 8, requires_grad=True) for n in (3, 4, 4))
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)

    output = attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (q, k, v)
    )
    assert attention.raw_tau.grad is not None
    assert torch.isfinite(attention.raw_tau.grad)


def test_empty_key_sequence_raises_but_empty_query_and_batch_are_supported():
    attention = NonCollapsingAttention(8)
    with pytest.raises(ValueError, match="key sequence"):
        attention(torch.empty(2, 0, 8), torch.empty(2, 0, 8), torch.empty(2, 0, 5))

    empty_query = attention(torch.empty(2, 0, 8), torch.randn(2, 3, 8), torch.randn(2, 3, 5))
    assert empty_query.shape == (2, 0, 5)
    empty_batch = attention(torch.empty(0, 2, 8), torch.empty(0, 3, 8), torch.empty(0, 3, 5))
    assert empty_batch.shape == (0, 2, 5)


def test_attention_validates_inputs_and_optional_finiteness():
    attention = NonCollapsingAttention(8, check_finite=True)
    good = torch.randn(2, 3, 8)

    with pytest.raises(TypeError, match="Tensor"):
        attention(good, good, [good])
    with pytest.raises(ValueError, match="rank"):
        attention(torch.randn(8), good, good)
    with pytest.raises(ValueError, match="batch"):
        attention(good, torch.randn(4, 3, 8), torch.randn(4, 3, 8))
    with pytest.raises(ValueError, match="broadcast"):
        attention(good, good, good, attn_mask=torch.ones(5, 5))
    with pytest.raises(ValueError, match="finite"):
        attention(good * float("inf"), good, good)
    with pytest.raises(TypeError, match="attn_mask"):
        attention(good, good, good, attn_mask=1)


def test_attention_handles_finite_extreme_magnitudes_and_noncontiguous_inputs():
    attention = NonCollapsingAttention(8)
    q = torch.tensor([[[1e-30, -1e-30] * 8, [1e30, -1e30] * 8]])[..., ::2]
    k = torch.tensor([[[1e-30, -1e-30] * 8] * 3])[..., ::2]
    v = torch.randn(1, 3, 8)
    output = attention(q, k, v)
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert torch.isfinite(output).all()


def test_scale_invariance_boundary_matches_normalization_epsilon():
    small = _unit(torch.tensor([[1e-4, 0.0]]))
    boundary = _unit(torch.tensor([[1e-2, 0.0]]))
    large = _unit(torch.tensor([[1.0, 0.0]]))

    assert torch.linalg.vector_norm(boundary - large).item() < 0.01
    assert torch.linalg.vector_norm(small - large).item() > 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau0": float("nan")},
        {"tau0": float("inf")},
        {"tau_max": float("nan")},
        {"tau_max": float("inf")},
        {"tau_min": float("nan")},
        {"tau_min": float("inf")},
        {"tau_min": 11.0},
        {"tau_min": 60.0},
    ],
)
def test_attention_rejects_nonfinite_temperature_configuration(kwargs):
    with pytest.raises(ValueError):
        NonCollapsingAttention(8, **kwargs)


@pytest.mark.parametrize("floor_ratio", [float("nan"), float("inf"), -float("inf")])
def test_scheduler_rejects_nonfinite_floor_ratio(floor_ratio):
    optimizer = SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    with pytest.raises(ValueError):
        AsymptoticLRScheduler(optimizer, floor_ratio=floor_ratio)


def test_initialization_rejects_unknown_gain_with_clear_error():
    with pytest.raises(ValueError, match="nonlinearity"):
        make_corrected_linear(8, 8, nonlinearity="not-an-activation")


def test_log_scaling_with_additive_bias_and_explicit_visibility():
    torch.manual_seed(7)
    attention = NonCollapsingAttention(4, length_scaling="log", train_length=2).double()
    q = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 6, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    visible = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, True, True, True, True, True],
            [False, False, False, False, False, False],
        ]
    )
    actual = attention(q, k, v, bias, visibility_mask=visible)
    expected = attention(q, k, v, bias.masked_fill(~visible, float("-inf")))
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[0, 2], torch.zeros(3, dtype=torch.float64))
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), (q, k, v, bias, attention.raw_tau)),
        torch.autograd.grad(expected.sum(), (q, k, v, bias, attention.raw_tau)),
    ):
        torch.testing.assert_close(actual_grad, expected_grad)
        assert torch.isfinite(actual_grad).all()


def test_visibility_mask_intersects_boolean_mask_and_is_validated():
    attention = NonCollapsingAttention(4, length_scaling="log", train_length=2)
    q, k, v = torch.randn(1, 2, 4), torch.randn(1, 4, 4), torch.randn(1, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    visible = torch.tensor([True, False, True, False])
    torch.testing.assert_close(
        attention(q, k, v, mask, visibility_mask=visible), attention(q, k, v, mask & visible)
    )
    with pytest.raises(TypeError, match="visibility_mask"):
        attention(q, k, v, visibility_mask=visible.float())
    with pytest.raises(ValueError, match="broadcastable"):
        attention(q, k, v, visibility_mask=torch.ones(5, dtype=torch.bool))
    with pytest.raises(ValueError, match="same device"):
        attention(q, k, v, visibility_mask=torch.ones(4, dtype=torch.bool, device="meta"))


@pytest.mark.parametrize("streaming", [False, True])
def test_attention_reports_module_device_mismatch(streaming):
    attention = NonCollapsingAttention(4)
    q = torch.empty(1, 2, 4, device="meta")
    with pytest.raises(ValueError, match=r"attention\.to\(q\.device\)"):
        if streaming:
            attention.forward_streaming(q, q, q, window=2)
        else:
            attention(q, q, q)


@pytest.mark.parametrize("length_scaling", ["none", "log"])
@pytest.mark.parametrize("batch_shape,head_axis", [((3,), 0), ((3, 2), 0), ((2, 3), -1)])
def test_streaming_alibi_explicit_head_axis_matches_dense(length_scaling, batch_shape, head_axis):
    torch.manual_seed(8)
    q = torch.randn(*batch_shape, 4, 4, dtype=torch.float64)
    k = torch.randn(*batch_shape, 6, 4, dtype=torch.float64)
    v = torch.randn(*batch_shape, 6, 2, dtype=torch.float64)
    attention = NonCollapsingAttention(4, length_scaling=length_scaling, train_length=2).double()
    axis = head_axis % len(batch_shape)
    bias = alibi_bias(batch_shape[axis], 4, 6, dtype=torch.float64)[0]
    shape = [1] * (len(batch_shape) + 2)
    shape[axis], shape[-2], shape[-1] = batch_shape[axis], 4, 6
    mask = streaming_mask(4, 6, window=2, sink_tokens=1)
    dense = bias.reshape(shape).masked_fill(~mask, float("-inf"))
    expected = attention(q, k, v, dense)
    actual = attention.forward_streaming(
        q, k, v, window=2, sink_tokens=1, alibi=True, head_axis=head_axis, chunk_size=2
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_streaming_head_axis_validation():
    attention = NonCollapsingAttention(4)
    q = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="requires alibi"):
        attention.forward_streaming(q, q, q, window=2, head_axis=0)
    with pytest.raises(ValueError, match="batch dimension"):
        attention.forward_streaming(q, q, q, window=2, alibi=True, head_axis=1)
    with pytest.raises(TypeError, match="head_axis"):
        attention.forward_streaming(q, q, q, window=2, alibi=True, head_axis=True)


def test_scheduler_resume_explains_missing_initial_lr():
    optimizer = SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    with pytest.raises(KeyError, match="restore optimizer state"):
        AsymptoticLRScheduler(optimizer, last_epoch=1)
