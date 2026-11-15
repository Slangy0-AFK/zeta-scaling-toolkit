import copy
import math
import pickle

import pytest
import torch
import torch.nn.functional as F
from hypothesis import assume, given
from hypothesis import strategies as st
from torch.optim import SGD

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    get_isotropic_coords,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)
from zeta_scaling_toolkit.core import _unit


@given(
    batch=st.integers(0, 3),
    heads=st.integers(1, 3),
    query_len=st.integers(0, 8),
    key_len=st.integers(1, 8),
    d_k=st.integers(1, 16),
    d_v=st.integers(1, 12),
)
def test_attention_random_shapes_return_finite_outputs(batch, heads, query_len, key_len, d_k, d_v):
    torch.manual_seed(0)
    q = torch.randn(batch, heads, query_len, d_k)
    k = torch.randn(batch, heads, key_len, d_k)
    v = torch.randn(batch, heads, key_len, d_v)
    output = NonCollapsingAttention(d_k)(q, k, v)
    assert output.shape == (batch, heads, query_len, d_v)
    assert torch.isfinite(output).all()


@given(dtype=st.sampled_from([torch.float32, torch.float64, torch.bfloat16, torch.float16]))
def test_attention_extreme_finite_inputs_stay_finite(dtype):
    maximum = {torch.float16: 6e4, torch.bfloat16: 1e30, torch.float32: 1e30, torch.float64: 1e300}[
        dtype
    ]
    minimum = {
        torch.float16: 1e-5,
        torch.bfloat16: 1e-30,
        torch.float32: 1e-30,
        torch.float64: 1e-300,
    }[dtype]
    q = torch.tensor([[[0.0, minimum, -minimum, maximum, -maximum, 1.0, -1.0, 0.5]]], dtype=dtype)
    k = q.expand(1, 3, 8).clone()
    v = torch.tensor([[[maximum] * 8, [-maximum] * 8, [1.0] * 8]], dtype=dtype)
    output = NonCollapsingAttention(8)(q, k, v)
    assert torch.isfinite(output).all()


@given(
    batch=st.integers(1, 3),
    query_len=st.integers(1, 8),
    key_len=st.integers(1, 8),
    d_k=st.integers(1, 12),
)
def test_attention_weights_sum_to_one_and_output_stays_in_value_hull(
    batch, query_len, key_len, d_k
):
    torch.manual_seed(0)
    q = torch.randn(batch, query_len, d_k)
    k = torch.randn(batch, key_len, d_k)
    values = torch.eye(key_len).expand(batch, -1, -1).clone()
    weights = NonCollapsingAttention(d_k)(q, k, values)
    torch.testing.assert_close(weights.sum(-1), torch.ones(batch, query_len), atol=1e-5, rtol=1e-5)
    assert torch.all(weights >= -1e-6)
    assert torch.all(weights <= 1.0 + 1e-6)

    feature_values = torch.randn(batch, key_len, 5)
    output = NonCollapsingAttention(d_k)(q, k, feature_values)
    assert torch.all(output >= feature_values.amin(dim=-2, keepdim=True) - 1e-5)
    assert torch.all(output <= feature_values.amax(dim=-2, keepdim=True) + 1e-5)


@given(
    batch=st.integers(1, 3),
    query_len=st.integers(1, 8),
    key_len=st.integers(1, 8),
    d_k=st.integers(1, 12),
)
def test_attention_is_scale_invariant_above_documented_boundary(batch, query_len, key_len, d_k):
    torch.manual_seed(1)
    q = torch.randn(batch, query_len, d_k) * 0.1
    k = torch.randn(batch, key_len, d_k) * 0.1
    assume(q.norm(dim=-1).min() >= 1e-2)
    assume(k.norm(dim=-1).min() >= 1e-2)
    v = torch.randn(batch, key_len, 5)
    attention = NonCollapsingAttention(d_k)
    torch.testing.assert_close(
        attention(q, k, v), attention(q * 10.0, k * 10.0, v), atol=3e-2, rtol=3e-2
    )


@given(
    batch=st.integers(1, 3),
    query_len=st.integers(1, 8),
    key_len=st.integers(1, 8),
    d_k=st.integers(1, 12),
)
def test_attention_is_equivariant_to_key_value_permutation(batch, query_len, key_len, d_k):
    torch.manual_seed(2)
    q = torch.randn(batch, query_len, d_k)
    k = torch.randn(batch, key_len, d_k)
    v = torch.randn(batch, key_len, 5)
    permutation = torch.randperm(key_len)
    attention = NonCollapsingAttention(d_k)
    torch.testing.assert_close(
        attention(q, k, v), attention(q, k[:, permutation], v[:, permutation])
    )


def test_attention_gradcheck_in_float64_including_temperature():
    torch.manual_seed(7)
    attention = NonCollapsingAttention(3).double()
    q = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 3, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    raw_tau = attention.raw_tau.detach().clone().requires_grad_()

    def operation(q_input, k_input, v_input, raw):
        logits = (_unit(q_input) @ _unit(k_input).transpose(-2, -1)) * F.softplus(raw).clamp(
            1.0, 50.0
        )
        return F.softmax(logits, dim=-1) @ v_input

    assert torch.autograd.gradcheck(operation, (q, k, v, raw_tau), eps=1e-6, atol=1e-5)


@given(
    floor=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    total=st.integers(min_value=1, max_value=40),
    warmup=st.integers(min_value=0, max_value=39),
)
def test_scheduler_properties_for_valid_configurations(floor, total, warmup):
    assume(warmup < total)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = SGD([parameter], lr=0.1)
    scheduler = AsymptoticLRScheduler(
        optimizer, floor_ratio=floor, total_steps=total, warmup_steps=warmup
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(total + 3):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert all(math.isfinite(rate) and 0 <= rate <= 0.1 for rate in rates)
    assert rates[total] == pytest.approx(0.1 * floor)
    assert all(rate == pytest.approx(0.1 * floor) for rate in rates[total:])
    assert all(left <= right for left, right in zip(rates[:warmup], rates[1 : warmup + 1]))
    assert all(left >= right for left, right in zip(rates[warmup:], rates[warmup + 1 :]))


@st.composite
def bounded_image_shapes(draw):
    height = draw(st.integers(min_value=1, max_value=2048))
    width = draw(st.integers(min_value=1, max_value=min(2048, 131072 // height)))
    return height, width


@given(shape=bounded_image_shapes())
def test_isotropic_coordinates_properties(shape):
    height, width = shape
    coords = get_isotropic_coords(height, width, dtype=torch.float64)
    assert coords.dtype == torch.float64
    assert torch.isfinite(coords).all()
    assert coords.mean(dim=(0, 1)).abs().max() < 1e-12
    radius = get_scale_invariant_vit_coords(height, width)[..., 0]
    assert radius.max() <= math.sqrt(0.5) + 1e-6
    if height > 1 and width > 1:
        x_step = coords[0, 1, 0] - coords[0, 0, 0]
        y_step = coords[1, 0, 1] - coords[0, 0, 1]
        assert float(x_step) == pytest.approx(float(y_step), abs=1e-12)


@given(height=st.integers(1, 32), width=st.integers(1, 32))
def test_coordinate_grid_pooling_is_resolution_consistent(height, width):
    fine = get_isotropic_coords(2 * height, 2 * width).permute(2, 0, 1)[None]
    pooled = F.avg_pool2d(fine, 2)[0].permute(1, 2, 0)
    torch.testing.assert_close(pooled, get_isotropic_coords(height, width), atol=1e-6, rtol=1e-6)


@given(size=st.integers(min_value=16, max_value=256))
def test_smooth_angle_has_no_branch_cut_jump_away_from_origin(size):
    coords = get_scale_invariant_vit_coords(size, size, smooth_angle=True)
    gap = torch.linalg.vector_norm(coords[size // 2 - 1, 0, 1:] - coords[size // 2, 0, 1:])
    pixel_step = 1.0 / size
    assert gap <= 2.2 * pixel_step


@given(
    in_features=st.integers(1, 64),
    out_features=st.integers(1, 64),
    nonlinearity=st.sampled_from(["linear", "relu", "leaky_relu", "tanh"]),
)
def test_linear_orthogonality_matches_requested_gain(in_features, out_features, nonlinearity):
    torch.manual_seed(3)
    layer = make_corrected_linear(in_features, out_features, nonlinearity=nonlinearity)
    gain = torch.nn.init.calculate_gain(nonlinearity)
    gram = (
        layer.weight.T @ layer.weight
        if out_features >= in_features
        else layer.weight @ layer.weight.T
    )
    torch.testing.assert_close(
        gram,
        torch.eye(min(in_features, out_features)) * gain**2,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("nonlinearity", "activation", "bounds"),
    [
        ("relu", F.relu, {10: (0.48, 1.96), 30: (0.67, 2.70), 100: (0.27, 1.10)}),
        (
            "leaky_relu",
            lambda x: F.leaky_relu(x, negative_slope=0.01),
            {10: (0.48, 1.96), 30: (0.67, 2.72), 100: (0.27, 1.10)},
        ),
        ("tanh", torch.tanh, {10: (0.32, 1.31), 30: (0.32, 1.31), 100: (0.32, 1.31)}),
        ("linear", lambda x: x, {10: (0.5, 2.0), 30: (0.5, 2.0), 100: (0.5, 2.0)}),
    ],
)
@pytest.mark.parametrize("depth", [10, 30, 100])
def test_deep_stack_norm_ratios_stay_within_measured_twofold_bounds(
    nonlinearity, activation, bounds, depth
):
    torch.manual_seed(100 + depth)
    sample = torch.randn(512, 256)
    initial_norm = sample.norm(dim=1).mean()
    for _ in range(depth):
        sample = activation(make_corrected_linear(256, 256, nonlinearity=nonlinearity)(sample))
    ratio = (sample.norm(dim=1).mean() / initial_norm).item()
    low, high = bounds[depth]
    assert low <= ratio <= high


def test_rectangular_orthogonal_projection_has_expected_average_norm_ratio():
    torch.manual_seed(29)
    sample = torch.randn(4096, 64)
    layer = make_corrected_linear(64, 16, nonlinearity="linear")
    ratio = (layer(sample).norm(dim=1).mean() / sample.norm(dim=1).mean()).item()
    assert ratio == pytest.approx(math.sqrt(16 / 64), abs=0.02)


def test_scheduler_state_round_trip_and_multiple_param_groups():
    p1, p2 = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    optimizer = SGD([{"params": [p1], "lr": 0.1}, {"params": [p2], "lr": 0.01}])
    scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=0.2, total_steps=8, warmup_steps=2)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    clone_optimizer = SGD([{"params": [p1], "lr": 0.1}, {"params": [p2], "lr": 0.01}])
    clone_scheduler = AsymptoticLRScheduler(
        clone_optimizer, floor_ratio=0.2, total_steps=8, warmup_steps=2
    )
    clone_optimizer.load_state_dict(optimizer_state)
    clone_scheduler.load_state_dict(scheduler_state)
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        clone_optimizer.step()
        clone_scheduler.step()
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
            [group["lr"] for group in clone_optimizer.param_groups]
        )


def test_scheduler_can_resume_from_last_epoch():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = SGD([parameter], lr=0.1)
    scheduler = AsymptoticLRScheduler(optimizer, total_steps=12, warmup_steps=3)
    for _ in range(6):
        optimizer.step()
        scheduler.step()
    resumed_optimizer = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    resumed_optimizer.param_groups[0]["initial_lr"] = 0.1
    resumed = AsymptoticLRScheduler(
        resumed_optimizer, total_steps=12, warmup_steps=3, last_epoch=scheduler.last_epoch - 1
    )
    assert resumed.last_epoch == scheduler.last_epoch
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        resumed_optimizer.step()
        resumed.step()
        assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
            optimizer.param_groups[0]["lr"]
        )


def test_attention_state_dict_deepcopy_pickle_and_compile_smoke():
    attention = NonCollapsingAttention(4)
    q, k, v = (torch.randn(1, 3, 4) for _ in range(3))
    expected = attention(q, k, v)

    loaded = NonCollapsingAttention(4)
    loaded.load_state_dict(attention.state_dict())
    torch.testing.assert_close(loaded(q, k, v), expected)
    torch.testing.assert_close(copy.deepcopy(attention)(q, k, v), expected)
    torch.testing.assert_close(pickle.loads(pickle.dumps(attention))(q, k, v), expected)

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable in this PyTorch version")
    try:
        compiled = torch.compile(attention, backend="eager")
        torch.testing.assert_close(compiled(q, k, v), expected)
    except RuntimeError as error:
        if "not yet supported for torch.compile" in str(error):
            pytest.skip(f"torch.compile is unsupported by this torch/Python combination: {error}")
        raise


def test_attention_is_deterministic_with_deterministic_algorithms_enabled():
    attention = NonCollapsingAttention(8)
    q, k, v = (torch.randn(2, 4, 8) for _ in range(3))
    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        first = attention(q, k, v)
        second = attention(q, k, v)
        torch.testing.assert_close(first, second)
    finally:
        torch.use_deterministic_algorithms(old)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_attention_cuda_smoke():
    attention = NonCollapsingAttention(8).cuda()
    q, k, v = (torch.randn(2, 4, 8, device="cuda") for _ in range(3))
    assert torch.isfinite(attention(q, k, v)).all()
