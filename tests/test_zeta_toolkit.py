"""Adversarial regression tests: each one pins a failure found in the v0.1 audit."""

import math
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch.optim import SGD

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    get_isotropic_coords,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)


# ---------- attention ----------
def test_attention_matches_reference_formula():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 5, 16) for _ in range(3))
    attn = NonCollapsingAttention(16)
    logits = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1) * attn.tau
    torch.testing.assert_close(attn(q, k, v), F.softmax(logits, -1) @ v, atol=1e-5, rtol=1e-5)


def test_attention_is_not_near_uniform():
    torch.manual_seed(0)
    k = torch.randn(1, 128, 64)
    w = F.softmax((F.normalize(k, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * NonCollapsingAttention(64).tau, -1)
    assert w[0].diagonal().mean() > 0.9  # v0.1 gave 0.0088 vs 0.0078 uniform


def test_temperature_is_capped_even_if_parameter_explodes():
    attn = NonCollapsingAttention(8, tau_max=50.0)
    attn.raw_tau.data.fill_(1e6)
    assert attn.tau.item() == 50.0


def test_temperature_trains():
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 3, 4) for _ in range(3))
    attn = NonCollapsingAttention(4)
    attn(q, k, v).square().sum().backward()
    assert attn.raw_tau.grad is not None and attn.raw_tau.grad.abs().item() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_zero_vectors_finite_in_all_dtypes_and_gradient_bounded(dtype):
    attn = NonCollapsingAttention(8)
    q = torch.zeros(1, 4, 8, dtype=dtype, requires_grad=True)
    k, v = (torch.randn(1, 4, 8).to(dtype) for _ in range(2))
    out = attn(q, k, v)
    assert torch.isfinite(out).all()
    out.float().sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.float().abs().max() < 1e5  # v0.1: ~1e12


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        NonCollapsingAttention(32)(*(torch.randn(1, 4, 64) for _ in range(3)))


def test_input_scale_invariance():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 16, 32) for _ in range(3))
    attn = NonCollapsingAttention(32)
    torch.testing.assert_close(attn(q, k, v), attn(q * 1e3, k * 1e-1, v), atol=1e-4, rtol=1e-4)  # holds for ||x|| >~ 1e-2


def test_mask_blocks_positions():
    q, k = torch.randn(1, 2, 8), torch.randn(1, 3, 8)
    v = torch.tensor([[[1.0] * 8, [2.0] * 8, [100.0] * 8]])
    mask = torch.tensor([[[True, True, False], [True, True, False]]])
    assert NonCollapsingAttention(8)(q, k, v, attn_mask=mask).max() < 3


def test_legacy_args_warn_and_do_nothing():
    with pytest.warns(DeprecationWarning):
        attn = NonCollapsingAttention(4, A_attn=2.5)
    q, k, v = (torch.randn(1, 3, 4) for _ in range(3))
    before = attn(q, k, v)
    attn.A_attn = 0.0  # benchmark scripts assign this
    torch.testing.assert_close(attn(q, k, v), before)


def test_legacy_checkpoint_loads():
    attn = NonCollapsingAttention(4)
    sd = {"q_scale": torch.ones(1), "k_scale": torch.ones(1)}
    with pytest.warns(UserWarning):
        attn.load_state_dict(sd)


def test_invalid_config_rejected():
    for kw in ({"tau0": 0.0}, {"tau0": 60.0}, {"tau_max": -1.0}, {"tau_min": 0.0}, {"tau_min": 2.0, "tau0": 1.0}):
        with pytest.raises(ValueError):
            NonCollapsingAttention(8, **kw)


# ---------- scheduler ----------
def _lrs(n, base=1.0, **kw):
    opt = SGD([torch.nn.Parameter(torch.zeros(1))], lr=base)
    sched = AsymptoticLRScheduler(opt, **kw)
    out = []
    for _ in range(n):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def test_starts_at_base_decays_to_floor_and_clamps():
    lrs = _lrs(10, base=0.1, floor_ratio=0.5, total_steps=4)
    assert lrs[0] == pytest.approx(0.1)
    for step in range(10):
        expected = 0.1 * (0.5 + 0.5 * (1 + math.cos(math.pi * min(step, 4) / 4)) / 2)
        assert lrs[step] == pytest.approx(expected)
    assert lrs[-1] == pytest.approx(0.05)


def test_monotone_without_warmup():
    lrs = _lrs(400, total_steps=300)
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_warmup_ramps_then_peaks_at_base():
    lrs = _lrs(60, total_steps=50, warmup_steps=10, floor_ratio=0.1)
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0) and lrs[10] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.1)
    assert all(a <= b for a, b in zip(lrs[:10], lrs[1:10]))


@pytest.mark.parametrize(
    "kw",
    [{"floor_ratio": 1.5}, {"floor_ratio": -0.1}, {"total_steps": 0}, {"total_steps": -5}, {"warmup_steps": 300}, {"warmup_steps": -1}],
)
def test_scheduler_rejects_bad_config(kw):
    with pytest.raises(ValueError):
        AsymptoticLRScheduler(SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0), **kw)


# ---------- coordinates ----------
def test_aspect_ratio_preserved():
    c = get_isotropic_coords(224, 448)
    assert (c[112, 274, 0] - c[112, 224, 0]).item() == pytest.approx((c[162, 224, 1] - c[112, 224, 1]).item())  # v0.1: 2.0x


def test_grid_is_centred_and_radius_bounded():
    for h, w in [(224, 224), (224, 448), (7, 13)]:
        c = get_isotropic_coords(h, w)
        assert c.mean(dim=(0, 1)).abs().max() < 1e-6  # v0.1: -0.0022
        assert get_scale_invariant_vit_coords(h, w)[..., 0].max() <= math.sqrt(0.5) + 1e-6


def test_resolution_consistency_by_pooling():
    fine = get_isotropic_coords(8, 12).permute(2, 0, 1)[None]
    pooled = F.avg_pool2d(fine, 2)[0].permute(1, 2, 0)
    torch.testing.assert_close(pooled, get_isotropic_coords(4, 6), atol=1e-6, rtol=1e-6)


def test_smooth_angle_is_continuous_across_the_branch_cut():
    raw = get_scale_invariant_vit_coords(32, 32)
    smooth = get_scale_invariant_vit_coords(32, 32, smooth_angle=True)
    assert smooth.shape == (32, 32, 3)
    # rows 15 and 16 straddle y=0; on the left half they sit either side of theta = +-pi
    assert (raw[15, :10, 1] - raw[16, :10, 1]).abs().min() > 6.0  # raw angle jumps ~2*pi
    assert (smooth[15, :10, 1:] - smooth[16, :10, 1:]).abs().max() < 0.2  # (cos, sin) moves only by the true pixel spacing


def test_deprecated_coord_args_warn_and_do_nothing():
    with pytest.warns(DeprecationWarning):
        a = get_scale_invariant_vit_coords(8, 8, T=1e9, A_coords=5)
    torch.testing.assert_close(a, get_scale_invariant_vit_coords(8, 8))


def test_invalid_coord_size_rejected():
    with pytest.raises(ValueError):
        get_isotropic_coords(0, 4)


# ---------- linear init ----------
@pytest.mark.parametrize("in_f,out_f", [(8, 16), (16, 8)])
def test_orthogonality_scaled_by_gain(in_f, out_f):
    torch.manual_seed(0)
    w = make_corrected_linear(in_f, out_f, nonlinearity="linear").weight
    gram = w.T @ w if out_f >= in_f else w @ w.T
    torch.testing.assert_close(gram, torch.eye(min(in_f, out_f)), atol=1e-6, rtol=1e-6)
    r = make_corrected_linear(in_f, out_f).weight
    gram = r.T @ r if out_f >= in_f else r @ r.T
    torch.testing.assert_close(gram, 2 * torch.eye(min(in_f, out_f)), atol=1e-5, rtol=1e-5)


def test_deep_relu_stack_keeps_norm():
    torch.manual_seed(0)
    x = torch.randn(512, 256)
    n0 = x.norm(dim=1).mean()
    for _ in range(30):
        x = F.relu(make_corrected_linear(256, 256)(x))
    assert 0.2 < (x.norm(dim=1).mean() / n0).item() < 5.0  # v0.1: 2e-5


def test_deep_linear_stack_is_isometric():
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    n0 = x.norm()
    for _ in range(30):
        x = make_corrected_linear(128, 128, nonlinearity="linear")(x)
    assert x.norm().item() / n0.item() == pytest.approx(1.0, abs=1e-4)


def test_no_bias_and_bad_nonlinearity():
    assert make_corrected_linear(8, 16, bias=False).bias is None
    with pytest.raises(ValueError):
        make_corrected_linear(8, 8, nonlinearity="nope")


def test_tau_stays_within_bounds_under_aggressive_optimization():
    """tau_min should prevent tau from decaying to near-zero (uniform attention)."""
    attn = NonCollapsingAttention(8, tau0=2.0, tau_min=1.0, tau_max=50.0)
    opt = torch.optim.Adam([attn.raw_tau], lr=0.5)
    q, k, v = torch.randn(1, 16, 8), torch.randn(1, 16, 8), torch.ones(1, 16, 8)
    for _ in range(1000):
        opt.zero_grad()
        out = attn(q, k, v)
        loss = (out - 0.5).pow(2).mean()
        loss.backward()
        opt.step()
    assert 1.0 <= attn.tau.item() <= 50.0, f"tau escaped bounds: {attn.tau.item()}"


def test_gelu_silu_nonlinearity_rejected():
    """make_corrected_linear should reject GELU/SiLU and suggest calibrate_gain."""
    for nl in ("gelu", "silu"):
        with pytest.raises(ValueError, match="calibrate_gain"):
            make_corrected_linear(8, 8, nonlinearity=nl)


def test_supported_nonlinearities_accepted():
    """make_corrected_linear should accept all torch.nn.init.calculate_gain nonlinearities."""
    for nl in ("linear", "relu", "leaky_relu", "tanh", "sigmoid", "selu", "conv1d"):
        layer = make_corrected_linear(8, 8, nonlinearity=nl)
        assert layer.weight is not None


def test_calibrate_gain_gelu_stack_converges():
    """calibrate_gain should bring a GELU stack norm ratio to near target (1.0)."""
    from zeta_scaling_toolkit import calibrate_gain

    torch.manual_seed(42)
    model = torch.nn.Sequential(
        make_corrected_linear(64, 64, nonlinearity="linear"),
        torch.nn.GELU(),
        make_corrected_linear(64, 64, nonlinearity="linear"),
        torch.nn.GELU(),
        make_corrected_linear(64, 64, nonlinearity="linear"),
        torch.nn.GELU(),
    )
    x = torch.randn(32, 64)
    ratio = calibrate_gain(model, x, target_ratio=1.0, iters=5, nonlinearity=torch.nn.GELU())
    assert 0.5 <= ratio <= 2.0, f"calibrate_gain failed to converge: ratio={ratio}"


def test_calibrate_gain_silu_stack_converges():
    """calibrate_gain should also work for SiLU (GELU-like instability)."""
    from zeta_scaling_toolkit import calibrate_gain

    torch.manual_seed(42)
    model = torch.nn.Sequential(
        make_corrected_linear(64, 64, nonlinearity="linear"),
        torch.nn.SiLU(),
        make_corrected_linear(64, 64, nonlinearity="linear"),
        torch.nn.SiLU(),
        make_corrected_linear(64, 64, nonlinearity="linear"),
        torch.nn.SiLU(),
    )
    x = torch.randn(32, 64)
    ratio = calibrate_gain(model, x, target_ratio=1.0, iters=5, nonlinearity=torch.nn.SiLU())
    assert 0.5 <= ratio <= 3.0, f"calibrate_gain failed for SiLU: ratio={ratio}"


def test_calibrate_gain_rejects_grad_input():
    """calibrate_gain should reject sample_input with requires_grad=True."""
    from zeta_scaling_toolkit import calibrate_gain

    model = torch.nn.Linear(32, 32)
    x = torch.randn(8, 32, requires_grad=True)
    with pytest.raises(ValueError, match="must not require grad"):
        calibrate_gain(model, x)


def test_fully_masked_attention_row_returns_zeros_not_nan():
    """Fully masked rows should return zeros, not NaN."""
    attn = NonCollapsingAttention(8)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    # First row fully masked (all False)
    mask = torch.tensor([[[False, False, False, False], [True, True, True, True], [True, True, True, True], [True, True, True, True]]])
    out = attn(q, k, v, attn_mask=mask)
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert (out[0, 0] == 0).all(), "First (fully masked) row should be zeros"


def test_fully_masked_attention_row_gradients_are_finite():
    """Fully masked rows should still have finite gradients."""
    attn = NonCollapsingAttention(8)
    q = torch.randn(1, 4, 8, requires_grad=True)
    k = torch.randn(1, 4, 8, requires_grad=True)
    v = torch.randn(1, 4, 8, requires_grad=True)
    mask = torch.tensor([[[False, False, False, False], [True, True, True, True], [True, True, True, True], [True, True, True, True]]])
    out = attn(q, k, v, attn_mask=mask)
    out.sum().backward()
    assert torch.isfinite(q.grad).all() and (q.grad.abs() < 1e5).all(), "q gradient not bounded"
    assert torch.isfinite(k.grad).all() and (k.grad.abs() < 1e5).all(), "k gradient not bounded"
    assert torch.isfinite(v.grad).all() and (v.grad.abs() < 1e5).all(), "v gradient not bounded"
