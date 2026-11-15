"""Follow-up regression tests for gaps found by mutation testing and an audit."""

import pytest
import torch
import torch.nn as nn

from zeta_scaling_toolkit import NonCollapsingAttention, calibrate_gain, make_corrected_linear


@pytest.mark.parametrize("scale", [1e10, 1e20, 1e30, 1e37])
def test_huge_finite_inputs_match_normal_scale(scale):
    """Without the overflow rescale, x**2 overflows, unit vectors become 0 and attention
    silently turns uniform while the output stays finite."""
    torch.manual_seed(0)
    attn = NonCollapsingAttention(16)
    q, k, v = (torch.randn(2, 8, 16) for _ in range(3))
    expected = attn(q, k, v)
    actual = attn(q * scale, k * scale, v)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_float16_near_max_inputs_match_normal_scale():
    torch.manual_seed(0)
    attn = NonCollapsingAttention(16)
    q, k = (torch.randn(1, 6, 16) for _ in range(2))
    v = torch.randn(1, 6, 16).half()
    big_q, big_k = (q * 1e4).half(), (k * 1e4).half()
    assert torch.isfinite(big_q).all() and torch.isfinite(big_k).all()
    actual = attn(big_q, big_k, v)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, attn(q, k, v.float()).half(), atol=2e-2, rtol=2e-2)


def test_temperature_is_sigmoid_bounded_in_range():
    """Pins the parameterisation (sigmoid with bounds; the raw parameter is logit-scaled)."""
    attn = NonCollapsingAttention(8, tau0=1.0, tau_min=0.0, tau_max=50.0)
    for raw in (-2.0, 0.0, 1.5):
        with torch.no_grad():
            attn.raw_tau.fill_(raw)
        sig = torch.sigmoid(torch.tensor(raw)).item()
        expected_tau = 0.0 + sig * (50.0 - 0.0)
        assert attn.tau.item() == pytest.approx(expected_tau, rel=1e-5)
    with torch.no_grad():
        attn.raw_tau.fill_(-2.0)
    attn.tau.backward()
    assert attn.raw_tau.grad is not None
    sig_grad = (
        torch.sigmoid(torch.tensor(-2.0)).item()
        * (1.0 - torch.sigmoid(torch.tensor(-2.0)).item())
        * (50.0 - 0.0)
    )
    assert attn.raw_tau.grad.item() == pytest.approx(sig_grad, rel=1e-4)


@pytest.mark.parametrize("tau0", [1.0, 7.0, 10.0, 49.0])
def test_initial_temperature_equals_tau0(tau0):
    assert NonCollapsingAttention(8, tau0=tau0).tau.item() == pytest.approx(tau0, rel=1e-4)


def test_temperature_respects_both_configured_bounds():
    attn = NonCollapsingAttention(8, tau0=5.0, tau_min=2.0, tau_max=10.0)
    with torch.no_grad():
        attn.raw_tau.fill_(100.0)
    assert attn.tau.item() == pytest.approx(10.0)
    with torch.no_grad():
        attn.raw_tau.fill_(-100.0)
    assert attn.tau.item() == pytest.approx(2.0)


def _stack(act, depth=30, width=64):
    layers = []
    for _ in range(depth):
        layers += [make_corrected_linear(width, width, nonlinearity="linear"), act()]
    return nn.Sequential(*layers)


@pytest.mark.parametrize("act", [nn.GELU, nn.SiLU, nn.ReLU])
def test_calibrate_gain_hits_the_batch_norm_target(act):
    torch.manual_seed(0)
    x = torch.randn(256, 64)
    net = _stack(act)
    calibrate_gain(net, x)
    with torch.no_grad():
        assert (net(x).norm() / x.norm()).item() == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("act", [nn.GELU, nn.SiLU, nn.ReLU])
@pytest.mark.parametrize("depth", [10, 30, 100])
def test_calibrate_gain_is_finite_and_hits_target_on_deep_stacks(act, depth):
    torch.manual_seed(0)
    x = torch.randn(256, 64)
    net = _stack(act, depth=depth)

    calibrate_gain(net, x)

    with torch.no_grad():
        output = net(x)
        ratio = output.norm() / x.norm()
    assert all(torch.isfinite(parameter).all() for parameter in net.parameters())
    assert torch.isfinite(output).all()
    assert ratio.item() == pytest.approx(1.0, rel=0.02)


def test_calibrate_gain_restores_module_modes_and_rng_on_success_and_failure():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    net.train()
    net[1].eval()
    x = torch.randn(16, 4)
    modes = [module.training for module in net.modules()]
    rng = torch.random.get_rng_state()

    calibrate_gain(net, x)

    assert [module.training for module in net.modules()] == modes
    assert torch.equal(torch.random.get_rng_state(), rng)

    with torch.no_grad():
        net[-1].weight.fill_(1e-40)
        net[-1].bias.zero_()
    rng = torch.random.get_rng_state()
    with pytest.raises(ValueError, match="non-finite"):
        calibrate_gain(net, x)
    assert [module.training for module in net.modules()] == modes
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(torch.isfinite(parameter).all() for parameter in net.parameters())


def test_calibrate_gain_warns_when_the_target_is_unreachable():
    """tanh is bounded, so a norm ratio of 1.0 cannot be reached; this used to pass silently."""
    torch.manual_seed(0)
    x = torch.randn(256, 64)
    with pytest.warns(RuntimeWarning, match="target_ratio"):
        calibrate_gain(_stack(nn.Tanh), x)


def test_calibrate_gain_is_quiet_when_the_target_is_reached():
    import warnings

    torch.manual_seed(0)
    x = torch.randn(256, 64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calibrate_gain(_stack(nn.ReLU), x)


def test_calibrate_gain_accepts_module_sequences_with_a_prefix_module():
    torch.manual_seed(0)
    layers = (nn.ReLU(), nn.Linear(4, 4))
    sample = torch.randn(16, 4)
    assert calibrate_gain(layers, sample, iters=1) is layers


def test_calibrate_gain_rejects_overflowing_scale():
    weight_overflow = nn.Sequential(nn.Linear(2, 2))
    with torch.no_grad():
        weight_overflow[0].weight.fill_(1e-38)
        weight_overflow[0].bias.zero_()
    with pytest.raises(ValueError, match="non-finite or non-positive scale"):
        calibrate_gain(weight_overflow, torch.ones(2, 2), target_ratio=1e308)


def test_known_limitation_silu_calibration_does_not_equalise_per_row_norms():
    """Characterisation test: the batch norm is preserved but the median row collapses.
    If this starts failing, the code improved: update the README limitation."""
    torch.manual_seed(0)
    x = torch.randn(256, 64)
    net = _stack(nn.SiLU)
    calibrate_gain(net, x)
    with torch.no_grad():
        row_ratio = net(x).norm(dim=1) / x.norm(dim=1)
    assert row_ratio.median().item() < 0.5
