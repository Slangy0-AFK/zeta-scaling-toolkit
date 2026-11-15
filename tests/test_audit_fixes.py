"""Regression tests for audit fixes: temperature initialization, calibrate_gain atomicity, etc."""

import pytest
import torch
import torch.nn as nn

from zeta_scaling_toolkit import NonCollapsingAttention, calibrate_gain

# ==============================================================================
# Fix #1: Temperature initialization validation and division by zero
# ==============================================================================


def test_temperature_tau_min_equals_tau_max_raises():
    """tau_min == tau_max is not supported; must raise ValueError."""
    with pytest.raises(ValueError, match="tau_min must be strictly less than tau_max"):
        NonCollapsingAttention(8, tau0=5.0, tau_min=5.0, tau_max=5.0)


def test_temperature_tau_min_greater_than_tau_max_raises():
    """tau_min > tau_max violates ordering; must raise ValueError."""
    with pytest.raises(ValueError, match="need 0 <= tau_min <= tau0 <= tau_max"):
        NonCollapsingAttention(8, tau0=5.0, tau_min=10.0, tau_max=5.0)


def test_temperature_tau0_less_than_tau_min_raises():
    """tau0 < tau_min violates range; must raise ValueError."""
    with pytest.raises(ValueError, match="need 0 <= tau_min <= tau0 <= tau_max"):
        NonCollapsingAttention(8, tau0=0.5, tau_min=1.0, tau_max=50.0)


def test_temperature_tau0_greater_than_tau_max_raises():
    """tau0 > tau_max violates range; must raise ValueError."""
    with pytest.raises(ValueError, match="need 0 <= tau_min <= tau0 <= tau_max"):
        NonCollapsingAttention(8, tau0=60.0, tau_min=1.0, tau_max=50.0)


def test_temperature_nan_tau0_raises():
    """NaN tau0 must be rejected."""
    with pytest.raises(ValueError, match="temperature values must be finite"):
        NonCollapsingAttention(8, tau0=float("nan"), tau_min=1.0, tau_max=50.0)


def test_temperature_nan_tau_min_raises():
    """NaN tau_min must be rejected."""
    with pytest.raises(ValueError, match="temperature values must be finite"):
        NonCollapsingAttention(8, tau0=10.0, tau_min=float("nan"), tau_max=50.0)


def test_temperature_nan_tau_max_raises():
    """NaN tau_max must be rejected."""
    with pytest.raises(ValueError, match="temperature values must be finite"):
        NonCollapsingAttention(8, tau0=10.0, tau_min=1.0, tau_max=float("nan"))


def test_temperature_inf_tau0_raises():
    """Infinite tau0 must be rejected."""
    with pytest.raises(ValueError, match="temperature values must be finite"):
        NonCollapsingAttention(8, tau0=float("inf"), tau_min=1.0, tau_max=50.0)


def test_temperature_normal_initialization():
    """Normal temperature initialization should work correctly."""
    attn = NonCollapsingAttention(8, tau0=10.0, tau_min=1.0, tau_max=50.0)
    assert attn.tau.item() == pytest.approx(10.0, rel=1e-4)


def test_temperature_at_lower_bound():
    """tau0 equal to tau_min should produce tau at lower bound."""
    attn = NonCollapsingAttention(8, tau0=1.0, tau_min=1.0, tau_max=50.0)
    assert attn.tau.item() == pytest.approx(1.0, rel=1e-4)


def test_temperature_at_upper_bound():
    """tau0 equal to tau_max should produce tau at upper bound."""
    attn = NonCollapsingAttention(8, tau0=50.0, tau_min=1.0, tau_max=50.0)
    assert attn.tau.item() == pytest.approx(50.0, rel=1e-4)


def test_temperature_forward_pass_respects_bounds():
    """Forward pass temperature must always stay within bounds."""
    attn = NonCollapsingAttention(8, tau0=10.0, tau_min=1.0, tau_max=50.0)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    output = attn(q, k, v)
    tau = attn.tau.item()
    assert 1.0 <= tau <= 50.0
    assert output.shape == q.shape


def test_temperature_gradient_at_lower_bound():
    """Temperature should have non-zero gradient when at lower bound."""
    attn = NonCollapsingAttention(8, tau0=1.0, tau_min=1.0, tau_max=50.0)
    tau = attn.tau
    tau.sum().backward()
    assert attn.raw_tau.grad is not None
    assert attn.raw_tau.grad.abs().item() > 0


def test_temperature_gradient_at_upper_bound():
    """Temperature should have non-zero gradient when at upper bound."""
    attn = NonCollapsingAttention(8, tau0=50.0, tau_min=1.0, tau_max=50.0)
    tau = attn.tau
    tau.sum().backward()
    assert attn.raw_tau.grad is not None
    assert attn.raw_tau.grad.abs().item() > 0


# ==============================================================================
# Fix #2: calibrate_gain atomicity (no partial mutations on failure)
# ==============================================================================


def test_calibrate_gain_reverts_all_parameters_on_failure():
    """If calibration fails on any layer, all parameters must revert to original."""
    layers = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
    )

    linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
    original_weights = [layer.weight.clone() for layer in linear_layers]
    original_biases = [
        layer.bias.clone() if layer.bias is not None else None for layer in linear_layers
    ]

    # Use zero input to force failure
    sample_input = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="cannot calibrate|zero or non-finite"):
        calibrate_gain(layers, sample_input, target_ratio=1.0, iters=1)

    for i, linear_layer in enumerate(linear_layers):
        torch.testing.assert_close(linear_layer.weight, original_weights[i])
        if original_biases[i] is not None:
            torch.testing.assert_close(linear_layer.bias, original_biases[i])


def test_calibrate_gain_successful_calibration_still_works():
    """Successful calibration should still modify parameters."""
    layers = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
    )

    linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
    original_weights = [layer.weight.clone() for layer in linear_layers]

    sample_input = torch.randn(8, 4)
    result = calibrate_gain(layers, sample_input, target_ratio=1.0, iters=2)

    weights_changed = False
    for i, linear_layer in enumerate(linear_layers):
        if not torch.allclose(linear_layer.weight, original_weights[i]):
            weights_changed = True
            break

    assert weights_changed, "calibrate_gain should modify weights for valid inputs"
    assert result is layers


def test_calibrate_gain_zero_norm_input_reverts():
    """Zero-norm input should cause reversion without modifying parameters."""
    layers = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
    )

    linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
    original_weights = [layer.weight.clone() for layer in linear_layers]

    sample_input = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="cannot calibrate|zero or non-finite"):
        calibrate_gain(layers, sample_input, target_ratio=1.0, iters=1)

    for i, linear_layer in enumerate(linear_layers):
        torch.testing.assert_close(linear_layer.weight, original_weights[i])


def test_calibrate_gain_nan_activation_reverts():
    """NaN activation should cause reversion without modifying parameters."""
    layers = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
    )

    linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
    original_weights = [layer.weight.clone() for layer in linear_layers]

    sample_input = torch.full((2, 4), float("inf"))
    with pytest.raises(ValueError, match="cannot calibrate|zero or non-finite|non-finite"):
        calibrate_gain(layers, sample_input, target_ratio=1.0, iters=1)

    for i, linear_layer in enumerate(linear_layers):
        torch.testing.assert_close(linear_layer.weight, original_weights[i])


def test_calibrate_gain_module_list_atomicity():
    """calibrate_gain on ModuleList should also be atomic."""
    layers = nn.ModuleList(
        [
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        ]
    )

    linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
    original_weights = [layer.weight.clone() for layer in linear_layers]

    # Use zero input to force failure
    sample_input = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="cannot calibrate|zero or non-finite"):
        calibrate_gain(layers, sample_input, target_ratio=1.0, iters=1)

    for i, linear_layer in enumerate(linear_layers):
        torch.testing.assert_close(linear_layer.weight, original_weights[i])


# ==============================================================================
# Fix #3: Temperature gradient stress tests
# ==============================================================================


@pytest.mark.parametrize("raw_tau_value", [-1000, -100, -20, -10, 0, 10, 20, 100, 1000])
def test_temperature_gradient_at_extreme_raw_tau(raw_tau_value):
    """Temperature should produce finite output and gradient at extreme raw_tau values."""
    attn = NonCollapsingAttention(8, tau0=10.0, tau_min=1.0, tau_max=50.0)

    with torch.no_grad():
        attn.raw_tau.fill_(float(raw_tau_value))

    tau = attn.tau
    assert torch.isfinite(tau).all(), f"Non-finite tau at raw_tau={raw_tau_value}"
    assert 1.0 <= tau.item() <= 50.0, f"Tau out of bounds at raw_tau={raw_tau_value}: {tau.item()}"

    tau.sum().backward()
    if attn.raw_tau.grad is not None:
        assert torch.isfinite(attn.raw_tau.grad).all(), (
            f"Non-finite gradient at raw_tau={raw_tau_value}"
        )


def test_temperature_forward_pass_with_extreme_raw_tau():
    """Forward pass should remain stable with extreme raw_tau values."""
    attn = NonCollapsingAttention(8, tau0=10.0, tau_min=1.0, tau_max=50.0)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)

    for raw_tau_value in [-1000, 0, 1000]:
        with torch.no_grad():
            attn.raw_tau.fill_(float(raw_tau_value))

        output = attn(q, k, v)
        assert torch.isfinite(output).all(), f"Non-finite output at raw_tau={raw_tau_value}"
        assert output.shape == q.shape


def test_temperature_recovery_after_extreme_perturbation():
    """Temperature should have gradient enabling recovery after extreme perturbations."""
    attn = NonCollapsingAttention(8, tau0=10.0, tau_min=1.0, tau_max=50.0)

    with torch.no_grad():
        attn.raw_tau.fill_(-20.0)

    tau1 = attn.tau.item()
    attn.zero_grad()
    tau1_tensor = attn.tau
    tau1_tensor.sum().backward()
    attn.raw_tau.grad.item() if attn.raw_tau.grad is not None else 0.0

    with torch.no_grad():
        attn.raw_tau.fill_(-10.0)

    tau2 = attn.tau.item()
    attn.zero_grad()
    tau2_tensor = attn.tau
    tau2_tensor.sum().backward()
    grad2 = attn.raw_tau.grad.item() if attn.raw_tau.grad is not None else 0.0

    assert tau1 < tau2, "Temperature should increase with larger raw_tau"
    assert grad2 > 0, "Gradient should be non-zero for gradient descent to work"
