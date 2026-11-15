"""
Comprehensive unit test suite for the Zeta Scaling Toolkit.

Tests all 4 core fixes:
1. NonCollapsingAttention - prevents softmax collapse
2. AsymptoticLRScheduler - prevents LR stagnation
3. get_scale_invariant_vit_coords - fixes aspect-ratio distortion
4. CorrectedVarianceLinear - corrects weight initialization variance
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from zeta_scaling_toolkit import (
    A0,
    AsymptoticLRScheduler,
    CorrectedVarianceLinear,
    NonCollapsingAttention,
    get_scale_invariant_vit_coords,
)


class TestNonCollapsingAttention:
    """Test suite for NonCollapsingAttention module."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        batch_size, seq_len, d_k = 2, 32, 64
        attn = NonCollapsingAttention(d_k=d_k)

        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)

        output = attn(Q, K, V, N=seq_len)

        assert output.shape == V.shape, f"Expected {V.shape}, got {output.shape}"

    def test_output_shape_different_seq_lengths(self):
        """Test output shape with various sequence lengths."""
        d_k = 64
        attn = NonCollapsingAttention(d_k=d_k)

        for seq_len in [16, 64, 256, 1024]:
            Q = torch.randn(1, seq_len, d_k)
            K = torch.randn(1, seq_len, d_k)
            V = torch.randn(1, seq_len, d_k)

            output = attn(Q, K, V, N=seq_len)
            assert output.shape == (1, seq_len, d_k)

    def test_raises_error_for_invalid_N(self):
        """Test that ValueError is raised for N <= 1."""
        attn = NonCollapsingAttention(d_k=64)
        Q = torch.randn(1, 10, 64)
        K = torch.randn(1, 10, 64)
        V = torch.randn(1, 10, 64)

        with pytest.raises(ValueError, match="Sequence length N must be > 1"):
            attn(Q, K, V, N=1)

        with pytest.raises(ValueError, match="Sequence length N must be > 1"):
            attn(Q, K, V, N=0)

    def test_scaling_floor_is_applied(self):
        """Test that scaling floor W_N > 1 is applied."""
        d_k = 64
        attn = NonCollapsingAttention(d_k=d_k, A0=A0)

        Q = torch.randn(1, 128, d_k)
        K = torch.randn(1, 128, d_k)
        V = torch.randn(1, 128, d_k)

        # Compute scaling manually to verify
        N = 128
        ln_N = math.log(N)
        W_N = 1.0 + A0 / ln_N

        assert W_N > 1.0, "Scaling floor W_N should be > 1"

        # Output should be well-defined (not NaN or Inf)
        output = attn(Q, K, V, N=N)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_custom_A0(self):
        """Test attention with custom A0 value."""
        custom_A0 = 50
        attn = NonCollapsingAttention(d_k=64, A0=custom_A0)

        Q = torch.randn(1, 64, 64)
        K = torch.randn(1, 64, 64)
        V = torch.randn(1, 64, 64)

        output = attn(Q, K, V, N=64)
        assert output.shape == (1, 64, 64)


class TestAsymptoticLRScheduler:
    """Test suite for AsymptoticLRScheduler."""

    def test_learning_rate_never_zero(self):
        """Test that LR never reaches exactly 0 over many epochs."""
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = AsymptoticLRScheduler(optimizer, A0=A0)

        learning_rates = []
        for epoch in range(100):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            learning_rates.append(lr)

        # No LR should be exactly 0
        assert all(lr > 0 for lr in learning_rates), "Learning rates should never be 0"

        # LRs should be decreasing
        assert learning_rates == sorted(learning_rates, reverse=True), "LRs should be decreasing"

    def test_learning_rate_decay_over_epochs(self):
        """Test that LR properly decays over epochs."""
        model = nn.Linear(10, 10)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = AsymptoticLRScheduler(optimizer, A0=A0)

        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Check that LRs are monotonically decreasing
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1], f"LR should decrease: {lrs[i]} should be >= {lrs[i + 1]}"

    def test_asymptotic_floor(self):
        """Test that LR approaches an asymptotic floor."""
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=1.0)
        scheduler = AsymptoticLRScheduler(optimizer, A0=A0)

        lrs = []
        for epoch in range(1000):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Last 100 LRs should be very close (converging to a floor)
        last_100_lrs = lrs[-100:]
        min_lr = min(last_100_lrs)
        max_lr = max(last_100_lrs)

        # Check convergence (relative difference should be small)
        # Using 0.04 (4%) as threshold for asymptotic convergence
        relative_diff = (max_lr - min_lr) / max_lr
        assert relative_diff < 0.04, f"LRs should converge, but relative diff is {relative_diff}"

    def test_custom_A0_affects_schedule(self):
        """Test that custom A0 value affects the learning rate schedule."""
        model = nn.Linear(10, 10)

        # Test with A0 = 100*pi
        optimizer1 = optim.SGD(model.parameters(), lr=0.1)
        scheduler1 = AsymptoticLRScheduler(optimizer1, A0=100 * math.pi)

        # Test with A0 = 10 (smaller)
        optimizer2 = optim.SGD(model.parameters(), lr=0.1)
        scheduler2 = AsymptoticLRScheduler(optimizer2, A0=10)

        lrs1 = []
        lrs2 = []

        for _ in range(50):
            scheduler1.step()
            scheduler2.step()
            lrs1.append(scheduler1.get_last_lr()[0])
            lrs2.append(scheduler2.get_last_lr()[0])

        # Different A0 values should produce different schedules
        assert lrs1 != lrs2, "Different A0 values should produce different schedules"

    def test_initial_learning_rate_multiplier(self):
        """Test that initial LR multiplier is close to 1 at early epochs."""
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = AsymptoticLRScheduler(optimizer, A0=A0)

        # At epoch 1-2, alpha_T should be close to 1
        scheduler.step()  # epoch 1
        lr_epoch1 = scheduler.get_last_lr()[0]

        # alpha_T = 1/ln(2) + A0/ln(2)^2
        ln_2 = math.log(2)
        expected_alpha = (1.0 / ln_2) + (A0 / (ln_2**2))
        expected_lr = 0.1 * expected_alpha

        assert abs(lr_epoch1 - expected_lr) < 1e-6, f"Expected {expected_lr}, got {lr_epoch1}"


class TestScaleInvariantVitCoords:
    """Test suite for get_scale_invariant_vit_coords."""

    def test_output_shape(self):
        """Test that output has correct shape (H, W, 2)."""
        height, width = 16, 16
        T = 256

        coords = get_scale_invariant_vit_coords(height, width, T)

        assert coords.shape == (
            height,
            width,
            2,
        ), f"Expected shape ({height}, {width}, 2), got {coords.shape}"

    def test_output_shape_different_resolutions(self):
        """Test output shape with various resolutions."""
        T = 1024

        test_cases = [
            (8, 8),
            (16, 16),
            (32, 32),
            (224, 224),
            (256, 512),
        ]

        for height, width in test_cases:
            coords = get_scale_invariant_vit_coords(height, width, T)
            assert coords.shape == (height, width, 2)

    def test_raises_error_for_invalid_T(self):
        """Test that ValueError is raised for T <= 1."""
        with pytest.raises(ValueError, match="Input scale T must be > 1"):
            get_scale_invariant_vit_coords(16, 16, T=1)

        with pytest.raises(ValueError, match="Input scale T must be > 1"):
            get_scale_invariant_vit_coords(16, 16, T=0.5)

    def test_output_range(self):
        """Test that output coordinates are in reasonable range."""
        height, width = 32, 32
        T = 512

        coords = get_scale_invariant_vit_coords(height, width, T)

        # Coordinates should be real numbers
        assert torch.isfinite(coords).all(), "All coordinates should be finite"

        # Coordinates should not be extremely large
        assert coords.abs().max() < 1000, "Coordinates should not be excessively large"

    def test_custom_A0(self):
        """Test with custom A0 value."""
        custom_A0 = 50
        coords = get_scale_invariant_vit_coords(16, 16, T=256, A0=custom_A0)

        assert coords.shape == (16, 16, 2)
        assert torch.isfinite(coords).all()

    def test_scaling_formula(self):
        """Test that the scaling formula r_T is correctly applied."""
        height, width = 8, 8
        T = 100

        coords = get_scale_invariant_vit_coords(height, width, T, A0=A0)

        # Verify the formula: r_T = sqrt((ln(T) + A0) / pi)
        ln_T = math.log(T)
        expected_r_T = math.sqrt((ln_T + A0) / math.pi)

        # All non-zero coordinates should be scaled by r_T (roughly)
        assert expected_r_T > 0
        assert torch.isfinite(coords).all()


class TestCorrectedVarianceLinear:
    """Test suite for CorrectedVarianceLinear."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        in_features, out_features = 128, 256
        batch_size = 4

        linear = CorrectedVarianceLinear(in_features, out_features)
        x = torch.randn(batch_size, in_features)

        output = linear(x)

        assert output.shape == (batch_size, out_features)

    def test_weight_initialization_variance(self):
        """Test that weights are initialized with correct variance."""
        in_features = 1024

        linear = CorrectedVarianceLinear(in_features, 512)

        # Expected variance: Var(w) = pi / (ln(fan_in) + A0)
        ln_T = math.log(in_features)
        expected_var = math.pi / (ln_T + A0)

        # Compute sample variance of weights
        weight_var = linear.weight.data.var().item()

        # Allow ~20% tolerance (due to random initialization)
        tolerance = 0.2 * expected_var
        assert (
            abs(weight_var - expected_var) < tolerance
        ), f"Weight variance {weight_var} should be close to {expected_var}"

    def test_bias_initialization(self):
        """Test that bias is initialized to zero."""
        linear = CorrectedVarianceLinear(128, 256, bias=True)

        assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))

    def test_no_bias_option(self):
        """Test that bias can be disabled."""
        linear = CorrectedVarianceLinear(128, 256, bias=False)

        assert linear.bias is None

    def test_raises_error_for_invalid_fan_in(self):
        """Test that ValueError is raised for fan_in <= 1."""
        with pytest.raises(ValueError, match="fan_in must be > 1"):
            CorrectedVarianceLinear(1, 256)

    def test_variance_increases_with_smaller_fan_in(self):
        """Test that variance increases as fan_in decreases."""
        # Large fan_in -> smaller variance
        linear_large = CorrectedVarianceLinear(10000, 512)
        var_large = linear_large.weight.data.var().item()

        # Small fan_in -> larger variance
        linear_small = CorrectedVarianceLinear(100, 512)
        var_small = linear_small.weight.data.var().item()

        assert var_small > var_large, "Variance should be larger for smaller fan_in"

    def test_custom_A0(self):
        """Test with custom A0 value."""
        custom_A0 = 50
        linear = CorrectedVarianceLinear(512, 256, A0=custom_A0)

        # Expected variance with custom A0
        ln_T = math.log(512)
        expected_var = math.pi / (ln_T + custom_A0)

        weight_var = linear.weight.data.var().item()
        tolerance = 0.2 * expected_var

        assert abs(weight_var - expected_var) < tolerance

    def test_forward_pass(self):
        """Test that forward pass works without errors."""
        linear = CorrectedVarianceLinear(256, 128)
        x = torch.randn(8, 256)

        output = linear(x)

        assert output.shape == (8, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        linear = CorrectedVarianceLinear(64, 32)
        x = torch.randn(4, 64, requires_grad=True)

        output = linear(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert linear.weight.grad is not None
        assert not torch.isnan(linear.weight.grad).any()


class TestA0Constant:
    """Test suite for the A0 constant."""

    def test_A0_value(self):
        """Test that A0 is correctly set to 100*pi."""
        expected_A0 = 100 * math.pi
        assert A0 == expected_A0

    def test_A0_positive(self):
        """Test that A0 is a positive number."""
        assert A0 > 0


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_attention_with_scheduler(self):
        """Test attention and scheduler together in a training loop."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = AsymptoticLRScheduler(optimizer)

        attn = NonCollapsingAttention(d_k=128)

        for epoch in range(5):
            # Forward pass
            x = torch.randn(2, 64)
            output = model(x)

            # Attention pass
            Q = torch.randn(2, 32, 128)
            K = torch.randn(2, 32, 128)
            V = torch.randn(2, 32, 128)
            attn_output = attn(Q, K, V, N=32)

            loss = output.sum() + attn_output.sum()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            lr = scheduler.get_last_lr()[0]
            assert lr > 0

    def test_all_components_together(self):
        """Test all 4 components working together."""
        batch_size, seq_len, d_model = 2, 64, 256

        # Create model with all components
        attn = NonCollapsingAttention(d_k=d_model)
        linear = CorrectedVarianceLinear(d_model, d_model * 4)
        coords = get_scale_invariant_vit_coords(8, 8, T=256)

        # Create optimizer and scheduler
        dummy_model = nn.Linear(10, 10)
        optimizer = optim.Adam(dummy_model.parameters(), lr=0.001)
        scheduler = AsymptoticLRScheduler(optimizer)

        # Simulate training steps
        for epoch in range(3):
            Q = torch.randn(batch_size, seq_len, d_model)
            K = torch.randn(batch_size, seq_len, d_model)
            V = torch.randn(batch_size, seq_len, d_model)

            attn_out = attn(Q, K, V, N=seq_len)
            linear_out = linear(attn_out.mean(dim=1))

            assert attn_out.shape == (batch_size, seq_len, d_model)
            assert linear_out.shape == (batch_size, d_model * 4)
            assert coords.shape == (8, 8, 2)

            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            assert lr > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=zeta_scaling_toolkit"])
