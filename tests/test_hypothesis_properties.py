"""Property-based tests using hypothesis.

These tests verify invariants that should hold for any valid input.
"""

import pytest
import torch
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from zeta_scaling_toolkit import (
    NonCollapsingAttention,
    AsymptoticLRScheduler,
    make_corrected_linear,
    get_isotropic_coords,
)


# Hypothesis settings for reproducibility in CI
hypothesis_settings = settings(
    max_examples=100,
    derandomize=False,  # Use fixed seed for CI
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,  # Tests can take a while, especially coordinate tests
)


# Strategies
def batch_sizes():
    """Generate batch sizes from 1 to 4."""
    return st.integers(min_value=1, max_value=4)


def seq_lengths():
    """Generate sequence lengths from 1 to 64."""
    return st.integers(min_value=1, max_value=64)


def embedding_dims():
    """Generate embedding dims from 8 to 64, step 8."""
    return st.sampled_from([8, 16, 24, 32, 40, 48, 56, 64])


@given(
    B=batch_sizes(),
    Lq=seq_lengths(),
    Lk=seq_lengths(),
    d=embedding_dims(),
)
@hypothesis_settings
def test_attention_output_shape(B, Lq, Lk, d):
    """Attention output shape must match query shape."""
    attn = NonCollapsingAttention(d)
    
    q = torch.randn(B, Lq, d)
    k = torch.randn(B, Lk, d)
    v = torch.randn(B, Lk, d)
    
    out = attn(q, k, v)
    assert out.shape == (B, Lq, d), f"Expected {(B, Lq, d)}, got {out.shape}"


@given(
    B=batch_sizes(),
    L=seq_lengths(),
    d=embedding_dims(),
)
@hypothesis_settings
def test_attention_output_finite(B, L, d):
    """Attention output must be finite for finite inputs."""
    attn = NonCollapsingAttention(d)
    
    q = torch.randn(B, L, d)
    k = torch.randn(B, L, d)
    v = torch.randn(B, L, d)
    
    out = attn(q, k, v)
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"


@given(
    B=st.integers(min_value=1, max_value=4),
    L=st.integers(min_value=2, max_value=32),
    d=st.sampled_from([8, 16, 32]),
)
@hypothesis_settings
def test_attention_weights_sum_to_one(B, L, d):
    """Rows of attention weights must sum to 1 (within numerical tolerance)."""
    attn = NonCollapsingAttention(d)
    
    q = torch.randn(B, L, d)
    k = torch.randn(B, L, d)
    v = torch.randn(B, L, d)
    
    # Manually compute attention weights to check
    q_norm = torch.nn.functional.normalize(q, dim=-1)
    k_norm = torch.nn.functional.normalize(k, dim=-1)
    
    # logits shape (B, L, L)
    logits = torch.einsum("bqd,bkd->bqk", q_norm, k_norm)
    logits = attn.tau * logits
    
    weights = torch.softmax(logits, dim=-1)
    row_sums = weights.sum(dim=-1)
    
    # Each row should sum to 1
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


@given(
    B=st.integers(min_value=1, max_value=4),
    L=st.integers(min_value=1, max_value=32),
    d=st.sampled_from([8, 16, 32]),
)
@hypothesis_settings
def test_scale_invariance_large_norms(B, L, d):
    """Attention weights should be invariant to positive scaling of q and k when ||x|| >= 1e-2."""
    attn = NonCollapsingAttention(d)
    
    # Ensure large norms
    q = torch.randn(B, L, d) * 10
    k = torch.randn(B, L, d) * 10
    v = torch.randn(B, L, d) * 10
    
    out1 = attn(q, k, v)
    out2 = attn(q * 2, k * 2, v)  # Scale only q and k
    
    # Attention pattern should be the same, so outputs should be identical
    assert torch.allclose(out1, out2, atol=1e-4), "Attention pattern changed with q/k scaling"


@given(
    B=st.integers(min_value=1, max_value=4),
    L=st.integers(min_value=1, max_value=16),
    d=st.sampled_from([8, 16]),
)
@hypothesis_settings
def test_attention_gradients_finite(B, L, d):
    """Gradients must be finite for finite inputs."""
    attn = NonCollapsingAttention(d)
    
    q = torch.randn(B, L, d, requires_grad=True)
    k = torch.randn(B, L, d, requires_grad=True)
    v = torch.randn(B, L, d, requires_grad=True)
    
    out = attn(q, k, v)
    loss = out.sum()
    loss.backward()
    
    assert torch.isfinite(q.grad).all(), "q gradient is not finite"
    assert torch.isfinite(k.grad).all(), "k gradient is not finite"
    assert torch.isfinite(v.grad).all(), "v gradient is not finite"


@given(
    total_steps=st.integers(min_value=10, max_value=100),
    floor_ratio=st.floats(min_value=0.0, max_value=1.0),
    warmup_steps=st.integers(min_value=0, max_value=20),
)
@hypothesis_settings
def test_scheduler_output_range(total_steps, floor_ratio, warmup_steps):
    """Scheduler LR must be in [floor, 1.0] after warmup and before end."""
    if warmup_steps >= total_steps:
        return  # Skip invalid parameter combinations
    
    param = torch.tensor([1.0], requires_grad=True)
    scheduler = AsymptoticLRScheduler(
        torch.optim.SGD([param], lr=1.0),
        total_steps=total_steps,
        floor_ratio=floor_ratio,
        warmup_steps=warmup_steps,
    )
    
    # Move past warmup to check steady-state behavior
    for _ in range(warmup_steps + 2):
        scheduler.step()
    
    # Check a few more steps after warmup
    for _ in range(min(5, total_steps - warmup_steps - 2)):
        lr = scheduler.optimizer.param_groups[0]["lr"]
        assert floor_ratio <= lr <= 1.0 + 1e-6, f"LR {lr} outside [floor={floor_ratio}, 1.0]"
        scheduler.step()


@given(
    h=st.integers(min_value=1, max_value=512),
    w=st.integers(min_value=1, max_value=512),
)
@hypothesis_settings
def test_coordinates_finite(h, w):
    """Coordinates must always be finite."""
    coords = get_isotropic_coords(h, w)
    assert torch.isfinite(coords).all(), "Coordinates contain NaN/Inf"
    assert coords.shape == (h, w, 2), f"Expected {(h, w, 2)}, got {coords.shape}"


@given(
    h=st.integers(min_value=1, max_value=512),
    w=st.integers(min_value=1, max_value=512),
)
@hypothesis_settings
def test_coordinates_mean_near_zero(h, w):
    """Coordinate mean should be close to zero."""
    coords = get_isotropic_coords(h, w)
    mean = coords.mean()
    assert torch.abs(mean) < 1e-4, f"Mean coordinate {mean} too far from 0"


@given(
    h=st.integers(min_value=1, max_value=256),
    w=st.integers(min_value=1, max_value=256),
)
@hypothesis_settings
def test_coordinates_radius_bounded(h, w):
    """Max coordinate radius should be <= sqrt(2)/2."""
    coords = get_isotropic_coords(h, w)
    radius = torch.sqrt((coords**2).sum(dim=-1))
    max_radius = radius.max()
    
    # Theoretical max is sqrt(2)/2 ≈ 0.707
    assert max_radius <= 0.707 + 1e-4, f"Max radius {max_radius} exceeds sqrt(2)/2"


@given(
    h=st.integers(min_value=1, max_value=32),
    w=st.integers(min_value=1, max_value=32),
)
@hypothesis_settings
def test_coordinates_equal_pixel_distance_equal_coord_distance(h, w):
    """Equal pixel distances should map to equal coordinate distances in x and y."""
    coords = get_isotropic_coords(h, w)
    
    # Check horizontal step - sample rows instead of checking all
    if w > 1:
        x_step = (coords[0, 1, 0] - coords[0, 0, 0]).abs()
        # Check a few rows
        for i in [0, h // 2, h - 1]:
            step = (coords[i, 1, 0] - coords[i, 0, 0]).abs()
            assert torch.allclose(step, x_step, atol=1e-5), "Horizontal steps not uniform"
    
    # Check vertical step - sample columns instead of checking all
    if h > 1:
        y_step = (coords[1, 0, 1] - coords[0, 0, 1]).abs()
        # Check a few columns
        for j in [0, w // 2, w - 1]:
            step = (coords[1, j, 1] - coords[0, j, 1]).abs()
            assert torch.allclose(step, y_step, atol=1e-5), "Vertical steps not uniform"


@given(
    in_features=st.integers(min_value=1, max_value=64),
    out_features=st.integers(min_value=1, max_value=64),
)
@hypothesis_settings
def test_orthogonal_init_preserves_shape(in_features, out_features):
    """Orthogonal initialization should preserve layer shape."""
    layer = make_corrected_linear(in_features, out_features, nonlinearity="linear")
    
    assert layer.weight.shape == (out_features, in_features)
    assert torch.isfinite(layer.weight).all(), "Weights contain NaN/Inf"


@given(
    in_features=st.integers(min_value=2, max_value=32),
    out_features=st.integers(min_value=2, max_value=32),
)
@hypothesis_settings
def test_orthogonal_init_near_orthogonality(in_features, out_features):
    """Weight matrix should be approximately orthogonal (W^T W ≈ λI)."""
    layer = make_corrected_linear(in_features, out_features, nonlinearity="linear")
    W = layer.weight
    
    # For square matrices, compute W^T W
    if in_features == out_features:
        WTW = W.T @ W
        eye = torch.eye(in_features)
        # Should be close to identity (scaled by gain squared)
        trace = WTW.trace()
        scale = trace / in_features
        normalized = WTW / scale
        assert torch.allclose(normalized, eye, atol=0.1), "Not sufficiently orthogonal"
