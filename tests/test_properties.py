import copy
import math
import pickle
import statistics

import pytest
import torch
import torch.nn.functional as F
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from benchmarks.length_generalization import SequenceLM, make_summaries, needle_batch
from benchmarks.real_task import MODEL_WIDTH, TransformerBlock
from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    get_isotropic_coords,
    get_scale_invariant_vit_coords,
    make_corrected_linear,
)
from zeta_scaling_toolkit.core import _unit
from zeta_scaling_toolkit.positions import (
    StreamingKVCache,
    alibi_bias,
    rotary_embedding,
    streaming_mask,
)


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


@pytest.mark.parametrize("floor_ratio", [0.0, 0.01, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("total_steps", [1, 2, 10, 100])
def test_scheduler_matches_stock_cosine_within_horizon(floor_ratio, total_steps):
    toolkit_parameter = torch.nn.Parameter(torch.zeros(1))
    cosine_parameter = torch.nn.Parameter(torch.zeros(1))
    toolkit_optimizer = SGD([toolkit_parameter], lr=0.001)
    cosine_optimizer = SGD([cosine_parameter], lr=0.001)
    toolkit = AsymptoticLRScheduler(
        toolkit_optimizer, floor_ratio=floor_ratio, total_steps=total_steps
    )
    cosine = CosineAnnealingLR(cosine_optimizer, T_max=total_steps, eta_min=floor_ratio * 0.001)

    for _ in range(total_steps):
        toolkit_optimizer.step()
        cosine_optimizer.step()
        toolkit.step()
        cosine.step()
        assert toolkit_optimizer.param_groups[0]["lr"] == pytest.approx(
            cosine_optimizer.param_groups[0]["lr"], abs=1e-12
        )


def test_scheduler_stays_at_floor_beyond_horizon_while_stock_cosine_rises():
    toolkit_parameter = torch.nn.Parameter(torch.zeros(1))
    cosine_parameter = torch.nn.Parameter(torch.zeros(1))
    toolkit_optimizer = SGD([toolkit_parameter], lr=0.001)
    cosine_optimizer = SGD([cosine_parameter], lr=0.001)
    toolkit = AsymptoticLRScheduler(toolkit_optimizer, floor_ratio=0.1, total_steps=4)
    cosine = CosineAnnealingLR(cosine_optimizer, T_max=4, eta_min=0.0001)

    for _ in range(14):
        toolkit_optimizer.step()
        cosine_optimizer.step()
        toolkit.step()
        cosine.step()

    assert toolkit_optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)
    assert cosine_optimizer.param_groups[0]["lr"] > 0.0001


@pytest.mark.parametrize("key_length", [128, 1024, 8192, 65536])
def test_random_cosine_distractor_mass_matches_gaussian_approximation(key_length):
    dimension = 64
    temperature = 10.0
    observed_weights = []
    for seed in range(5):
        generator = torch.Generator().manual_seed(seed)
        query = torch.randn(dimension, generator=generator)
        query = query / query.norm()
        keys = torch.randn(key_length - 1, dimension, generator=generator)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        distractor_mass = torch.exp(temperature * (keys @ query)).sum(dtype=torch.float64)
        observed_weights.append(
            math.exp(temperature) / (math.exp(temperature) + distractor_mass.item())
        )

    observed = statistics.mean(observed_weights)
    expected = math.exp(temperature) / (
        math.exp(temperature) + (key_length - 1) * math.exp(temperature**2 / (2 * dimension))
    )
    assert abs(observed - expected) / expected <= 0.05


@pytest.mark.parametrize("key_length", [2**10, 2**13, 2**16])
def test_log_length_attention_retrieves_exact_needle_with_chunked_cpu_path(key_length):
    generator = torch.Generator().manual_seed(411)
    keys = torch.randn(1, key_length, 64, generator=generator)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    query = keys[:, :1]
    values = torch.zeros(1, key_length, 1)
    values[:, 0] = 1.0
    mask = torch.ones(2, key_length, dtype=torch.bool)
    mask[1, -1] = False
    queries = query.expand(1, 2, 64)

    with torch.no_grad():
        weight = NonCollapsingAttention(64, length_scaling="log", train_length=64)(
            queries, keys, values, attn_mask=mask
        )[0, 0, 0]

    assert weight >= 0.9


@pytest.mark.parametrize("key_length", [2**10, 2**13, 2**16])
def test_log_length_attention_noisy_needle_beats_top_distractor(key_length):
    generator = torch.Generator().manual_seed(811)
    keys = torch.randn(1, key_length, 64, generator=generator)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    needle = keys[0, 0]
    noise = torch.randn(64, generator=generator)
    noise = noise - torch.dot(noise, needle) * needle
    noise = noise / noise.norm()
    query = (0.7 * needle + math.sqrt(1.0 - 0.7**2) * noise).view(1, 1, 64)
    scores = keys[0] @ query[0, 0]
    distractor_index = int(scores[1:].argmax()) + 1
    values = torch.zeros(1, key_length, 2)
    values[0, 0, 0] = 1.0
    values[0, distractor_index, 1] = 1.0
    mask = torch.ones(2, key_length, dtype=torch.bool)
    mask[1, -1] = False
    queries = query.expand(1, 2, 64)

    with torch.no_grad():
        weights = NonCollapsingAttention(64, length_scaling="log", train_length=64)(
            queries, keys, values, attn_mask=mask
        )[0, 0]

    assert float(scores[0]) == pytest.approx(0.7, abs=1e-5)
    assert weights[0] > weights[1]
    assert weights[0] - weights[1] > 0.1


def test_log_length_attention_respects_effective_temperature_cap():
    generator = torch.Generator().manual_seed(971)
    keys = torch.randn(1, 4096, 8, generator=generator)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    query = keys[:, :1]
    values = torch.zeros(1, 4096, 1)
    values[:, 0] = 1.0
    capped = NonCollapsingAttention(8, length_scaling="log", train_length=2, tau_eff_max=12.0)
    uncapped = NonCollapsingAttention(8, length_scaling="log", train_length=2, tau_eff_max=200.0)
    uncapped.load_state_dict(capped.state_dict())

    with torch.no_grad():
        capped_weight = capped(query, keys, values)[0, 0, 0]
        uncapped_weight = uncapped(query, keys, values)[0, 0, 0]

    assert capped_weight < uncapped_weight


def test_log_length_attention_matches_unscaled_at_train_length_and_padding_count():
    torch.manual_seed(29)
    query = torch.randn(1, 2, 8)
    keys = torch.randn(1, 8, 8)
    values = torch.randn(1, 8, 3)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :4] = True
    mask[1, 4:] = True
    scaled = NonCollapsingAttention(8, length_scaling="log", train_length=4)
    unscaled = NonCollapsingAttention(8)
    unscaled.load_state_dict(scaled.state_dict())

    torch.testing.assert_close(
        scaled(query[:, :1], keys, values, mask[:1]),
        unscaled(query[:, :1], keys, values, mask[:1]),
        atol=0,
        rtol=0,
    )
    output = scaled(query, keys, values, mask)
    expected = unscaled(query, keys, values, mask)
    torch.testing.assert_close(output[0, 0], expected[0, 0], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(output[0, 1], expected[0, 1])


def test_log_length_attention_float64_gradcheck_and_finite_temperature_gradient():
    torch.manual_seed(53)
    attention = NonCollapsingAttention(3, length_scaling="log", train_length=2).double()
    query = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    keys = torch.randn(1, 3, 3, dtype=torch.float64, requires_grad=True)
    values = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda q, k, v: attention(q, k, v), (query, keys, values))
    attention(query, keys, values).sum().backward()
    assert torch.isfinite(attention.raw_tau.grad)


def test_log_length_attention_masked_row_gradcheck():
    torch.manual_seed(59)
    attention = NonCollapsingAttention(3, length_scaling="log", train_length=2).double()
    query = torch.randn(1, 3, 3, dtype=torch.float64, requires_grad=True)
    keys = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
    values = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False], [False, False, False, False]]
    )
    assert torch.autograd.gradcheck(
        lambda query_input, key_input, value_input: attention(
            query_input, key_input, value_input, mask
        ),
        (query, keys, values),
    )
    output = attention(query, keys, values, mask)
    assert torch.equal(output[0, 2], torch.zeros(2, dtype=torch.float64))
    output.sum().backward()
    for gradient in (query.grad, keys.grad, values.grad, attention.raw_tau.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_log_length_attention_low_precision_outputs_are_finite(dtype):
    torch.manual_seed(61)
    query = torch.randn(1, 2, 8, dtype=dtype)
    keys = torch.randn(1, 16, 8, dtype=dtype)
    values = torch.randn(1, 16, 4, dtype=dtype)
    output = NonCollapsingAttention(8, length_scaling="log", train_length=4)(query, keys, values)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length_scaling": "unknown"},
        {"length_scaling": "log"},
        {"length_scaling": "log", "train_length": 1},
        {"tau_eff_max": 0.0},
        {"tau_eff_max": float("inf")},
    ],
)
def test_attention_rejects_invalid_length_scaling_options(kwargs):
    with pytest.raises((TypeError, ValueError)):
        NonCollapsingAttention(4, **kwargs)


def test_unscaled_attention_accepts_legacy_temperature_above_default_effective_cap():
    attention = NonCollapsingAttention(4, tau0=300.0, tau_min=250.0, tau_max=400.0)
    query = torch.randn(1, 1, 4)
    assert torch.isfinite(attention(query, query, query)).all()


def test_attention_empty_queries_keep_autograd_connections():
    query = torch.empty(1, 0, 4, requires_grad=True)
    keys = torch.randn(1, 3, 4, requires_grad=True)
    values = torch.randn(1, 3, 2, requires_grad=True)
    attention = NonCollapsingAttention(4)
    attention(query, keys, values).sum().backward()
    for gradient in (query.grad, keys.grad, values.grad, attention.raw_tau.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_log_length_attention_handles_empty_batches_with_a_mask():
    query = torch.empty(0, 2, 4, requires_grad=True)
    keys = torch.empty(0, 3, 4, requires_grad=True)
    values = torch.empty(0, 3, 2, requires_grad=True)
    attention = NonCollapsingAttention(4, length_scaling="log", train_length=2)
    output = attention(query, keys, values, torch.ones(2, 3, dtype=torch.bool))
    assert output.shape == (0, 2, 2)
    output.sum().backward()
    assert all(gradient is not None for gradient in (query.grad, keys.grad, values.grad))


def test_benchmark_transformer_initialization_uses_layer_specific_gains():
    block = TransformerBlock(corrected_attention=False, corrected_init=True, head_size=8)
    residual_scale = 1.0 / math.sqrt(4)
    torch.testing.assert_close(
        block.qkv.weight.T @ block.qkv.weight,
        torch.eye(MODEL_WIDTH),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        block.projection.weight.T @ block.projection.weight,
        torch.eye(MODEL_WIDTH) * residual_scale**2,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        block.feedforward[0].weight.T @ block.feedforward[0].weight,
        torch.eye(MODEL_WIDTH) * 2.0,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        block.feedforward[2].weight @ block.feedforward[2].weight.T,
        torch.eye(MODEL_WIDTH) * residual_scale**2,
        atol=1e-5,
        rtol=1e-5,
    )


def test_needle_batch_contains_one_matching_random_query_key_per_example():
    inputs, targets = needle_batch(32, 64, torch.Generator().manual_seed(7))
    query_keys = inputs[:, -1]
    assert query_keys.unique().numel() > 1
    keys, values = inputs[:, :-2:2], inputs[:, 1:-2:2]
    matches = keys == query_keys[:, None]
    assert torch.all(matches.sum(dim=1) == 1)
    assert torch.equal(values[matches], targets)


def test_evaluation_summaries_require_ten_pairs_and_match_hyperparameters():
    records = []
    for seed in range(10):
        for variant, value in (("rope", 10.0), ("rope-log", 9.0)):
            records.append(
                {
                    "task": "char",
                    "variant": variant,
                    "init_mode": "baseline",
                    "scheduler": "cosine",
                    "steps": 500,
                    "seed": seed,
                    "tau0": 10.0,
                    "rope_base": 10000.0,
                    "metrics": {"128": value},
                }
            )
    assert make_summaries(records[:-2]) == []
    summaries = make_summaries(records)
    assert len(summaries) == 1
    assert summaries[0]["comparison"] == "qk_normalization_and_log_scaling"
    assert summaries[0]["mean_difference"] == -1.0
    assert summaries[0]["paired_95pct_bootstrap_ci"] == [-1.0, -1.0]
    assert summaries[0]["seeds_better"] == 10


@pytest.mark.parametrize("variant", ["rope", "rope-log", "alibi", "rope-log-stream"])
def test_evaluation_chunked_attention_respects_causal_prefix(variant):
    torch.manual_seed(79)
    model = SequenceLM(19, variant, False, 64, 10.0, 10000.0).eval()
    tokens = torch.randint(0, 19, (1, 140))
    with torch.no_grad():
        full = model(tokens)
        prefix = model(tokens[:, :129])
    torch.testing.assert_close(full[:, :129], prefix, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("offset", [0, 2**11, 2**20])
def test_rotary_embedding_relative_dot_product_is_offset_invariant(offset):
    torch.manual_seed(73)
    query = torch.randn(1, 1, 32)
    key = torch.randn(1, 1, 32)
    relative = 17
    rotated_query = rotary_embedding(query, torch.tensor([offset], dtype=torch.int64))
    rotated_key = rotary_embedding(key, torch.tensor([offset + relative], dtype=torch.int64))
    shifted_query = rotary_embedding(query, torch.tensor([offset + 2**20], dtype=torch.int64))
    shifted_key = rotary_embedding(
        key, torch.tensor([offset + 2**20 + relative], dtype=torch.int64)
    )
    torch.testing.assert_close(
        (rotated_query * rotated_key).sum(),
        (shifted_query * shifted_key).sum(),
        atol=1e-4,
        rtol=0,
    )


def test_rotary_embedding_handles_large_positions_and_scaling_modes():
    values = torch.randn(2, 4, 32)
    positions = torch.tensor([10**9, 10**9 + 1, 10**9 + 2, 10**9 + 3], dtype=torch.int64)
    for mode in (None, "linear", "ntk"):
        result = rotary_embedding(values, positions, scaling=mode, scale_factor=4.0)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(values.norm(dim=-1), result.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_rotary_embedding_matches_known_angle_orientation_and_frequency():
    values = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
    result = rotary_embedding(values, torch.tensor([1], dtype=torch.int64))
    expected = torch.tensor([[[0.0, 0.0, math.cos(0.01), math.sin(0.01)]]])
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rotary_embedding_is_finite_for_low_precision_inputs(dtype):
    values = torch.randn(1, 3, 16, dtype=dtype)
    positions = torch.tensor([0, 2**20, 10**9], dtype=torch.int64)
    result = rotary_embedding(values, positions)
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


def test_streaming_mask_matches_causal_window_and_sink_visibility():
    mask = streaming_mask(q_len=3, k_len=8, window=3, sink_tokens=2)
    expected = torch.tensor(
        [
            [True, True, False, True, True, True, False, False],
            [True, True, False, False, True, True, True, False],
            [True, True, False, False, False, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_streaming_kv_cache_retains_sinks_and_window_with_bounded_memory():
    cache = StreamingKVCache(window=3, sink_tokens=2)
    for position in range(20):
        key = torch.full((1, 1, 4), float(position), requires_grad=True)
        value = key + 1
        keys, values, positions = cache.append(key, value)
        expected = [index for index in range(position + 1) if index < 2 or index >= position - 2]
        assert positions.tolist() == expected
        assert keys.shape[-2] <= 5
        assert keys[0, :, 0].tolist() == [float(index) for index in expected]
        assert values[0, :, 0].tolist() == [float(index + 1) for index in expected]
        assert not keys.requires_grad
        assert not values.requires_grad
    cache.clear()
    assert cache.next_position == 0
    assert cache.keys is None and cache.values is None and cache.positions is None


def test_streaming_kv_cache_attention_matches_dense_mask():
    torch.manual_seed(83)
    keys = torch.randn(1, 12, 4)
    values = torch.randn(1, 12, 3)
    queries = torch.randn(1, 12, 4)
    attention = NonCollapsingAttention(4, length_scaling="log", train_length=3)
    cache = StreamingKVCache(window=3, sink_tokens=2)
    for position in range(12):
        cached_keys, cached_values, _ = cache.append(
            keys[:, position : position + 1], values[:, position : position + 1]
        )
        streamed = attention(queries[:, position : position + 1], cached_keys, cached_values)
        mask = streaming_mask(1, position + 1, window=3, sink_tokens=2)
        dense = attention(
            queries[:, position : position + 1],
            keys[:, : position + 1],
            values[:, : position + 1],
            mask,
        )
        torch.testing.assert_close(streamed, dense, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("window,sinks", [(0, 0), (1, -1), (True, 0), (1, True)])
def test_streaming_kv_cache_rejects_invalid_configuration(window, sinks):
    with pytest.raises(ValueError):
        StreamingKVCache(window=window, sink_tokens=sinks)


def test_streaming_kv_cache_rejects_incompatible_appends():
    cache = StreamingKVCache(window=2, sink_tokens=1)
    with pytest.raises(TypeError):
        cache.append(None, torch.ones(1, 1, 4))
    with pytest.raises(ValueError):
        cache.append(torch.ones(1, 2, 4), torch.ones(1, 2, 4))
    with pytest.raises(ValueError):
        cache.append(torch.ones(1, 1, 4), torch.ones(2, 1, 4))
    cache.append(torch.ones(1, 1, 4), torch.ones(1, 1, 4))
    with pytest.raises(ValueError):
        cache.append(torch.ones(1, 1, 5), torch.ones(1, 1, 4))


@settings(derandomize=True, deadline=None, max_examples=40)
@given(
    window=st.integers(min_value=1, max_value=8),
    sinks=st.integers(min_value=0, max_value=8),
    length=st.integers(min_value=1, max_value=30),
)
def test_streaming_kv_cache_visibility_property(window, sinks, length):
    cache = StreamingKVCache(window=window, sink_tokens=sinks)
    for position in range(length):
        value = torch.full((1, 1, 2), float(position))
        keys, _, positions = cache.append(value, value)
        expected = [
            index
            for index in range(position + 1)
            if index < sinks or index >= position - window + 1
        ]
        assert positions.tolist() == expected
        assert keys.shape[-2] <= window + sinks


def test_alibi_bias_decreases_monotonically_with_absolute_distance():
    bias = alibi_bias(num_heads=4, q_len=5, k_len=5)
    for head in range(4):
        for distance in range(4):
            near = bias[0, head, 4, 4 - distance]
            far = bias[0, head, 4, 3 - distance]
            assert near > far


@pytest.mark.parametrize(
    ("values", "positions", "kwargs", "error"),
    [
        (torch.ones(1, 2, dtype=torch.int64), torch.arange(1), {}, TypeError),
        (torch.ones(1, 3), torch.arange(1), {}, ValueError),
        (torch.ones(1, 2), torch.ones(1), {}, TypeError),
        (torch.ones(1, 2), torch.arange(3), {}, ValueError),
        (torch.ones(1, 2), torch.arange(1), {"base": 1}, ValueError),
        (torch.ones(1, 2), torch.arange(1), {"scaling": "invalid"}, ValueError),
        (torch.ones(1, 2), torch.arange(1), {"scale_factor": 0}, ValueError),
    ],
)
def test_rotary_embedding_rejects_invalid_inputs(values, positions, kwargs, error):
    with pytest.raises(error):
        rotary_embedding(values, positions, **kwargs)


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((0, 1, 1), {}, ValueError),
        ((1, -1, 1), {}, ValueError),
        ((1, 1, 1), {"dtype": torch.int64}, TypeError),
    ],
)
def test_alibi_bias_rejects_invalid_inputs(args, kwargs, error):
    with pytest.raises(error):
        alibi_bias(*args, **kwargs)


@pytest.mark.parametrize("args", [(-1, 1, 1, 0), (2, 1, 1, 0), (1, 1, 0, 0), (1, 1, 1, -1)])
def test_streaming_mask_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        streaming_mask(*args)


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
