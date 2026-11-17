"""Compatibility checks for the isolated Transformers 5.23.0 Llama adapter."""

import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "5.23.0":
    pytest.skip("compatibility port targets transformers 5.23.0", allow_module_level=True)

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/context-100k-experiment-v2"))
from compat_adapter import ExperimentalPositionAdapter, ZetaLlamaAdapter  # noqa: E402


@pytest.fixture
def model():
    torch.manual_seed(74)
    config = LlamaConfig(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def test_native_adapter_parity_and_8192_style_default_guard(model):
    adapter = ZetaLlamaAdapter(model)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.inference_mode():
        native = model(
            tokens,
            attention_mask=torch.ones_like(tokens),
            position_ids=torch.arange(tokens.shape[1])[None, :],
            use_cache=False,
        ).logits
    actual = adapter.forward(tokens)

    torch.testing.assert_close(actual, native, rtol=1e-5, atol=1e-6)
    assert adapter.cache.get_seq_length() == tokens.shape[1]
    with pytest.raises(ValueError, match="configured position limit"):
        ZetaLlamaAdapter(model).forward(torch.ones((1, 33), dtype=torch.long))


def test_explicit_extension_advances_beyond_65537_and_preserves_causality(model):
    tokens = torch.tensor([[3, 4, 5, 6]])
    start = 65_535
    adapter = ExperimentalPositionAdapter(
        model, max_positions=100_001, window=3, initial_position=start
    )
    actual = adapter.forward(tokens)
    positions = torch.arange(start, start + tokens.shape[1])
    query, key = positions[:, None], positions[None, :]
    visible = (key <= query) & (key >= query - 2)
    mask = torch.zeros(1, 1, len(positions), len(positions)).masked_fill(
        ~visible, torch.finfo(torch.float32).min
    )
    with torch.inference_mode():
        native = model(
            tokens,
            attention_mask=mask,
            position_ids=positions[None, :],
            use_cache=False,
        ).logits
    torch.testing.assert_close(actual, native, rtol=1e-5, atol=1e-6)

    changed = tokens.clone()
    changed[0, -1] = 7
    other = ExperimentalPositionAdapter(
        model, max_positions=100_001, window=3, initial_position=start
    )
    changed_logits = other.forward(changed)
    assert torch.equal(actual[:, :-1], changed_logits[:, :-1])
    assert adapter.cache.get_seq_length() == 65_539
    assert adapter.cache.layers[0].storage.positions.tolist() == [65_536, 65_537, 65_538]
    assert torch.isfinite(actual).all()


def test_high_position_rope_matches_independent_fp64_reference(model):
    positions = torch.tensor([[0, 65_536, 100_000]], dtype=torch.long)
    head_dim = model.config.head_dim
    x = torch.linspace(-1, 1, head_dim).view(1, 1, 1, head_dim).expand(1, 1, 3, head_dim)
    cosine, sine = model.model.rotary_emb(x, positions)
    rotated = x * cosine.unsqueeze(1) + torch.cat(
        (-x[..., head_dim // 2 :], x[..., : head_dim // 2]), dim=-1
    ) * sine.unsqueeze(1)

    base = float(model.config.rope_parameters["rope_theta"])
    frequencies = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    angles = positions.to(torch.float64).unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    first, second = x.double().chunk(2, dim=-1)
    reference = x.double() * angles.cos() + torch.cat((-second, first), dim=-1) * angles.sin()

    assert torch.isfinite(rotated).all()
    torch.testing.assert_close(rotated.double(), reference, rtol=0, atol=5e-3)


def test_only_native_unscaled_rope_can_be_extended(model):
    model.config.rope_parameters = {"rope_type": "linear", "rope_theta": 10_000.0, "factor": 2.0}
    with pytest.raises(ValueError, match="unmodified native default RoPE"):
        ExperimentalPositionAdapter(model, max_positions=100_001)
