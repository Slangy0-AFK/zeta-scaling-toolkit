"""Position-extension tests with short, synthetic offset histories."""

import importlib.util
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.52.2":
    pytest.skip("prototype requires transformers 4.52.2", allow_module_level=True)

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "smollm2_position_extension",
    ROOT / "research/smollm2-integration-v1/position_extension.py",
)
assert spec is not None and spec.loader is not None
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)
ExperimentalPositionAdapter = extension.ExperimentalPositionAdapter


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


def test_original_adapter_guard_and_original_range_parity(model):
    original = extension.adapter.ZetaLlamaAdapter(model)
    experimental = ExperimentalPositionAdapter(model, max_positions=64)
    ids = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(original.forward(ids), experimental.forward(ids), rtol=0, atol=0)
    with pytest.raises(ValueError, match="position limit"):
        original.forward(torch.ones(1, 29, dtype=torch.long))
    assert experimental.manifest()["rope_scaling"] is None


def test_offset_continuation_matches_causal_native_attention(model):
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    start, window = 31, 3
    extended = ExperimentalPositionAdapter(
        model, max_positions=64, initial_position=start, window=window
    )
    actual = torch.cat((extended.forward(ids[:, :2]), extended.forward(ids[:, 2:])), dim=1)
    positions = torch.arange(start, start + ids.shape[1])
    q, k = positions[:, None], positions[None, :]
    visible = (k <= q) & (k >= q - window + 1)
    mask = torch.zeros(1, 1, len(positions), len(positions)).masked_fill(
        ~visible, torch.finfo(torch.float32).min
    )
    with torch.inference_mode():
        expected = model(
            ids,
            attention_mask=mask,
            position_ids=positions[None, :],
            cache_position=positions,
            use_cache=False,
        ).logits
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for layer in extended.cache.layers:
        assert layer.next_position == start + ids.shape[1]
        assert layer.positions.tolist() == [33, 34, 35]
    changed = ids.clone()
    changed[0, -1] = 8
    other = ExperimentalPositionAdapter(model, max_positions=64, initial_position=start, window=3)
    torch.testing.assert_close(actual[:, :-1], other.forward(changed)[:, :-1], rtol=0, atol=0)


def test_opt_in_and_offset_contract_rejections(model):
    with pytest.raises(ValueError, match="exceed"):
        ExperimentalPositionAdapter(model, max_positions=32)
    with pytest.raises(ValueError, match="cannot contain"):
        ExperimentalPositionAdapter(model, max_positions=64, initial_position=32, sink_tokens=1)
    with pytest.raises(ValueError, match="experimental range"):
        ExperimentalPositionAdapter(model, max_positions=64, initial_position=64)
    model.config.rope_scaling = {"rope_type": "linear", "factor": 2.0}
    with pytest.raises(ValueError, match="unmodified"):
        ExperimentalPositionAdapter(model, max_positions=64)


def test_native_rope_matches_independent_split_half_reference(model):
    positions = torch.tensor([[0, 31, 32, 63]])
    x = torch.linspace(-1, 1, 8).view(1, 1, 1, 8).expand(1, 1, 4, 8)
    cosine, sine = model.model.rotary_emb(x, positions)
    theta = float(model.config.rope_theta)
    frequencies = theta ** (-torch.arange(0, 8, 2, dtype=torch.float64) / 8)
    angles = positions.to(torch.float64).unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    first, second = x.to(torch.float64).chunk(2, dim=-1)
    reference = x * angles.cos() + torch.cat((-second, first), dim=-1) * angles.sin()
    actual = x * cosine.unsqueeze(1) + torch.cat(
        (-x[..., 4:], x[..., :4]), dim=-1
    ) * sine.unsqueeze(1)
    torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=1e-5)
