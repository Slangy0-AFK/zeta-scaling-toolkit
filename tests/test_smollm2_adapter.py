"""Offline contract tests for the opt-in SmolLM2 cache adapter."""

import importlib.util
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.52.2":
    pytest.skip("adapter requires transformers 4.52.2", allow_module_level=True)

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

ADAPTER = Path(__file__).resolve().parents[1] / "research/smolLM2-integration-v1/adapter.py"
spec = importlib.util.spec_from_file_location("smollm2_adapter", ADAPTER)
assert spec is not None and spec.loader is not None
adapter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_module)
ZetaLlamaAdapter = adapter_module.ZetaLlamaAdapter


@pytest.fixture
def model():
    torch.manual_seed(48)
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


def dense_window_mask(length, window, sinks):
    q = torch.arange(length)[:, None]
    k = torch.arange(length)[None, :]
    visible = (k <= q) & ((k < sinks) | (k >= q - window + 1))
    return torch.zeros(1, 1, length, length).masked_fill(~visible, torch.finfo(torch.float32).min)


@pytest.mark.parametrize("batch", [1, 2])
def test_full_cache_matches_native_prefill_and_continuations(model, batch):
    tokens = torch.arange(1, batch * 6 + 1).reshape(batch, 6)
    adapter = ZetaLlamaAdapter(model)
    actual = torch.cat(
        [
            adapter.forward(tokens[:, :1], attention_mask=torch.ones(batch, 1)),
            adapter.forward(
                tokens[:, 1:4],
                position_ids=torch.arange(1, 4).expand(batch, -1),
            ),
            adapter.forward(tokens[:, 4:5]),
            adapter.forward(tokens[:, 5:]),
        ],
        dim=1,
    )
    with torch.inference_mode():
        expected = model(tokens, attention_mask=torch.ones_like(tokens)).logits
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for layer in adapter.cache.layers:
        assert layer.keys.shape == (batch, 2, 6, 8)
        assert layer.values.shape == (batch, 2, 6, 8)
        assert layer.positions.tolist() == list(range(6))


@pytest.mark.parametrize(("window", "sinks"), [(1, 0), (3, 2), (3, 6)])
def test_bounded_cache_matches_native_dense_window(model, window, sinks):
    tokens = torch.arange(1, 17).reshape(2, 8)
    adapter = ZetaLlamaAdapter(model, window=window, sink_tokens=sinks)
    actual = adapter.forward(tokens)
    with torch.inference_mode():
        expected = model(
            tokens,
            attention_mask=dense_window_mask(8, window, sinks).expand(2, -1, -1, -1),
            position_ids=torch.arange(8).expand(2, -1),
            use_cache=False,
        ).logits
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    positions = [pos for pos in range(8) if pos < sinks or pos >= 8 - window]
    for layer in adapter.cache.layers:
        assert layer.positions.tolist() == positions
        assert layer.next_position == 8
        assert layer.keys.shape == (2, 2, len(positions), 8)
        assert torch.isfinite(layer.keys).all() and torch.isfinite(layer.values).all()


def test_eos_stops_each_sequence_and_seals_batch(model):
    with torch.inference_mode():
        candidates = torch.arange(1, 20).reshape(-1, 1)
        choices = model(candidates).logits[:, -1].argmax(-1)
    first = 0
    second = next(i for i in range(1, len(choices)) if choices[i] != choices[first])
    prompts = candidates[[first, second]]
    eos = int(choices[first])
    with torch.inference_mode():
        expected = model.generate(
            prompts,
            attention_mask=torch.ones_like(prompts),
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=eos,
        )[:, 1:].tolist()
    expected = [row[: row.index(eos) + 1] if eos in row else row for row in expected]
    adapter = ZetaLlamaAdapter(model)
    assert adapter.generate(prompts, max_new_tokens=3, eos_token_id=eos) == expected
    assert expected[0] == [eos] and expected[1][0] != eos
    with pytest.raises(RuntimeError, match="cannot be continued"):
        adapter.forward(prompts)


def test_rejects_padding_positions_batch_changes_and_overflow(model):
    adapter = ZetaLlamaAdapter(model, window=3, sink_tokens=1)
    with pytest.raises(ValueError, match="unpadded"):
        adapter.forward(torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 0]]))
    with pytest.raises(ValueError, match="position_ids"):
        adapter.forward(torch.tensor([[1]]), position_ids=torch.tensor([[1]]))
    adapter.forward(torch.tensor([[1]]))
    with pytest.raises(ValueError, match="batch size"):
        adapter.forward(torch.tensor([[1], [2]]))
    with pytest.raises(ValueError, match="position limit"):
        adapter.forward(torch.ones(1, 32, dtype=torch.long))


def test_layer_failure_invalidates_cache(model, monkeypatch):
    adapter = ZetaLlamaAdapter(model)
    layer = adapter.cache.layers[1]

    def fail(*args, **kwargs):
        raise RuntimeError("forced layer error")

    monkeypatch.setattr(layer, "append", fail)
    with pytest.raises(RuntimeError, match="forced layer error"):
        adapter.forward(torch.tensor([[1]]))
    with pytest.raises(RuntimeError, match="invalid"):
        adapter.forward(torch.tensor([[1]]))
