"""Transformers 5.23.0 Llama adapter with an explicit position-extension opt-in."""

from typing import Optional

import torch
import transformers
from transformers.cache_utils import Cache, CacheLayerMixin

from zeta_scaling_toolkit import StreamingKVCache

TRANSFORMERS_VERSION = "5.23.0"


class ZetaCacheLayer(CacheLayerMixin):
    """Transformers 5.x cache layer backed by Zeta's streaming KV cache."""

    is_sliding = False

    def __init__(self, window: int, sink_tokens: int, max_positions: int) -> None:
        super().__init__()
        self.storage = StreamingKVCache(window, sink_tokens)
        self.max_positions = max_positions

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.storage.next_position >= self.max_positions:
            raise ValueError("input exceeds the configured position limit")
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        keys, values, _ = self.storage.append(key_states, value_states)
        self.keys, self.values = keys, values
        return keys, values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        retained = 0 if self.storage.positions is None else self.storage.positions.numel()
        return retained + query_length, 0

    def get_seq_length(self) -> int:
        return self.storage.next_position

    def get_max_length(self) -> int:
        return self.max_positions

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.storage.keys is None:
            return
        self.storage.keys = self.storage.keys.index_select(0, beam_idx.to(self.storage.keys.device))
        assert self.storage.values is not None
        self.storage.values = self.storage.values.index_select(
            0, beam_idx.to(self.storage.values.device)
        )
        self.keys, self.values = self.storage.keys, self.storage.values

    def reset(self) -> None:
        self.storage.clear()
        self.keys = self.values = None
        self.is_initialized = False


class ZetaLlamaCache(Cache):
    """Per-layer streaming cache implementing the Transformers 5.x Cache contract."""

    def __init__(self, num_layers: int, window: int, sink_tokens: int, max_positions: int) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        super().__init__(
            layers=[ZetaCacheLayer(window, sink_tokens, max_positions) for _ in range(num_layers)]
        )

    def retained_positions(self, next_position: int, device: torch.device) -> torch.Tensor:
        layer = self.layers[0]
        previous = layer.storage.positions
        positions = torch.tensor([next_position], dtype=torch.long, device=device)
        if previous is not None:
            positions = torch.cat((previous, positions))
        keep = (positions < layer.storage.sink_tokens) | (
            positions >= next_position - layer.storage.window + 1
        )
        return positions[keep]

    def storage_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            for tensor in (layer.storage.keys, layer.storage.values, layer.storage.positions):
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
        return total

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()


class ZetaLlamaAdapter:
    """Run native eager Llama attention with a bounded or full Zeta cache."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        window: Optional[int] = None,
        sink_tokens: int = 0,
    ) -> None:
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(f"this adapter requires transformers=={TRANSFORMERS_VERSION}")
        if model.training:
            raise ValueError("model must be in evaluation mode")
        config = model.config
        if config.model_type != "llama" or config._attn_implementation != "eager":
            raise ValueError("native Llama eager attention is required")
        self.model = model
        self.full_cache = window is None
        self.max_positions = int(config.max_position_embeddings)
        cache_window = self.max_positions if window is None else window
        self.cache = ZetaLlamaCache(
            config.num_hidden_layers, cache_window, sink_tokens, self.max_positions
        )
        self.batch_size: Optional[int] = None
        self.valid = True
        self.sealed = False

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.valid:
            raise RuntimeError("adapter state is invalid after an interrupted forward pass")
        if self.sealed:
            raise RuntimeError("generation finished; this batch cache cannot be continued")
        if self.model.training:
            raise ValueError("model must remain in evaluation mode")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or not input_ids.numel():
            raise ValueError("input_ids must be a nonempty rank-2 int64 tensor")
        batch, length = input_ids.shape
        if self.batch_size is not None and batch != self.batch_size:
            raise ValueError("batch size cannot change while continuing a cache")
        parameter = next(self.model.parameters())
        if input_ids.device != parameter.device:
            raise ValueError("input_ids must be on the model device")
        if attention_mask is not None and (
            attention_mask.shape != input_ids.shape
            or attention_mask.device != input_ids.device
            or not bool(torch.all(attention_mask == 1))
        ):
            raise ValueError("only unpadded attention masks for the current inputs are supported")
        start = self.cache.get_seq_length()
        if any(layer.storage.next_position != start for layer in self.cache.layers):
            raise RuntimeError("per-layer cache positions are inconsistent")
        if start + length > self.max_positions:
            raise ValueError("input exceeds the model's configured position limit")
        expected = torch.arange(start, start + length, dtype=torch.long, device=input_ids.device)
        if position_ids is not None and (
            position_ids.shape != input_ids.shape
            or position_ids.dtype != torch.long
            or position_ids.device != input_ids.device
            or not torch.equal(position_ids, expected.expand(batch, -1))
        ):
            raise ValueError("position_ids must match the shared absolute positions")
        self.batch_size = batch
        outputs = []
        completed = False
        try:
            for offset in range(length):
                position = start + offset
                retained = self.cache.retained_positions(position, input_ids.device)
                mask = torch.zeros(
                    (batch, 1, 1, retained.numel()),
                    dtype=parameter.dtype,
                    device=input_ids.device,
                )
                output = self.model(
                    input_ids=input_ids[:, offset : offset + 1],
                    attention_mask=mask,
                    position_ids=expected[offset : offset + 1].expand(batch, 1),
                    past_key_values=self.cache,
                    use_cache=True,
                )
                if any(
                    layer.storage.next_position != position + 1
                    or not torch.equal(layer.storage.positions, retained)
                    for layer in self.cache.layers
                ):
                    raise RuntimeError("per-layer cache positions or retained keys differ")
                if not torch.isfinite(output.logits).all():
                    raise RuntimeError("non-finite model logits")
                outputs.append(output.logits)
            completed = True
        finally:
            if not completed:
                self.valid = False
        return torch.cat(outputs, dim=1)

    def generate(
        self, input_ids: torch.Tensor, *, max_new_tokens: int, eos_token_id: int
    ) -> list[list[int]]:
        """Greedily generate independently stopped rows and seal the resulting cache."""
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            isinstance(eos_token_id, bool)
            or not isinstance(eos_token_id, int)
            or not 0 <= eos_token_id < self.model.config.vocab_size
        ):
            raise ValueError("eos_token_id must be a valid token ID")
        logits = self.forward(input_ids)[:, -1]
        batch = input_ids.shape[0]
        finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        sequences: list[list[int]] = [[] for _ in range(batch)]
        for step in range(max_new_tokens):
            tokens = logits.argmax(dim=-1)
            tokens = torch.where(finished, torch.full_like(tokens, eos_token_id), tokens)
            for index in range(batch):
                if not bool(finished[index]):
                    sequences[index].append(int(tokens[index]))
            finished |= tokens == eos_token_id
            if bool(finished.all()) or step + 1 == max_new_tokens:
                break
            logits = self.forward(tokens[:, None])[:, -1]
        self.sealed = True
        return sequences


class ExperimentalPositionAdapter(ZetaLlamaAdapter):
    """Opt in to native unscaled RoPE beyond the model's configured limit."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_positions: int,
        window: Optional[int] = None,
        sink_tokens: int = 0,
        initial_position: int = 0,
    ) -> None:
        original_max = int(model.config.max_position_embeddings)
        if (
            isinstance(max_positions, bool)
            or not isinstance(max_positions, int)
            or not original_max < max_positions <= torch.iinfo(torch.int64).max
        ):
            raise ValueError("max_positions must exceed the configured limit and fit in int64")
        if (
            isinstance(initial_position, bool)
            or not isinstance(initial_position, int)
            or not 0 <= initial_position < max_positions
        ):
            raise ValueError("initial_position must be in the experimental range")
        if initial_position and sink_tokens:
            raise ValueError("an empty offset history cannot contain initial sink tokens")
        rope_parameters = model.config.rope_parameters
        is_native_rope = (
            isinstance(rope_parameters, dict)
            and rope_parameters.get("rope_type") == "default"
            and set(rope_parameters).issubset({"rope_type", "rope_theta"})
            and "rope_theta" in rope_parameters
        )
        if not is_native_rope or model.model.rotary_emb.rope_type != "default":
            raise ValueError("only the unmodified native default RoPE is supported")
        if float(rope_parameters["rope_theta"]) <= 1:
            raise ValueError("native RoPE base must be greater than one")
        super().__init__(model, window=window, sink_tokens=sink_tokens)
        cache_window = self.max_positions if window is None else window
        self.max_positions = max_positions
        self.cache = ZetaLlamaCache(
            model.config.num_hidden_layers, cache_window, sink_tokens, max_positions
        )
        for layer in self.cache.layers:
            layer.storage.next_position = initial_position
        self.initial_position = initial_position

    def manifest(self) -> dict[str, object]:
        layer = self.cache.layers[0].storage
        return {
            "mode": "experimental_native_unscaled_rope",
            "original_max_positions": int(self.model.config.max_position_embeddings),
            "experimental_max_positions": self.max_positions,
            "initial_position": self.initial_position,
            "synthetic_empty_offset_history": self.initial_position != 0,
            "window": layer.window,
            "sink_tokens": layer.sink_tokens,
            "rope_theta": float(self.model.config.rope_parameters["rope_theta"]),
            "rope_scaling": None,
        }
