"""Opt-in native Llama inference with Zeta's per-layer streaming KV cache.

This prototype targets Transformers 4.52.2 eager attention. It does not patch
the model or implement attention. A call may contain several tokens, but each
is evaluated separately because StreamingKVCache appends one position.
"""

from typing import Optional

import torch
import transformers
from transformers.cache_utils import Cache

from zeta_scaling_toolkit import StreamingKVCache


class ZetaLlamaCache(Cache):
    """Transformers cache backed by one Zeta cache per decoder layer."""

    def __init__(self, num_layers: int, window: int, sink_tokens: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.layers = [StreamingKVCache(window, sink_tokens) for _ in range(num_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < len(self.layers):
            raise ValueError("invalid decoder layer")
        layer = self.layers[layer_idx]
        position = None if cache_kwargs is None else cache_kwargs.get("cache_position")
        if (
            not isinstance(position, torch.Tensor)
            or position.dtype != torch.long
            or position.shape != (1,)
            or int(position.item()) != layer.next_position
        ):
            raise ValueError("cache_position must match the next absolute position")
        keys, values, _ = layer.append(key_states, value_states)
        return keys, values

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.layers[0 if layer_idx is None else layer_idx].next_position

    def get_max_cache_shape(self) -> None:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError("beam search is not supported")

    def retained_positions(self, next_position: int, device: torch.device) -> torch.Tensor:
        layer = self.layers[0]
        previous = layer.positions
        positions = torch.tensor([next_position], dtype=torch.long, device=device)
        if previous is not None:
            positions = torch.cat((previous, positions))
        keep = (positions < layer.sink_tokens) | (positions >= next_position - layer.window + 1)
        return positions[keep]

    def storage_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            for tensor in (layer.keys, layer.values, layer.positions):
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
        return total


class ZetaLlamaAdapter:
    """Run a batch of equally positioned, unpadded sequences in evaluation mode."""

    def __init__(
        self, model: torch.nn.Module, *, window: Optional[int] = None, sink_tokens: int = 0
    ) -> None:
        if transformers.__version__ != "4.52.2":
            raise RuntimeError("this prototype requires transformers==4.52.2")
        if model.training:
            raise ValueError("model must be in evaluation mode")
        config = model.config
        if config.model_type != "llama" or config._attn_implementation != "eager":
            raise ValueError("native Llama eager attention is required")
        self.model = model
        self.full_cache = window is None
        self.max_positions = int(config.max_position_embeddings)
        self.cache = ZetaLlamaCache(
            config.num_hidden_layers, self.max_positions if window is None else window, sink_tokens
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
        if input_ids.device != next(self.model.parameters()).device:
            raise ValueError("input_ids must be on the model device")
        if attention_mask is not None and (
            attention_mask.shape != input_ids.shape
            or attention_mask.device != input_ids.device
            or not bool(torch.all(attention_mask == 1))
        ):
            raise ValueError("only unpadded attention masks for the current inputs are supported")
        start = self.cache.get_seq_length()
        if any(layer.next_position != start for layer in self.cache.layers):
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
                    dtype=next(self.model.parameters()).dtype,
                    device=input_ids.device,
                )
                output = self.model(
                    input_ids=input_ids[:, offset : offset + 1],
                    attention_mask=mask,
                    position_ids=expected[offset : offset + 1].expand(batch, 1),
                    cache_position=expected[offset : offset + 1],
                    past_key_values=self.cache,
                    use_cache=True,
                )
                if any(
                    layer.next_position != position + 1
                    or not torch.equal(layer.positions, retained)
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
        """Greedily generate independently stopped rows; the batch is sealed afterwards."""
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
