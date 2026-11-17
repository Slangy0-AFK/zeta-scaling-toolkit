"""Explicit native-RoPE position experiment; the original adapter is unchanged."""

import importlib.util
from pathlib import Path
from typing import Optional

ADAPTER_PATH = Path(__file__).resolve().parents[1] / "smolLM2-integration-v1" / "adapter.py"
spec = importlib.util.spec_from_file_location("zeta_native_llama_adapter", ADAPTER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load original adapter: {ADAPTER_PATH}")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class ExperimentalPositionAdapter(adapter.ZetaLlamaAdapter):
    """Opt in to native unscaled RoPE above the configured position limit.

    A nonzero initial position is a synthetic offset with an empty history, not
    a reconstruction of preceding tokens. It is intended for bounded tests.
    """

    def __init__(
        self,
        model,
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
            or not original_max < max_positions <= 65537
        ):
            raise ValueError("max_positions must exceed the configured limit and be <= 65537")
        if (
            isinstance(initial_position, bool)
            or not isinstance(initial_position, int)
            or not 0 <= initial_position < max_positions
        ):
            raise ValueError("initial_position must be in the experimental range")
        if initial_position and sink_tokens:
            raise ValueError("an empty offset history cannot contain initial sink tokens")
        if model.config.rope_scaling is not None or model.model.rotary_emb.rope_type != "default":
            raise ValueError("only the unmodified native default RoPE is supported")
        super().__init__(model, window=window, sink_tokens=sink_tokens)
        self.max_positions = max_positions
        if window is None:
            self.cache = adapter.ZetaLlamaCache(
                model.config.num_hidden_layers, max_positions, sink_tokens
            )
        for layer in self.cache.layers:
            layer.next_position = initial_position
        self.initial_position = initial_position

    def manifest(self) -> dict:
        return {
            "mode": "experimental_native_unscaled_rope",
            "original_max_positions": int(self.model.config.max_position_embeddings),
            "experimental_max_positions": self.max_positions,
            "initial_position": self.initial_position,
            "synthetic_empty_offset_history": self.initial_position != 0,
            "window": self.cache.layers[0].window,
            "sink_tokens": self.cache.layers[0].sink_tokens,
            "rope_theta": float(self.model.config.rope_theta),
            "rope_scaling": None,
        }
