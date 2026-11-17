# Adapter design

This is an isolated, opt-in prototype for the pinned SmolLM2 checkpoint and
Transformers 4.52.2 Llama eager attention. It does not patch the model or
replace its attention. The original model's Q/K/V projections, split-half RoPE,
GQA expansion, scaled dot-product attention, output projections and weights
are used. BF16 checkpoint values are loaded as FP32; the conversion preserves
those values exactly. No training is performed.

## Cache and positions

`ZetaLlamaCache` implements the Transformers `Cache` interface and owns one
existing `StreamingKVCache` per decoder layer. The native Llama attention
applies RoPE to keys before calling `update`; the adapter stores those
already-rotated keys without applying Zeta's different rotary convention.
Each layer stores `[batch, 5, retained_tokens, 64]` keys and values for this
model. Native eager attention expands the five KV heads to 15 query heads;
the adapter does not invent a GQA mapping or store expanded heads.

The cache accepts one position per update and validates the incoming absolute
`cache_position` against the per-layer next-position counter. It retains
initial sinks and the recent window using the existing Zeta policy. Physical
cache size and logical position are distinct: this prototype's
`get_seq_length` returns the logical position and must not be used as a
generic replacement for other Transformers cache implementations. The
adapter passes absolute `cache_position` and `position_ids` explicitly and
checks that all layers retain the same positions after every token.

Full-cache mode uses a window equal to the model's configured position limit
and never evicts within that limit. Bounded mode uses the supplied positive
window and nonnegative sink count; overlap is retained once. Both modes reject
positions beyond the configured 8,192-token limit. This deliberate guard means
the prototype cannot currently test continuation past that limit.

## Masks, calls and limitations

Every model call processes **one token**. A multi-token `forward` is sequential
prefill, not a native batched prefill. For each token, the adapter constructs
an explicit zero-valued 4D additive mask of the retained-key width; every
retained key is causal because the cache contains only earlier keys and the
current key. This avoids treating discontiguous sink/recent positions as a
contiguous slice. The existing cache selects those keys, and native attention
consumes them. A separate dense native-attention oracle applies the same
absolute-position causal sink/window policy to the entire sequence.

Multiple calls on the same adapter continue the same caches. A batch must have
equal logical positions, fixed size and no padding; provided masks must be all
ones for the current inputs, and explicit position IDs must match the shared
absolute positions. Unsupported padding, changed batch size, beam reordering,
non-eager backends, incompatible Transformers versions and positions beyond
the model limit are rejected. Partial layer updates after an exception or
interruption invalidate the adapter. There is no cache serialization, beam
support, efficient multi-token prefill or independent per-row reset.

`generate` performs greedy decoding with independent EOS stopping. Finished
rows receive dummy EOS tokens while other rows continue; their output is not
extended. The adapter is sealed after generation, so such padded cache rows
cannot later be reused as real conversation history. The lower-level `forward`
API treats EOS as an ordinary input token and supports continuation across
calls. This is not a general `model.generate` cache implementation.

Evicted KV entries and token identities are not directly retrievable. Retained
later states may contain indirect traces, but arbitrary recovery is neither
implemented nor established.
