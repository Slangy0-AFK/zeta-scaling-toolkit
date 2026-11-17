# Zeta integration audit

## What exists

The project is a PyTorch component toolkit, not an integrated pretrained
conversation engine. [NonCollapsingAttention](../../zeta_scaling_toolkit/core.py)
consumes projected Q/K/V and produces attention outputs. It normalizes Q and K,
uses a bounded temperature parameter and optionally scales it with visible
length. `forward_streaming` applies sink-plus-recent-window visibility to
Q/K/V and computes chunked attention. Substituting it for a pretrained model's
ordinary scaled-dot-product attention changes attention mathematics; the
checkpoint has no trained Zeta temperature. Freezing a newly initialized
temperature preserves pretrained weights but is not behavioral equivalence.

[StreamingKVCache](../../zeta_scaling_toolkit/positions.py) operates on cached
key/value tensors, one appended position at a time, and stores detached tensors
plus absolute int64 positions. It retains initial sinks and the most recent
window, deduplicating overlaps. It is not a cache wrapper already connected to
the pretrained model. [Existing tests](../../tests/test_properties.py)
exercise retention and agreement with a dense mask for synthetic tensors, not
pretrained-model generation, GQA integration or multi-turn resumption.

## Contracts and incompatibilities

- **Heads:** Q/K/V batch dimensions must broadcast. A GQA head dimension of
  15 queries versus 5 KV heads does not broadcast directly. Keep each layer's
  cache at 5 KV heads and repeat/group KV as the native backend does before
  attention, or implement an equivalent tested grouped kernel. MQA's single
  KV head can broadcast, but backend layout and mapping must still be tested.
  Storing expanded heads is correct only if mapping is right and wastes memory.
- **RoPE:** Zeta's `rotary_embedding` uses interleaved even/odd pairs; the
  candidate Transformers Llama and Qwen2 implementations use split-half
  rotation. Their base values also differ from Zeta's default. Do not apply
  Zeta's rotary function to pretrained projections or rotate cached keys
  twice. Let the original model apply its own RoPE at original absolute
  positions, then cache those already-rotated keys.
- **Positions:** Cache positions are absolute and monotonically increase even
  after eviction; physical cache length is not the next position. A separate
  per-sequence counter and retained-position map are essential. Existing
  `append` generates positions internally, expects one token and shares the
  same positions across its batch: variable-length padded batches and beam
  reordering need additional handling or explicit rejection. Empty/reset
  state, fresh conversations and generation continuation must be explicit.
- **Masks:** Native causal/padding masks and custom sparse ordering must match
  the retained absolute key positions, not be sliced as if keys were
  contiguous. Prefill must apply causality independently for every query,
  including the first time eviction occurs. The current streaming attention
  assumes query positions align with the final contiguous key positions and
  its mask has no general padding input. It cannot simply be called on a
  discontiguous compressed cache with multi-token queries.
- **Persistence:** Cache instances can persist across calls if explicitly
  retained and only newly processed tokens are appended. Today no pretrained
  generation loop implements that contract. Evicted KV entries and their raw
  token identities are not directly retrievable from the cache. Some
  information could be indirectly encoded in retained later hidden states;
  neither exact recall nor complete loss follows automatically.

In Transformers v4.52.2, [Llama attention](https://github.com/huggingface/transformers/blob/v4.52.2/src/transformers/models/llama/modeling_llama.py)
rotates Q/K before calling the cache update, then expands KV for GQA.
[Cache documentation](https://huggingface.co/docs/transformers/v4.52.2/en/cache_explanation)
explains `cache_position` and mask responsibilities. This is a versioned
reference, not proof that the adapter works with any installed backend.

## Minimum future integration

Build an opt-in, inference-only, per-layer adapter around the actual
`StreamingKVCache` and the model's native attention path. Initially support
batch size one, greedy decoding and one token at a time for correctness;
explicitly reject unsupported padding, beam search and incompatible cache
APIs. Preserve the original model's projections, RoPE and dot-product
attention; use retained absolute positions to implement visibility. Track
logical length separately from cache size and verify all layers agree.
This changes inference visibility, not checkpoint weights or network
dimensions. Efficient multi-token prefill is a later separately tested
optimization.

Before any boundary study, test full-cache parity with native inference before
eviction; sink/window outputs against an independent native-attention dense
mask for multiple steps and multi-turn continuation; GQA mapping; RoPE and
absolute positions across eviction; causal masks; reset; overflow; and explicit
unsupported-case errors. Include checkpoint/tokenizer hashes and cache
configuration in the experiment manifest. A separate non-collapsing-attention
intervention, if ever investigated, needs its own frozen temperature and
controls; do not silently conflate it with the cache intervention.
