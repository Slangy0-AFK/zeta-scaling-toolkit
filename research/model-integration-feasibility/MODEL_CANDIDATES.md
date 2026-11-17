# Pretrained conversational model candidates

Status: desk study, 2026-11-16. No weights were downloaded or model inference
performed. Context lengths below are published configuration limits, not
measured failure boundaries. Model availability and licenses were checked in
the public model cards and metadata on this date; verify them again before use.

| Model | Architecture and attention | Published context | Tokenizer and weights | License |
|---|---|---:|---|---|
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/a10cc1512eabd3dde888204e902eca88bddb4951) | 360M decoder, Llama implementation, RoPE, 32 layers, 15 query / 5 KV heads, head dimension 64 | 8,192 | Model-specific GPT-2-style tokenizer and chat template; public, ungated safetensors | Apache-2.0 |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) | 0.49B Qwen2 decoder, RoPE, 24 layers, 14 query / 2 KV heads, head dimension 64 | 32,768 | Qwen2 tokenizer and chat template; public, ungated safetensors | Apache-2.0 |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) | 1.54B Qwen2 decoder, RoPE, 28 layers, 12 query / 2 KV heads, head dimension 128 | 32,768 | Qwen2 tokenizer and chat template; public, ungated safetensors | Apache-2.0 |

Sources: [SmolLM2 card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/README.md)
and [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/config.json);
[Qwen 0.5B card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/README.md)
and [configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json);
[Qwen 1.5B card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/README.md)
and [configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/config.json).
The Qwen cards state 32,768 for these specific models despite broader family
claims; the 0.5B tokenizer metadata advertises 131,072, which must not be
mistaken for this model's configured length. SmolLM2's tokenizer advertises
8,192. Each model's chat template and special tokens must be pinned.

All three use grouped-query attention (GQA). In a compatible Transformers
implementation, each layer caches keys and values shaped approximately
`[batch, kv_heads, cached_positions, head_dim]`; the attention kernel repeats
each KV head for its query-head group. This is a runtime interface, not a
property of the weight file. Transformers versions and backends differ; see
the [Llama implementation at v4.52.2](https://github.com/huggingface/transformers/blob/v4.52.2/src/transformers/models/llama/modeling_llama.py)
and [Qwen2 implementation at v4.52.2](https://github.com/huggingface/transformers/blob/v4.52.2/src/transformers/models/qwen2/modeling_qwen2.py).

Compatibility is plausible, not demonstrated: all need the per-layer cache,
absolute-position and mask adapter described in
[ZETA_INTEGRATION_AUDIT.md](./ZETA_INTEGRATION_AUDIT.md). Nominal CPU weight
and KV estimates, and the difference between loading and a feasible study, are
in [CPU_FEASIBILITY.md](./CPU_FEASIBILITY.md). SmolLM2 is the smallest
candidate and has the shortest advertised context, making it the best initial
engineering target; its limited instruction and reasoning capability may make
the scientific task infeasible even when integration works. Qwen 0.5B has a
smaller KV cache per token but a much longer nominal boundary. The 1.5B model
offers a larger model at substantially greater weight and KV cost, with no
established advantage on this task.
