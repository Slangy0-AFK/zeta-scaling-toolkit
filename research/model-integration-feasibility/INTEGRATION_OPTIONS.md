# Integration options

## A. Attention / KV-cache path

**Recommended first intervention:** insert the existing
`StreamingKVCache` per layer into the pretrained model's native inference
path. For every incoming token, retain sinks plus recent keys and values and
perform the model's original attention over exactly those visible positions.
The same input stream enters both conditions, but which previous tokens remain
visible to later attention differs. The checkpoint, tokenizer, RoPE, projections
and network dimensions remain unchanged. The cache is real Zeta code, not a
new retrieval system.

Complexity is moderate to high: a version-specific adapter must reconcile
GQA, cache interfaces, RoPE, masks, positions, prefill and multi-turn state.
The model was pretrained with full attention, not this sparse visibility.
Window eviction can remove needed evidence, and cache compaction does not
guarantee that absolute RoPE positions beyond the advertised length are
supported or useful. If a hard position guard prevents continuation, do not
remove it silently; report that blocker. A cache can reduce active KV storage,
but actual latency, total memory and quality are unmeasured.

Optionally replacing native attention with `NonCollapsingAttention` is a
different intervention: normalized Q/K and a new Zeta temperature alter model
behavior even without eviction. No pretrained temperature exists. It requires
a preregistered temperature and separate controls and is not the minimal
first integration.

## B. External context-management layer

Keep the model and native attention unchanged and place an orchestration layer
around inference. It could maintain conversation tokens and select or compress
what gets sent on each call. A retained transcript alone does not give the
model access to evicted facts. The existing Zeta `StreamingKVCache` stores
per-layer tensors, not text, semantic summaries or queryable facts, so it
cannot directly supply such a layer. Using ordinary truncation, summarization
or retrieval instead would be a **new comparator**, not an implementation of
the existing Zeta mechanism.

The wrapper can be simpler for a conventional text window but needs a
separately designed state/update/retrieval mechanism to preserve nonlocal
facts. It changes the effective prompt and can omit, contradict or leak
information; it cannot overcome an underlying model's token/position limits
without changing what the model sees. An external controller could in
principle carry a genuine per-layer Zeta cache across model calls, but it
still needs option A's cache-path integration; merely passing shorter text
is not equivalent.

## Scientific comparison

Option A gives the cleanest first comparison: same model weights and native
attention, with only the use of the existing Zeta sink/window cache changed.
Use the ordinary full KV cache in the baseline, not an intentionally
handicapped no-cache or reduced-window baseline. Compare both quality and
cost, including baseline caching; verify numerical parity when Zeta's cache
retains all positions. This is a window-visibility intervention, not a
demonstration of semantic persistent memory. A conventional external
management system can be added later as a separately labeled comparison.
