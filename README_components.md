## Components (v0.2.0)

- `NonCollapsingAttention`: QK-normalised attention, `logits = tau * cos(q, k)`, with a learnable temperature `tau` (softplus, starts at 10, hard-capped at `tau_max=50`). Float32 internals, optional `attn_mask`, raises on `d_k` mismatch. `A_attn` is deprecated and ignored. **Breaking:** state dict now has `raw_tau` instead of `q_scale`/`k_scale` (legacy checkpoints load with a warning; `raw_tau` keeps its init).
- `AsymptoticLRScheduler`: linear warmup (`warmup_steps`, default 0), then cosine decay to `floor_ratio * base_lr` at `total_steps`, then a hard clamp. This is cosine annealing with a minimum LR, not a true asymptote. Arguments are validated.
- `get_isotropic_coords` / `get_scale_invariant_vit_coords`: pixel-centred, image-centred coordinates divided by `max(h, w)`, so aspect ratio is preserved (v0.1 stretched it). `smooth_angle=True` returns `(r, cos, sin)` to avoid the jump at +-pi. `T` and `A_coords` are deprecated and ignored.
- `make_corrected_linear`: orthogonal init scaled by `calculate_gain(nonlinearity)` (sqrt(2) for ReLU). Pass `nonlinearity="linear"` for exact isometry. Layers with `out < in` shrink norms by about `sqrt(out/in)`.

None of this is validated on real training runs; `tests/` checks numerical properties only.
