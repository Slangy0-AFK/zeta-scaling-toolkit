"""Reuse identical RoPE training runs across inference-only scaling candidates."""

from collections.abc import Callable

if __package__:
    from .length_generalization import SequenceLM
else:
    from length_generalization import SequenceLM


def reuse_rope_trial(
    cache: dict[int, SequenceLM],
    method: str,
    seed: int,
    settings: dict[str, float],
    train: Callable[[], SequenceLM],
) -> SequenceLM:
    if (
        method not in ("rope-full", "rope-linear", "rope-ntk")
        or settings["learning_rate"] != 0.001
        or settings["tau0"] != 10.0
    ):
        return train()
    if seed not in cache:
        cache[seed] = train()
    model = cache[seed]
    model.variant = method
    for block in model.blocks:
        block.variant = method
        block.rope_scale = settings["rope_scale"]
    return model
