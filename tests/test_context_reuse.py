import torch

from benchmarks.context_char import train_char
from benchmarks.context_needle_tuned import train_needle
from benchmarks.context_reuse import reuse_rope_trial
from benchmarks.context_successor import candidate


def test_rope_training_reuse_matches_independent_scaled_training():
    cache = {}
    calls = []

    def run(method, index):
        settings = candidate(method, index)

        def train():
            calls.append(method)
            return train_needle(method, seed=100, steps=1, batch_size=2, settings=settings)

        return reuse_rope_trial(cache, method, 100, settings, train)

    run("rope-full", 1)
    tokens = torch.arange(300).unsqueeze(0) % 97
    for method in ("rope-linear", "rope-ntk"):
        reused = run(method, 0)
        independent = train_needle(
            method, seed=100, steps=1, batch_size=2, settings=candidate(method, 0)
        )
        with torch.no_grad():
            torch.testing.assert_close(reused(tokens), independent(tokens))
    assert calls == ["rope-full"]


def test_rope_training_reuse_does_not_merge_other_hyperparameters():
    cache = {}
    calls = []
    model = object()

    def train():
        calls.append(True)
        return model

    assert reuse_rope_trial(cache, "rope-full", 42, candidate("rope-full", 0), train) is model
    assert reuse_rope_trial(cache, "rope-full", 42, candidate("rope-full", 0), train) is model
    assert len(calls) == 2
    assert not cache


def test_character_rope_reuse_matches_independent_training():
    data = torch.arange(500) % 11
    settings = candidate("rope-ntk", 0)
    cache = {}

    def train():
        return train_char(data, 11, "rope-full", 100, 1, 256, 2, 64, 4)

    reused = reuse_rope_trial(cache, "rope-ntk", 100, settings, train)
    independent = train()
    independent.variant = "rope-ntk"
    for block in independent.blocks:
        block.variant = "rope-ntk"
        block.rope_scale = settings["rope_scale"]
    tokens = data[:300].unsqueeze(0)
    with torch.no_grad():
        torch.testing.assert_close(reused(tokens), independent(tokens))
