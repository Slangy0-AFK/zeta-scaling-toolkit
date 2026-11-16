import copy

import pytest
import torch
import torch.nn.functional as F

from benchmarks.context_ability import VOCABULARY_SIZE, planted_needle_batch
from benchmarks.length_generalization import SequenceLM


@pytest.mark.parametrize("length", [8, 256])
def test_last_rope_query_matches_full_output_and_parameter_gradients(length):
    torch.manual_seed(93)
    full = SequenceLM(VOCABULARY_SIZE, "rope-full", False, 256, 10.0, 10000.0)
    fast = copy.deepcopy(full)
    tokens, targets = planted_needle_batch(2, length, torch.Generator().manual_seed(7))
    expected = full(tokens)[:, -1]
    actual = fast.forward_last(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    F.cross_entropy(expected, targets).backward()
    F.cross_entropy(actual, targets).backward()
    for expected_parameter, actual_parameter in zip(full.parameters(), fast.parameters()):
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, atol=1e-6, rtol=1e-5
        )


def test_last_rope_query_preserves_short_training_trajectory():
    torch.manual_seed(94)
    full = SequenceLM(VOCABULARY_SIZE, "rope-full", False, 256, 10.0, 10000.0)
    fast = copy.deepcopy(full)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001) for model in (full, fast)]
    generator = torch.Generator().manual_seed(8)
    for _ in range(10):
        tokens, targets = planted_needle_batch(4, 256, generator)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(full(tokens)[:, -1], targets).backward()
        F.cross_entropy(fast.forward_last(tokens), targets).backward()
        for optimizer in optimizers:
            optimizer.step()
    for expected, actual in zip(full.parameters(), fast.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "variant", ["abs", "alibi", "rope-log-full", "rope-window", "rope-log-window"]
)
def test_last_query_keeps_other_variants_unchanged(variant):
    torch.manual_seed(95)
    model = SequenceLM(VOCABULARY_SIZE, variant, False, 256, 10.0, 10000.0)
    tokens = torch.arange(80).unsqueeze(0) % VOCABULARY_SIZE
    torch.testing.assert_close(model.forward_last(tokens), model(tokens)[:, -1])
