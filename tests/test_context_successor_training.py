import pytest
import torch

from benchmarks.context_successor import METHODS, candidate, train_successor


@pytest.mark.parametrize("method", METHODS)
def test_successor_baselines_train_and_use_requested_evaluation_variant(method):
    model = train_successor(method, seed=100, steps=1, batch_size=2, settings=candidate(method, 0))
    assert not model.training
    assert model.variant == method
    assert all(block.variant == method for block in model.blocks)
    if method in ("rope-linear", "rope-ntk"):
        assert all(block.rope_scale == 4.0 for block in model.blocks)
    with torch.no_grad():
        assert torch.isfinite(model(torch.zeros(1, 16, dtype=torch.long))).all()
