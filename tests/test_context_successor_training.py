import pytest
import torch

from benchmarks.context_successor import (
    METHODS,
    candidate,
    select_candidates,
    successor_candidate,
    train_successor,
)


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


def test_successor_candidate_adds_extended_tuning_options_without_changing_originals():
    assert successor_candidate("alibi", 0) == candidate("alibi", 0)
    assert successor_candidate("rope-log-window", 1) == candidate("rope-log-window", 1)
    assert successor_candidate("alibi", 2) == {
        "learning_rate": 0.0015,
        "tau0": 10.0,
        "rope_scale": 1.0,
    }
    assert successor_candidate("rope-linear", 2)["rope_scale"] == 8.0
    assert successor_candidate("rope-log-full", 2)["tau0"] == 15.0
    with pytest.raises(ValueError, match="unknown method or candidate"):
        successor_candidate("alibi", 3)


def test_extended_successor_candidate_trains():
    settings = successor_candidate("alibi", 2)
    model = train_successor("alibi", seed=100, steps=1, batch_size=2, settings=settings)
    assert model.variant == "alibi"
    with torch.no_grad():
        assert torch.isfinite(model(torch.zeros(1, 16, dtype=torch.long))).all()


def test_successor_tuning_selects_from_every_complete_candidate():
    records = []
    for candidate_index, correct in ((0, 5), (1, 6), (2, 7)):
        for seed in range(100, 110):
            records.append(
                {
                    "method": "alibi",
                    "candidate": candidate_index,
                    "seed": seed,
                    "steps": 500,
                    "metrics": {
                        "4_digit_held_out": {
                            "4096": {
                                "1024_4095": {"correct": correct, "total": 10},
                            }
                        }
                    },
                }
            )
    assert select_candidates(records, ["alibi"], 500) == {"alibi": 2}
    assert select_candidates(records[:20], ["alibi"], 500) == {"alibi": 1}
    with pytest.raises(ValueError, match="missing original tuning candidates"):
        select_candidates(records[10:], ["alibi"], 500)
    with pytest.raises(ValueError, match="incomplete tuning seeds"):
        select_candidates(records[:-1], ["alibi"], 500)
