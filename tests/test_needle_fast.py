import copy
import json
import sys

import pytest
import torch
import torch.nn.functional as F

from benchmarks import context_ability
from benchmarks.context_ability import VOCABULARY_SIZE, planted_needle_batch, train_needle
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


def test_needle_training_supports_both_output_paths():
    full = train_needle(seed=42, steps=1, length=8, batch_size=2, full_output=True)
    fast = train_needle(seed=42, steps=1, length=8, batch_size=2)
    for expected, actual in zip(full.parameters(), fast.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert not full.training and not fast.training


def test_needle_training_rejects_invalid_batches_and_nonfinite_loss(monkeypatch):
    with pytest.raises(ValueError, match="batch_size"):
        planted_needle_batch(0, 8, torch.Generator())
    with pytest.raises(ValueError, match="length"):
        planted_needle_batch(2, 3, torch.Generator())
    monkeypatch.setattr(
        context_ability.F, "cross_entropy", lambda *args: torch.tensor(float("nan"))
    )
    with pytest.raises(RuntimeError, match="non-finite needle loss"):
        train_needle(seed=42, steps=1, length=8, batch_size=2)


def test_needle_sanity_resume_records_output_path(tmp_path, monkeypatch):
    output = tmp_path / "sanity.jsonl"
    calls = []
    monkeypatch.setattr(
        context_ability,
        "train_needle",
        lambda *args: calls.append(args) or object(),
    )
    monkeypatch.setattr(context_ability, "evaluate_needle_sanity", lambda *args: (180, 200))
    args = [
        "context_ability",
        "--steps",
        "1",
        "--train-length",
        "8",
        "--seeds",
        "42",
        "43",
        "--full-output",
        "--max-runs",
        "1",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", args)
    context_ability.main()
    assert len(output.read_text().splitlines()) == len(calls) == 1
    context_ability.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == len(calls) == 2
    assert [record["seed"] for record in records] == [42, 43]
    assert all(record["training_implementation"] == "full-output" for record in records)
    assert all(record["accuracy"] == 0.9 for record in records)

    args[2] = "2"
    with pytest.raises(ValueError, match="different experiment configuration"):
        context_ability.main()

    args[2] = "1"
    args[4] = "3"
    with pytest.raises(SystemExit):
        context_ability.main()
