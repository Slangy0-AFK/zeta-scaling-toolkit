"""Checks for the toy benchmark's comparison setup."""

import importlib.util
import math
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "toy_copy_task", Path(__file__).resolve().parents[1] / "benchmarks" / "toy_copy_task.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_corrected_attention_uses_same_initial_parameters_and_projections():
    torch.manual_seed(benchmark.SEED)
    standard = benchmark.CopyTransformer(False)
    torch.manual_seed(benchmark.SEED)
    corrected = benchmark.CopyTransformer(True)
    standard_state = standard.state_dict()
    corrected_state = corrected.state_dict()
    for name, value in standard_state.items():
        torch.testing.assert_close(value, corrected_state[name])
    assert set(corrected_state) - set(standard_state) == {
        f"blocks.{index}.corrected_attention.{scale}"
        for index in range(len(corrected.blocks))
        for scale in ("raw_tau",)
    }
    tokens = torch.randint(benchmark.VOCAB_SIZE, (2, benchmark.SEQ_LEN))
    assert torch.isfinite(standard(tokens)).all()
    assert torch.isfinite(corrected(tokens)).all()


def test_training_loops_produce_finite_metrics():
    generator = torch.Generator().manual_seed(benchmark.SEED)
    batches = torch.randint(benchmark.VOCAB_SIZE, (2, 2, benchmark.SEQ_LEN), generator=generator)
    inputs = torch.randn(2, 2, benchmark.D_MODEL, generator=generator)
    for corrected in (False, True):
        losses = benchmark.train_transformer(corrected, batches)
        assert len(losses) == 2 and all(math.isfinite(loss) for loss in losses)
        assert math.isfinite(benchmark.train_mlp(corrected, inputs, inputs[0]))
