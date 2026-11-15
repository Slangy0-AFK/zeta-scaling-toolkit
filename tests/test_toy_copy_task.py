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
    for (name, value), (other_name, other_value) in zip(
        standard.state_dict().items(), corrected.state_dict().items()
    ):
        assert name == other_name
        torch.testing.assert_close(value, other_value)
    for block in corrected.blocks:
        block.corrected_attention.A_attn = 0.0
    tokens = torch.randint(benchmark.VOCAB_SIZE, (2, benchmark.SEQ_LEN))
    torch.testing.assert_close(standard(tokens), corrected(tokens), atol=1e-6, rtol=1e-5)


def test_training_loops_produce_finite_metrics():
    generator = torch.Generator().manual_seed(benchmark.SEED)
    batches = torch.randint(benchmark.VOCAB_SIZE, (2, 2, benchmark.SEQ_LEN), generator=generator)
    inputs = torch.randn(2, 2, benchmark.D_MODEL, generator=generator)
    for corrected in (False, True):
        losses = benchmark.train_transformer(corrected, batches)
        assert len(losses) == 2 and all(math.isfinite(loss) for loss in losses)
        assert math.isfinite(benchmark.train_mlp(corrected, inputs, inputs[0]))
