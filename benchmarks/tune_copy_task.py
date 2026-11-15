"""Grid search for the toy copy task; results do not predict real-model performance."""

import argparse
import itertools
import json
import math
import statistics

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from toy_copy_task import BATCH_SIZE, LR, SEQ_LEN, VOCAB_SIZE, CopyTransformer

from zeta_scaling_toolkit import AsymptoticLRScheduler


def train_trial(
    seed, steps, batch_size, lr, corrected_attention, A_attn, floor_ratio, constant_lr=False
):
    """Train and evaluate once; replace this function to tune another training loop."""
    generator = torch.Generator().manual_seed(seed)
    batches = torch.randint(VOCAB_SIZE, (steps, batch_size, SEQ_LEN), generator=generator)
    evaluation = torch.randint(VOCAB_SIZE, (batch_size, SEQ_LEN), generator=generator)
    torch.manual_seed(seed)
    model = CopyTransformer(corrected_attention)
    if corrected_attention:
        for block in model.blocks:
            block.corrected_attention.A_attn = A_attn
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if constant_lr:
        scheduler = None
    elif floor_ratio is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=steps)
    else:
        scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=floor_ratio)
    model.train()
    for tokens in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tokens.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss for seed {seed}")
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    model.eval()
    with torch.no_grad():
        logits = model(evaluation)
        evaluation_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), evaluation.reshape(-1))
    if not torch.isfinite(evaluation_loss):
        raise RuntimeError(f"Non-finite evaluation loss for seed {seed}")
    return loss.item(), evaluation_loss.item()


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def floor_float(value):
    result = nonnegative_float(value)
    if result > 1:
        raise argparse.ArgumentTypeError("floor ratio must be between 0 and 1")
    return result


def positive_float(value):
    result = nonnegative_float(value)
    if result == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=300)
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--learning-rates", type=positive_float, nargs="+", default=[LR])
    parser.add_argument("--constant-lr-ratio", type=positive_float, default=0.75)
    parser.add_argument(
        "--attention-values", type=nonnegative_float, nargs="+", default=[0, 0.5, 1]
    )
    parser.add_argument(
        "--floor-ratios", type=floor_float, nargs="+", default=[0.01, 0.25, 0.5, 0.75]
    )
    args = parser.parse_args()
    if any(seed < 0 or seed >= 2**63 for seed in args.seeds):
        parser.error("seeds must be in the range [0, 2**63)")
    if any(
        not math.isfinite(lr * args.constant_lr_ratio) or lr * args.constant_lr_ratio == 0
        for lr in args.learning_rates
    ):
        parser.error("constant learning rates must be finite and greater than zero")
    torch.set_num_threads(1)

    for lr in args.learning_rates:
        configurations = [
            ("A", False, None, None),
            *(("B", False, None, floor) for floor in args.floor_ratios),
            *(("C", True, attention, None) for attention in args.attention_values),
            *(
                ("D", True, attention, floor)
                for attention, floor in itertools.product(args.attention_values, args.floor_ratios)
            ),
            ("E", False, None, None),
        ]
        for config, corrected, attention, floor in configurations:
            constant_lr = config == "E"
            trial_lr = lr * args.constant_lr_ratio if constant_lr else lr
            results = [
                train_trial(
                    seed,
                    args.steps,
                    args.batch_size,
                    trial_lr,
                    corrected,
                    attention,
                    floor,
                    constant_lr=constant_lr,
                )
                for seed in args.seeds
            ]
            record = {
                "config": config,
                "learning_rate": trial_lr,
                "comparison_base_lr": lr,
                "scheduler": (
                    "constant"
                    if constant_lr
                    else "cosine"
                    if floor is None
                    else "cosine_with_floor"
                ),
                "A_attn": attention,
                "floor_ratio": floor,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "seeds": args.seeds,
                "final_batch_losses": [result[0] for result in results],
                "evaluation_losses": [result[1] for result in results],
                "mean_final_batch_loss": statistics.mean(result[0] for result in results),
                "mean_evaluation_loss": statistics.mean(result[1] for result in results),
            }
            print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
