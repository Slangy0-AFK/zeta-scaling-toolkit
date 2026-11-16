"""Tune and evaluate planted-needle baselines after the learning sanity gate."""

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zeta_scaling_toolkit import AsymptoticLRScheduler

if __package__:
    from .context_ability import VOCABULARY_SIZE, evaluate_needle_sanity, planted_needle_batch
    from .context_reuse import reuse_rope_trial
    from .context_successor import METHODS, candidate
    from .length_generalization import SequenceLM, positive_int
else:
    from context_ability import VOCABULARY_SIZE, evaluate_needle_sanity, planted_needle_batch
    from context_reuse import reuse_rope_trial
    from context_successor import METHODS, candidate
    from length_generalization import SequenceLM, positive_int


def sanity_passed(records: list[dict[str, Any]]) -> bool:
    if len(records) != 10 or {record["seed"] for record in records} != set(range(42, 52)):
        raise ValueError("the retrieval sanity gate requires exactly seeds 42 through 51")
    if any(
        record["variant"] != "rope-full"
        or record["train_length"] != 256
        or record["steps"] != 3000
        or record["total"] < 200
        for record in records
    ):
        raise ValueError("retrieval sanity records do not match the specified gate")
    return sum(record["correct"] / record["total"] >= 0.9 for record in records) >= 8


def select_candidates(
    records: list[dict[str, Any]], methods: list[str], steps: int
) -> dict[str, int]:
    selected = {}
    for method in methods:
        scores = []
        for index in (0, 1):
            trials = [
                record
                for record in records
                if record["method"] == method
                and record["candidate"] == index
                and record["steps"] == steps
            ]
            if len(trials) != 10 or {record["seed"] for record in trials} != set(range(100, 110)):
                raise ValueError(
                    f"incomplete retrieval tuning seeds for {method} candidate {index}"
                )
            length = "256" if method == "abs" else "1024"
            scores.append(
                sum(record["metrics"][length]["accuracy"] for record in trials) / len(trials)
            )
        selected[method] = max((0, 1), key=lambda index: scores[index])
    return selected


def train_needle(
    method: str, seed: int, steps: int, batch_size: int, settings: dict[str, float]
) -> SequenceLM:
    torch.manual_seed(seed)
    training_variant = "rope-full" if method in ("rope-linear", "rope-ntk") else method
    model = SequenceLM(VOCABULARY_SIZE, training_variant, False, 256, settings["tau0"], 10000.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=0.1, total_steps=steps)
    generator = torch.Generator().manual_seed(seed + 50000)
    for _ in range(steps):
        inputs, targets = planted_needle_batch(batch_size, 256, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.forward_last(inputs), targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite needle loss for {method}/{seed}")
        loss.backward()
        optimizer.step()
        scheduler.step()
    model.eval()
    if method in ("rope-linear", "rope-ntk"):
        model.variant = method
        for block in model.blocks:
            block.variant = method
            block.rope_scale = settings["rope_scale"]
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=3000)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--eval-queries", type=positive_int, default=200)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 110)))
    parser.add_argument("--candidates", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--tuning-records", type=Path)
    parser.add_argument(
        "--sanity-records",
        type=Path,
        default=Path("benchmarks/context-ability-needle-sanity.jsonl"),
    )
    parser.add_argument("--max-runs", type=positive_int)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/context-ability-needle-tuning3000.jsonl")
    )
    args = parser.parse_args()
    if (
        len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
        or len(set(args.methods)) != len(args.methods)
        or len(set(args.candidates)) != len(args.candidates)
        or args.eval_queries < 200
    ):
        parser.error(
            "seeds, methods, and candidates must be distinct; at least 200 queries are required"
        )
    if args.tuning == (args.tuning_records is not None):
        parser.error("choose either tuning or separate tuning records for reporting")
    sanity = [json.loads(line) for line in args.sanity_records.read_text().splitlines() if line]
    if not sanity_passed(sanity):
        raise ValueError(
            "retrieval sanity gate failed; do not run long-context retrieval comparisons"
        )
    selected = None
    if args.tuning_records is not None:
        if set(args.seeds) & set(range(100, 110)):
            parser.error("reporting seeds must be disjoint from tuning seeds")
        tuning_records = [
            json.loads(line) for line in args.tuning_records.read_text().splitlines() if line
        ]
        if any(
            not record["tuning"]
            or record["steps"] != args.steps
            or record["batch_size"] != args.batch_size
            or record["eval_queries"] != args.eval_queries
            or record["settings"] != candidate(record["method"], record["candidate"])
            for record in tuning_records
        ):
            raise ValueError("retrieval tuning records do not match the reporting configuration")
        selected = select_candidates(tuning_records, args.methods, args.steps)
    torch.set_num_threads(1)
    configuration = {
        "task": "planted-needle",
        "tuning": args.tuning,
        "steps": args.steps,
        "train_length": 256,
        "batch_size": args.batch_size,
        "eval_queries": args.eval_queries,
    }
    records: list[dict[str, Any]] = []
    if args.output.exists():
        records = [json.loads(line) for line in args.output.read_text().splitlines() if line]
    if any(any(record[key] != value for key, value in configuration.items()) for record in records):
        raise ValueError("output contains a different experiment configuration")
    completed = {(record["method"], record["candidate"], record["seed"]) for record in records}
    cache: dict[int, SequenceLM] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_runs = 0
    with args.output.open("a", encoding="utf-8") as output:
        for method in args.methods:
            for index in [selected[method]] if selected is not None else args.candidates:
                for seed in args.seeds:
                    if (method, index, seed) in completed:
                        continue
                    settings = candidate(method, index)
                    model = reuse_rope_trial(
                        cache,
                        method,
                        seed,
                        settings,
                        lambda: train_needle(method, seed, args.steps, args.batch_size, settings),
                    )
                    lengths = (
                        [256 if method == "abs" else 1024]
                        if args.tuning
                        else ([256] if method == "abs" else [256, 512, 1024, 4096])
                    )
                    metrics = {}
                    for length in lengths:
                        correct, total = evaluate_needle_sanity(
                            model,
                            seed,
                            length,
                            args.eval_queries,
                            min(args.batch_size, 2) if length >= 4096 else args.batch_size,
                        )
                        metrics[str(length)] = {
                            "correct": correct,
                            "total": total,
                            "accuracy": correct / total,
                        }
                    record = {
                        **configuration,
                        "method": method,
                        "candidate": index,
                        "settings": settings,
                        "seed": seed,
                        "training_implementation": (
                            "last-query"
                            if method in ("rope-full", "rope-linear", "rope-ntk")
                            else "full-output"
                        ),
                        "metrics": metrics,
                        "environment": {
                            "python": platform.python_version(),
                            "torch": torch.__version__,
                            "platform": platform.platform(),
                            "torch_threads": torch.get_num_threads(),
                        },
                    }
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                    completed.add((method, index, seed))
                    print(
                        json.dumps({"method": method, "candidate": index, "seed": seed}), flush=True
                    )
                    new_runs += 1
                    if args.max_runs is not None and new_runs >= args.max_runs:
                        return


if __name__ == "__main__":
    main()
