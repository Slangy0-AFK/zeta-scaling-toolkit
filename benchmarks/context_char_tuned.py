"""Tune and compare positional baselines on the character context-benefit task."""

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .context_char import context_benefit, evaluate_char_contexts, train_char
    from .context_reuse import reuse_rope_trial
    from .context_successor import METHODS, candidate
    from .length_generalization import SequenceLM, dataset_info, positive_int
else:
    from context_char import context_benefit, evaluate_char_contexts, train_char
    from context_reuse import reuse_rope_trial
    from context_successor import METHODS, candidate
    from length_generalization import SequenceLM, dataset_info, positive_int


def select_candidates(
    records: list[dict[str, Any]], methods: list[str], steps: int
) -> dict[str, int]:
    selected = {}
    for method in methods:
        losses = []
        for index in (0, 1):
            trials = [
                record
                for record in records
                if record["method"] == method
                and record["candidate"] == index
                and record["steps"] == steps
            ]
            if {record["seed"] for record in trials} != set(range(100, 110)) or len(trials) != 10:
                raise ValueError(
                    f"incomplete character tuning seeds for {method} candidate {index}"
                )
            losses.append(sum(record["tuning_nll"] for record in trials) / len(trials))
        selected[method] = min((0, 1), key=lambda index: losses[index])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=500)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--eval-batch-size", type=positive_int, default=2)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 110)))
    parser.add_argument("--candidates", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--tuning-records", type=Path)
    parser.add_argument("--max-runs", type=positive_int)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path.home() / ".cache" / "zeta-scaling-toolkit" / "tinyshakespeare.txt",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/context-ability-char-tuning500.jsonl")
    )
    args = parser.parse_args()
    if (
        len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
        or len(set(args.methods)) != len(args.methods)
        or len(set(args.candidates)) != len(args.candidates)
    ):
        parser.error("seeds, methods, and candidates must be distinct; seeds must be nonnegative")
    if args.tuning and args.tuning_records is not None:
        parser.error("tuning and reporting are separate runs")
    if not args.tuning and args.tuning_records is None:
        parser.error("reporting requires separate tuning records")
    torch.set_num_threads(1)
    text, digest = dataset_info(args.data_path)
    vocabulary = sorted(set(text))
    ids = {character: index for index, character in enumerate(vocabulary)}
    data = torch.tensor([ids[character] for character in text], dtype=torch.long)
    split = int(len(data) * 0.9)
    training, evaluation = data[:split], data[split:]
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
            or record["eval_batch_size"] != args.eval_batch_size
            or record["dataset_sha256"] != digest
            or record["settings"] != candidate(record["method"], record["candidate"])
            for record in tuning_records
        ):
            raise ValueError("character tuning records do not match the reporting configuration")
        selected = select_candidates(tuning_records, args.methods, args.steps)
    configuration = {
        "task": "char-context",
        "tuning": args.tuning,
        "steps": args.steps,
        "train_length": 256,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "dataset_sha256": digest,
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
                    training_variant = (
                        "rope-full" if method in ("rope-linear", "rope-ntk") else method
                    )
                    model = reuse_rope_trial(
                        cache,
                        method,
                        seed,
                        settings,
                        lambda: train_char(
                            training,
                            len(vocabulary),
                            training_variant,
                            seed,
                            args.steps,
                            256,
                            args.batch_size,
                            64,
                            4,
                            settings["learning_rate"],
                            settings["tau0"],
                        ),
                    )
                    if args.tuning and method != "abs":
                        benefit = context_benefit(model, evaluation, seed)
                        metrics = {"context_nll": benefit}
                        tuning_nll = benefit["1024"]
                    else:
                        metrics = evaluate_char_contexts(
                            model, evaluation, seed, [256, 512, 1024, 4096], args.eval_batch_size
                        )
                        tuning_nll = (
                            math.log(metrics["perplexity"]["256"])
                            if method == "abs"
                            else metrics["context_nll"]["1024"]
                        )
                    record = {
                        **configuration,
                        "method": method,
                        "candidate": index,
                        "settings": settings,
                        "seed": seed,
                        "tuning_nll": tuning_nll,
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
