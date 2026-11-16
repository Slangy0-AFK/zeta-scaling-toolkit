"""Hardware preflight and optional non-research CUDA performance smoke test."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import torch

if __package__:
    from .context_retrieval import (
        INITIAL_METHODS,
        TRAINING_LENGTH,
        TRAINING_PAIR_COUNT,
        VOCABULARY_SIZE,
        _build_training_pool,
        _save_model,
        candidate_settings,
        evaluate_model,
        generate_dataset,
        positive_int,
        train_model,
    )
    from .length_generalization import SequenceLM
else:
    from context_retrieval import (
        INITIAL_METHODS,
        TRAINING_LENGTH,
        TRAINING_PAIR_COUNT,
        VOCABULARY_SIZE,
        _build_training_pool,
        _save_model,
        candidate_settings,
        evaluate_model,
        generate_dataset,
        positive_int,
        train_model,
    )
    from length_generalization import SequenceLM


def hardware_report() -> dict:
    model_bytes = {}
    for method in INITIAL_METHODS:
        settings = candidate_settings(method, 0)
        model = SequenceLM(
            VOCABULARY_SIZE, method, False, TRAINING_LENGTH, settings["tau0"], 10000.0
        )
        model_bytes[method] = sum(p.numel() * p.element_size() for p in model.parameters())
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "compute_capability": [props.major, props.minor],
                    "free_bytes": free,
                    "total_bytes": total,
                }
            )
    return {
        "cpu_cores_logical": os.cpu_count(),
        "cpu_cores_affinity": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_version": torch.version.cuda,
        "gpus": gpus,
        "parameter_bytes_per_model": model_bytes,
        "one_fit_parameter_gradient_adam_lower_bound_bytes": 4 * max(model_bytes.values()),
        "memory_estimate_note": (
            "Lower bound only: parameters, gradients and two Adam moments; excludes "
            "activations, workspaces, allocator reserve and up to seven retained models. "
            "Measure peak on the target GPU for each method."
        ),
        "single_fit_estimate": "unavailable without CUDA smoke measurement",
    }


def run_cuda_smoke(prefix: Path, steps: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested, but CUDA is unavailable")
    result_path = Path(str(prefix) + ".json")
    checkpoint_path = Path(str(prefix) + ".pt")
    if result_path.exists() or checkpoint_path.exists():
        raise FileExistsError("refusing to overwrite an existing smoke artifact")
    torch.set_num_threads(1)
    training_inputs, training_targets, _ = _build_training_pool()
    device = torch.device("cuda")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model = train_model(
        "rope-log-full", 0, 200, steps, 16, training_inputs, training_targets, device
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("model parameters are not on CUDA")
    examples = generate_dataset("hardware-smoke", 7, 3, TRAINING_LENGTH, TRAINING_PAIR_COUNT)
    metrics = evaluate_model(model, examples, batch_size=3)
    if metrics["overall"]["total"] != 3:
        raise RuntimeError("smoke evaluation denominator mismatch")
    checkpoint_hash = _save_model(model, checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not all(
        torch.equal(state[name], value.detach().cpu()) for name, value in model.state_dict().items()
    ):
        raise RuntimeError("checkpoint round trip mismatch")
    settings = candidate_settings("rope-log-full", 0)
    restored = SequenceLM(
        VOCABULARY_SIZE, "rope-log-full", False, TRAINING_LENGTH, settings["tau0"], 10000.0
    ).to(device)
    restored.load_state_dict(state)
    if not all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in model.state_dict().items()
    ):
        raise RuntimeError("checkpoint CUDA restoration mismatch")
    rate = steps / elapsed
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "smoke_only": True,
        "research_result": False,
        "method": "rope-log-full",
        "candidate": 0,
        "seed": 200,
        "steps": steps,
        "batch_size": 16,
        "context_length": TRAINING_LENGTH,
        "device": str(next(model.parameters()).device),
        "training_seconds_including_model_setup": elapsed,
        "steps_per_second": rate,
        "estimated_3000_step_fit_seconds": 3000 / rate,
        "estimated_80_fit_training_seconds": 80 * 3000 / rate,
        "estimate_note": "Excludes dataset generation, evaluation, checkpoint I/O and method-specific throughput.",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "process_peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
        "evaluation_denominator": metrics["overall"]["total"],
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_cpu_and_cuda_round_trip": True,
        "hardware": hardware_report(),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-prefix", type=Path, help="Run a separate CUDA-only smoke fit")
    parser.add_argument("--smoke-steps", type=positive_int, default=20)
    args = parser.parse_args()
    if args.smoke_prefix:
        result = run_cuda_smoke(args.smoke_prefix, args.smoke_steps)
    else:
        result = hardware_report()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
