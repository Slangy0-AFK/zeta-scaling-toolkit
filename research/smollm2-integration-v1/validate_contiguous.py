"""Resource-bounded contiguous pretrained-model position validation."""

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import torch
import transformers
from position_extension import ADAPTER_PATH, ExperimentalPositionAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer
from validate_positions import REVISION, WEIGHT_SHA256, digest


def available_memory_mib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable is not reported by /proc/meminfo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8193)
    parser.add_argument("--max-seconds", type=float, default=2100)
    parser.add_argument("--min-available-mib", type=float, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 1 <= args.length <= 65537 or args.max_seconds <= 0 or args.min_available_mib < 1024:
        raise ValueError("invalid length or resource budget")
    if digest(args.snapshot / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("model weight checksum mismatch")
    if available_memory_mib() < args.min_available_mib + 2560:
        raise RuntimeError("insufficient available memory to load the model safely")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()
    model_adapter = ExperimentalPositionAdapter(
        model, max_positions=65537, window=64, sink_tokens=4
    )
    ids = tokenizer.encode("The quiet sky is blue.", add_special_tokens=False)
    if not ids:
        raise RuntimeError("empty token sequence")
    tokens = (ids * (args.length // len(ids) + 1))[: args.length]
    input_sha256 = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    started = time.perf_counter()
    cpu_start = time.process_time()
    progress = []
    offset = 0
    status = "completed"
    reason = None
    while offset < args.length:
        if time.perf_counter() - started >= args.max_seconds:
            status, reason = "incomplete", "wall-time budget reached"
            break
        if available_memory_mib() < args.min_available_mib:
            status, reason = "incomplete", "available-memory budget reached"
            break
        end = min(offset + 128, args.length)
        logits = model_adapter.forward(torch.tensor([tokens[offset:end]], dtype=torch.long))
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits")
        offset = end
        positions = model_adapter.cache.layers[0].positions.tolist()
        expected = [pos for pos in range(offset) if pos < 4 or pos >= offset - 64]
        if positions != expected or any(
            layer.next_position != offset or layer.positions.tolist() != expected
            for layer in model_adapter.cache.layers
        ):
            raise RuntimeError("unexpected retained or next positions")
        progress.append(
            {
                "processed_tokens": offset,
                "wall_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu_start,
                "cache_bytes": model_adapter.cache.storage_bytes(),
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "available_memory_mib": available_memory_mib(),
            }
        )
        if offset % 1024 == 0 or offset == args.length:
            print(f"processed {offset}/{args.length} tokens", flush=True)
    result = {
        "status": status,
        "reason": reason,
        "requested_tokens": args.length,
        "processed_tokens": offset,
        "input_sha256": input_sha256,
        "model_revision": REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "tokenizer_sha256": digest(args.snapshot / "tokenizer.json"),
        "config_sha256": digest(args.snapshot / "config.json"),
        "original_adapter_sha256": digest(ADAPTER_PATH),
        "prototype_sha256": digest(Path(__file__).with_name("position_extension.py")),
        "validation_sha256": digest(Path(__file__)),
        "manifest": model_adapter.manifest(),
        "budget": {
            "max_seconds": args.max_seconds,
            "min_available_mib": args.min_available_mib,
            "chunk_tokens": 128,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "torch.float32",
            "device": "cpu",
            "cpu_threads": 2,
        },
        "retained_absolute_positions": model_adapter.cache.layers[0].positions.tolist()
        if offset
        else [],
        "all_logits_finite": True,
        "progress": progress,
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(f"status: {status}; result: {args.output}", flush=True)


if __name__ == "__main__":
    main()
