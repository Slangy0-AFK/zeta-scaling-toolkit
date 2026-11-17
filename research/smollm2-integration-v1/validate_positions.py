"""Bounded pretrained-model position probes; no long-context quality evaluation."""

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import torch
import transformers
from position_extension import ADAPTER_PATH, ExperimentalPositionAdapter, adapter
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

WEIGHT_SHA256 = "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
ATOL, RTOL = 2e-4, 1e-4
ROPE_ATOL = 3e-3


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compare(name, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: mismatched shapes")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError(f"{name}: non-finite output")
    difference = (actual - expected).abs()
    failures = int((~torch.isclose(actual, expected, atol=ATOL, rtol=RTOL)).sum())
    if failures:
        raise AssertionError(f"{name}: {failures} values outside tolerance")
    return {"max_abs_difference": float(difference.max()), "out_of_tolerance": failures}


def native_window(model, tokens, start, window):
    positions = torch.arange(start, start + tokens.shape[1])
    q, k = positions[:, None], positions[None, :]
    visible = (k <= q) & (k >= q - window + 1)
    mask = torch.zeros(1, 1, len(positions), len(positions)).masked_fill(
        ~visible, torch.finfo(torch.float32).min
    )
    with torch.inference_mode():
        return model(
            tokens,
            attention_mask=mask,
            position_ids=positions[None, :],
            cache_position=positions,
            use_cache=False,
        ).logits


def original_parity(model, tokens):
    standard = adapter.ZetaLlamaAdapter(model)
    experimental = ExperimentalPositionAdapter(model, max_positions=65537)
    standard_logits = standard.forward(tokens)
    experimental_logits = experimental.forward(tokens)
    if not torch.equal(standard_logits, experimental_logits):
        raise AssertionError("original and opt-in adapters differed in the original range")
    with torch.inference_mode():
        native_logits = model(tokens, attention_mask=torch.ones_like(tokens)).logits
    return {
        "original_adapter_vs_experimental": compare(
            "adapter parity", experimental_logits, standard_logits
        ),
        "experimental_vs_native_prefill": compare(
            "native prefill", experimental_logits, native_logits
        ),
        "token_count": tokens.shape[1],
        "manifest": experimental.manifest(),
    }


def rotary_reference(model):
    positions = torch.tensor([[0, 8191, 8192, 16384, 32768, 65536]])
    query = torch.linspace(-1, 1, 64).view(1, 1, 1, 64).expand(1, 1, positions.shape[1], 64)
    cosine, sine = model.model.rotary_emb(query, positions)
    native, _ = apply_rotary_pos_emb(query, query, cosine, sine)
    frequencies = float(model.config.rope_theta) ** (
        -torch.arange(0, 64, 2, dtype=torch.float64) / 64
    )
    angles = positions.double().unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    first, second = query.double().chunk(2, dim=-1)
    reference = query.double() * angles.cos() + torch.cat((-second, first), dim=-1) * angles.sin()
    error = (native.double() - reference).abs()
    if not torch.isfinite(native).all() or float(error.max()) > ROPE_ATOL:
        raise AssertionError("native RoPE deviates from the independent FP64 reference")
    return {
        "reference": "FP64 split-half rotation and independently computed inverse frequencies",
        "max_absolute_tolerance": ROPE_ATOL,
        "positions": [int(position) for position in positions[0]],
        "max_abs_error_by_position": [
            float(error[0, 0, index].max()) for index in range(len(positions[0]))
        ],
        "shape": list(native.shape),
    }


def high_offsets(model, tokens):
    results = []
    for start in (8190, 16382, 32766, 65532):
        extended = ExperimentalPositionAdapter(
            model, max_positions=65537, window=3, initial_position=start
        )
        began = time.perf_counter()
        logits = torch.cat(
            (extended.forward(tokens[:, :2]), extended.forward(tokens[:, 2:])), dim=1
        )
        elapsed = time.perf_counter() - began
        expected = native_window(model, tokens, start, 3)
        comparison = compare(f"offset {start}", logits, expected)
        final = start + tokens.shape[1]
        retained = list(range(final - 3, final))
        for layer in extended.cache.layers:
            if layer.next_position != final or layer.positions.tolist() != retained:
                raise AssertionError("cache position reset or incorrect eviction")
            if not torch.isfinite(layer.keys).all() or not torch.isfinite(layer.values).all():
                raise AssertionError("non-finite key/value cache")
        altered = tokens.clone()
        altered[0, -1] = (int(altered[0, -1]) + 1) % model.config.vocab_size
        control = ExperimentalPositionAdapter(
            model, max_positions=65537, window=3, initial_position=start
        )
        altered_logits = control.forward(altered)
        if not torch.equal(logits[:, :-1], altered_logits[:, :-1]):
            raise AssertionError("future input changed an earlier logit")
        results.append(
            {
                "manifest": extended.manifest(),
                "processed_absolute_positions": list(range(start, final)),
                "retained_absolute_positions": retained,
                "next_absolute_position": final,
                "native_dense_mask": comparison,
                "future_token_invariance": True,
                "finite": bool(torch.isfinite(logits).all()),
                "finite_cache": True,
                "cache_bytes": extended.cache.storage_bytes(),
                "wall_seconds": elapsed,
            }
        )
    return results


def actual_lengths(model, token_ids):
    model_adapter = ExperimentalPositionAdapter(
        model, max_positions=65537, window=64, sink_tokens=4
    )
    lengths = (16, 32, 64, 128)
    tokens = torch.tensor([(token_ids * (128 // len(token_ids) + 1))[:128]])
    results = []
    offset = 0
    start = time.perf_counter()
    cpu_start = time.process_time()
    for length in lengths:
        begin = time.perf_counter()
        cpu = time.process_time()
        logits = model_adapter.forward(tokens[:, offset:length])
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite long-input logits")
        results.append(
            {
                "actual_contiguous_tokens": length,
                "chunk_tokens": length - offset,
                "chunk_wall_seconds": time.perf_counter() - begin,
                "chunk_cpu_seconds": time.process_time() - cpu,
                "cumulative_wall_seconds": time.perf_counter() - start,
                "cumulative_cpu_seconds": time.process_time() - cpu_start,
                "cache_bytes": model_adapter.cache.storage_bytes(),
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "finite_logits": True,
            }
        )
        offset = length
    return {"manifest": model_adapter.manifest(), "measurements": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if digest(args.snapshot / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("model weight checksum mismatch")
    torch.set_num_threads(2)
    total_start = time.perf_counter()
    cpu_start = time.process_time()
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()
    words = tokenizer.encode("Hello world. It is another day.", add_special_tokens=False)
    if len(words) < 5:
        raise RuntimeError("unexpected native tokenizer output")
    tokens = torch.tensor([words[:5]])
    parity = original_parity(model, tokens)
    rope = rotary_reference(model)
    offsets = high_offsets(model, tokens)
    lengths = actual_lengths(model, words)
    output = {
        "status": "pass_for_bounded_probes_only",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "config_sha256": digest(args.snapshot / "config.json"),
        "tokenizer_sha256": digest(args.snapshot / "tokenizer.json"),
        "original_adapter_sha256": digest(ADAPTER_PATH),
        "prototype_sha256": digest(Path(__file__).with_name("position_extension.py")),
        "validation_sha256": digest(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "torch.float32",
            "device": "cpu",
            "cpu_threads": torch.get_num_threads(),
        },
        "logit_tolerance": {"atol": ATOL, "rtol": RTOL},
        "original_range_parity": parity,
        "rotary_reference": rope,
        "synthetic_empty_history_offset_probes": offsets,
        "actual_contiguous_input": lengths,
        "total_wall_seconds": time.perf_counter() - total_start,
        "total_cpu_seconds": time.process_time() - cpu_start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
