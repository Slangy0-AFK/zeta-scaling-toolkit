"""Run bounded, offline native-model parity and cache validation."""

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import torch
import transformers
from adapter import ZetaLlamaAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
WEIGHT_SHA256 = "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e"
ATOL = 2e-4
RTOL = 1e-4


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compare(name, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: mismatched logit shapes")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError(f"{name}: non-finite logits")
    delta = (actual - expected).abs()
    failures = int((~torch.isclose(actual, expected, atol=ATOL, rtol=RTOL)).sum())
    result = {
        "name": name,
        "shape": list(actual.shape),
        "max_absolute_difference": float(delta.max()),
        "mean_absolute_difference": float(delta.mean()),
        "out_of_tolerance": failures,
    }
    if failures:
        raise AssertionError(f"{name}: {failures} values outside tolerance; {result}")
    return result


def timer(call):
    start = time.perf_counter()
    cpu = time.process_time()
    value = call()
    return value, {
        "wall_seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time() - cpu,
    }


def native_call(model, tokens, cache=None, start=0):
    batch, length = tokens.shape
    with torch.inference_mode():
        return model(
            tokens,
            attention_mask=torch.ones(batch, start + length, dtype=torch.long),
            position_ids=torch.arange(start, start + length).expand(batch, -1),
            cache_position=torch.arange(start, start + length),
            past_key_values=cache,
            use_cache=True,
        )


def masked_native(model, tokens, window, sinks):
    length = tokens.shape[1]
    query = torch.arange(length)[:, None]
    key = torch.arange(length)[None, :]
    visible = (key <= query) & ((key < sinks) | (key >= query - window + 1))
    mask = torch.zeros(1, 1, length, length).masked_fill(~visible, torch.finfo(torch.float32).min)
    with torch.inference_mode():
        return model(
            tokens,
            attention_mask=mask.expand(tokens.shape[0], -1, -1, -1),
            position_ids=torch.arange(length).expand(tokens.shape[0], -1),
            use_cache=False,
        ).logits


def metadata(snapshot, model, load, tokenizer):
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weight_sha256": digest(snapshot / "model.safetensors"),
        "tokenizer_sha256": digest(snapshot / "tokenizer.json"),
        "tokenizer_config_sha256": digest(snapshot / "tokenizer_config.json"),
        "config_sha256": digest(snapshot / "config.json"),
        "adapter_sha256": digest(Path(__file__).with_name("adapter.py")),
        "validation_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "cpu_threads": torch.get_num_threads(),
        "eos_token_id": tokenizer.eos_token_id,
        "model_load": load,
        "tolerance": {"absolute": ATOL, "relative": RTOL},
    }


def parity(model, tokenizer):
    hello = tokenizer.encode("Hello", add_special_tokens=False)
    world = tokenizer.encode("World", add_special_tokens=False)
    suffix = tokenizer.encode(" science matters today.", add_special_tokens=False)
    if len(hello) != 1 or len(world) != 1 or len(suffix) < 3:
        raise RuntimeError("unexpected native tokenizer output")
    tokens = torch.tensor([hello + suffix, world + suffix], dtype=torch.long)
    cases = []
    for batch in (1, 2):
        selected = tokens[:batch]
        native, native_prefill = timer(lambda: native_call(model, selected))
        sequential_cache = DynamicCache()
        sequential = torch.cat(
            [
                native_call(model, selected[:, index : index + 1], sequential_cache, index).logits
                for index in range(selected.shape[1])
            ],
            dim=1,
        )
        adapter = ZetaLlamaAdapter(model)
        actual, adapter_prefill = timer(
            lambda: adapter.forward(
                selected,
                attention_mask=torch.ones_like(selected),
                position_ids=torch.arange(selected.shape[1]).expand(batch, -1),
            )
        )
        cases.append(compare(f"prefill_batch_{batch}", actual, native.logits))
        cases[-1].update(
            {
                "native_timing": native_prefill,
                "adapter_timing": adapter_prefill,
                "cache_bytes": adapter.cache.storage_bytes(),
            }
        )
        cases.append(
            compare(f"native_sequential_vs_prefill_batch_{batch}", sequential, native.logits)
        )
        cases.append(compare(f"adapter_vs_native_sequential_batch_{batch}", actual, sequential))
        if not torch.equal(actual, sequential):
            raise AssertionError("adapter must exactly match native token-by-token inference")

    cache = DynamicCache()
    adapter = ZetaLlamaAdapter(model)
    pieces = (tokens[:, :1], tokens[:, 1:3], tokens[:, 3:4], tokens[:, 4:])
    start = 0
    for index, piece in enumerate(pieces):
        native, native_time = timer(lambda: native_call(model, piece, cache, start))
        actual, adapter_time = timer(lambda: adapter.forward(piece))
        label = (
            "single_token_batch_2",
            "continuation_1_two_tokens",
            "continuation_2_one_token",
            "continuation_3_one_token",
        )[index]
        cases.append(compare(label, actual, native.logits))
        cases[-1].update({"native_timing": native_time, "adapter_timing": adapter_time})
        start += piece.shape[1]

    eos_inputs = torch.tensor([[hello[0]], [world[0]]])
    native_logits = native_call(model, eos_inputs).logits[:, -1]
    eos = int(native_logits[0].argmax())
    if int(native_logits[1].argmax()) == eos:
        raise RuntimeError("EOS test inputs did not produce different first tokens")
    adapter = ZetaLlamaAdapter(model)
    generated = adapter.generate(eos_inputs, max_new_tokens=3, eos_token_id=eos)
    with torch.inference_mode():
        expected = model.generate(
            eos_inputs,
            attention_mask=torch.ones_like(eos_inputs),
            do_sample=False,
            max_new_tokens=3,
            eos_token_id=eos,
            pad_token_id=eos,
        )[:, eos_inputs.shape[1] :].tolist()
    expected = [row[: row.index(eos) + 1] if eos in row else row for row in expected]
    if generated != expected or generated[0] != [eos]:
        raise AssertionError(f"per-row EOS mismatch: adapter={generated}, native={expected}")
    cases.append(
        {
            "name": "greedy_per_row_eos",
            "eos_token_id": eos,
            "adapter_tokens": generated,
            "native_tokens": expected,
        }
    )
    eos_tokens = torch.tensor([[hello[0], tokenizer.eos_token_id]])
    actual = ZetaLlamaAdapter(model).forward(eos_tokens)
    cases.append(
        compare("native_eos_token_as_input", actual, native_call(model, eos_tokens).logits)
    )
    return cases


def bounded(model, tokenizer):
    ids = tokenizer.encode(
        "Hello science matters today. More facts arrive.", add_special_tokens=False
    )
    if len(ids) < 7:
        raise RuntimeError("test needs at least seven input tokens")
    tokens = torch.tensor([ids, list(reversed(ids))], dtype=torch.long)
    window, sinks = 3, 2
    adapter = ZetaLlamaAdapter(model, window=window, sink_tokens=sinks)
    actual = torch.cat(
        [
            adapter.forward(tokens[:, :2]),
            adapter.forward(tokens[:, 2:5]),
            adapter.forward(tokens[:, 5:]),
        ],
        dim=1,
    )
    expected = masked_native(model, tokens, window, sinks)
    result = compare("bounded_native_dense_mask", actual, expected)
    length = tokens.shape[1]
    positions = [
        position for position in range(length) if position < sinks or position >= length - window
    ]
    shapes = []
    for layer in adapter.cache.layers:
        if layer.positions.tolist() != positions or layer.next_position != length:
            raise AssertionError("wrong absolute positions after eviction")
        if layer.keys.shape != (2, 5, len(positions), 64):
            raise AssertionError(f"unexpected KV shape: {layer.keys.shape}")
        if not torch.isfinite(layer.keys).all() or not torch.isfinite(layer.values).all():
            raise AssertionError("non-finite bounded cache")
        shapes.append(list(layer.keys.shape))
    result.update(
        {
            "window": window,
            "sink_tokens": sinks,
            "input_tokens_per_row": length,
            "retained_absolute_positions": positions,
            "layer_count": len(shapes),
            "all_layer_key_shapes": shapes,
            "cache_bytes": adapter.cache.storage_bytes(),
            "finite_cache": True,
            "finite_logits": True,
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--phase", choices=("parity", "bounded"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    weight_hash = digest(args.snapshot / "model.safetensors")
    if weight_hash != WEIGHT_SHA256:
        raise ValueError(f"unexpected model weights: {weight_hash}")
    if args.phase == "bounded":
        prior = json.loads((args.output_dir / "NATIVE_PARITY_RESULTS.json").read_text())
        if (
            prior["status"] != "pass"
            or prior["metadata"]["weight_sha256"] != weight_hash
            or prior["metadata"]["adapter_sha256"] != digest(Path(__file__).with_name("adapter.py"))
            or prior["metadata"]["validation_sha256"] != digest(Path(__file__))
        ):
            raise RuntimeError("native parity must pass for these weights and code before eviction")
    output = args.output_dir / (
        "NATIVE_PARITY_RESULTS.json" if args.phase == "parity" else "CACHE_VALIDATION_RESULTS.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    start = time.perf_counter()
    cpu = time.process_time()
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    model, load = timer(
        lambda: AutoModelForCausalLM.from_pretrained(
            args.snapshot,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        ).eval()
    )
    info = metadata(args.snapshot, model, load, tokenizer)
    if args.phase == "parity":
        results = parity(model, tokenizer)
    else:
        results = bounded(model, tokenizer)
    wall = time.perf_counter() - start
    cpu_time = time.process_time() - cpu
    record = {
        "status": "pass",
        "phase": args.phase,
        "metadata": info,
        "results": results,
        "resources": {
            "wall_seconds_including_load": wall,
            "process_cpu_seconds": cpu_time,
            "mean_cores_used": cpu_time / wall,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "logical_cpus_available": len(os.sched_getaffinity(0)),
        },
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
        file.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
