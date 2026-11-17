"""Single-stream, resource-bounded SmolLM2 causal and retention investigation."""

import argparse
import hashlib
import json
import platform
import random
import re
import resource
import time
from pathlib import Path

import torch
import transformers
from position_extension import ADAPTER_PATH, ExperimentalPositionAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from validate_contiguous import available_memory_mib
from validate_positions import ATOL, REVISION, RTOL, WEIGHT_SHA256, compare, digest, original_parity

from zeta_scaling_toolkit import StreamingKVCache

TARGETS = (8193, 8256, 8512, 9000, 10000)
WINDOW, SINKS, SEED = 64, 64, 20261117


def chat_parts(tokenizer):
    marker = "ZETA_CONTENT_MARKER"
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], tokenize=False, add_generation_prompt=True
    )
    if template.count(marker) != 1:
        raise ValueError("unexpected tokenizer chat template")
    return template.split(marker)


def build_stream(tokenizer, length):
    """Make one fixed-position stream; no generated answers occur in queries."""
    if length < 256:
        raise ValueError("stream must contain at least 256 tokens")
    rng = random.Random(SEED)
    codes = rng.sample(range(100000, 1000000), 4)
    facts = []
    prefix, _ = chat_parts(tokenizer)
    tokens = tokenizer.encode(prefix, add_special_tokens=False)
    early, middle = (80, 128) if length == 256 else (128, 4096)
    for label, code, start in zip(
        ("amber", "birch", "cobalt", "dune"), codes, (len(tokens), early, middle, length - 20)
    ):
        if start < len(tokens):
            raise ValueError("fact positions overlap or length is too short")
        filler = tokenizer.encode(" The weather was calm.", add_special_tokens=False)
        if not filler:
            raise ValueError("empty filler")
        tokens.extend((filler * ((start - len(tokens)) // len(filler) + 1))[: start - len(tokens)])
        text = f" The code for {label} is {code}."
        encoded = tokenizer.encode(text, add_special_tokens=False)
        tokens.extend(encoded)
        facts.append({"label": label, "code": str(code), "start": start, "end": len(tokens)})
    if len(tokens) > length:
        raise ValueError("final fact does not fit")
    tokens.extend((filler * ((length - len(tokens)) // len(filler) + 1))[: length - len(tokens)])
    if facts[0]["end"] > SINKS or facts[-1]["start"] < length - WINDOW:
        raise ValueError("sink or recent fact is not fully within its intended cache region")
    return tokens, facts


def fork(source):
    other = ExperimentalPositionAdapter(
        source.model,
        max_positions=source.max_positions,
        window=source.cache.layers[0].window,
        sink_tokens=source.cache.layers[0].sink_tokens,
    )
    other.batch_size = source.batch_size
    for src, dst in zip(source.cache.layers, other.cache.layers):
        dst.next_position = src.next_position
        dst.keys = None if src.keys is None else src.keys.clone()
        dst.values = None if src.values is None else src.values.clone()
        dst.positions = None if src.positions is None else src.positions.clone()
    return other


def expected_positions(end, window, sinks):
    return list(range(min(end, sinks))) + list(range(max(sinks, end - window), end))


def check_cache(model_adapter, end):
    expected = expected_positions(end, WINDOW, SINKS)
    config = model_adapter.model.config
    for layer in model_adapter.cache.layers:
        if layer.next_position != end or layer.positions.tolist() != expected:
            raise AssertionError("incorrect cache positions")
        shape = (
            model_adapter.batch_size,
            config.num_key_value_heads,
            len(expected),
            config.head_dim,
        )
        if layer.keys.shape != shape or layer.values.shape != shape:
            raise AssertionError("incorrect cache tensor shape")
        if not torch.isfinite(layer.keys).all() or not torch.isfinite(layer.values).all():
            raise AssertionError("nonfinite cache")


class IndependentNativeReference:
    """Native batched attention with independently managed bounded KV state."""

    def __init__(self, model):
        self.model = model
        self.cache = DynamicCache()
        self.positions = torch.empty(0, dtype=torch.long)
        self.next_position = 0

    @torch.inference_mode()
    def forward(self, tokens):
        start = self.next_position
        end = start + tokens.shape[1]
        query = torch.arange(start, end)
        keys = torch.cat((self.positions, query))
        allowed = (keys[None, :] <= query[:, None]) & (
            (keys[None, :] < SINKS) | (keys[None, :] >= query[:, None] - WINDOW + 1)
        )
        mask = torch.zeros(1, 1, len(query), len(keys), dtype=torch.float32)
        mask.masked_fill_(~allowed[None, None], torch.finfo(torch.float32).min)
        logits = self.model(
            tokens,
            attention_mask=mask,
            position_ids=query[None, :],
            cache_position=query,
            past_key_values=self.cache,
            use_cache=True,
        ).logits
        keep = (keys < SINKS) | (keys >= end - WINDOW)
        self.positions = keys[keep]
        self.next_position = end
        if self.positions.tolist() != expected_positions(end, WINDOW, SINKS):
            raise AssertionError("independent native reference retained incorrect positions")
        for index in range(len(self.cache.key_cache)):
            self.cache.key_cache[index] = self.cache.key_cache[index][..., keep, :]
            self.cache.value_cache[index] = self.cache.value_cache[index][..., keep, :]
            if (
                not torch.isfinite(self.cache.key_cache[index]).all()
                or not torch.isfinite(self.cache.value_cache[index]).all()
            ):
                raise AssertionError("nonfinite native reference cache")
        if not torch.isfinite(logits).all():
            raise AssertionError("nonfinite native reference logits")
        return logits


def native_suffix(model, model_adapter, tokens):
    """Dense native attention over a captured prefix plus a multi-token suffix."""
    start = model_adapter.cache.get_seq_length()
    query = torch.arange(start, start + tokens.shape[1])
    previous = model_adapter.cache.layers[0].positions
    keys = torch.cat((previous, query))
    allowed = (keys[None, :] <= query[:, None]) & (
        (keys[None, :] < SINKS) | (keys[None, :] >= query[:, None] - WINDOW + 1)
    )
    mask = torch.zeros(1, 1, len(query), len(keys), dtype=torch.float32)
    mask.masked_fill_(~allowed[None, None], torch.finfo(torch.float32).min)
    cache = DynamicCache()
    for index, layer in enumerate(model_adapter.cache.layers):
        cache.update(layer.keys.clone(), layer.values.clone(), index)
    with torch.inference_mode():
        out = model(
            tokens,
            attention_mask=mask,
            position_ids=query[None, :],
            cache_position=query,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
        )
    if len(out.attentions) != len(model_adapter.cache.layers):
        raise AssertionError("missing attention probabilities")
    max_masked = 0.0
    for probabilities in out.attentions:
        if probabilities.shape != (1, model.config.num_attention_heads, len(query), len(keys)):
            raise AssertionError("malformed attention probabilities")
        if not torch.isfinite(probabilities).all():
            raise AssertionError("nonfinite attention probabilities")
        masked = probabilities.masked_select(~allowed[None, None])
        if masked.numel():
            max_masked = max(max_masked, float(masked.abs().max()))
        torch.testing.assert_close(
            probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)), atol=1e-5, rtol=1e-5
        )
    if max_masked != 0:
        raise AssertionError("attention assigned nonzero probability to masked keys")
    return out.logits, max_masked


def probe_suffix(model, model_adapter, tokens):
    native, max_masked = native_suffix(model, model_adapter, tokens)
    changed = tokens.clone()
    changed[0, -1] = (int(changed[0, -1]) + 1) % model.config.vocab_size
    changed_native, _ = native_suffix(model, model_adapter, changed)
    control = fork(model_adapter)
    actual = model_adapter.forward(tokens)
    alternate = control.forward(changed)
    if not torch.equal(actual[:, :-1], alternate[:, :-1]):
        raise AssertionError("future token changed earlier adapter logits")
    if not torch.equal(native[:, :-1], changed_native[:, :-1]):
        raise AssertionError("future token changed earlier native logits")
    comparison = compare("captured-prefix native suffix", actual, native)
    return (
        {
            "first_position": model_adapter.cache.get_seq_length() - tokens.shape[1],
            "last_position": model_adapter.cache.get_seq_length() - 1,
            "native_suffix_comparison": comparison,
            "future_token_invariance": True,
            "max_masked_attention_probability": max_masked,
            "finite_attention_probabilities": True,
        },
        actual,
    )


def query_facts(model_adapter, tokenizer, facts):
    results = []
    _, suffix = chat_parts(tokenizer)
    for fact in facts:
        branch = fork(model_adapter)
        question = f"\nWhat is the code for {fact['label']}? Reply with only the digits."
        ids = tokenizer.encode(question + suffix, add_special_tokens=False)
        question_tokens = tokenizer.decode(ids)
        if any(f["code"] in question_tokens for f in facts):
            raise AssertionError("question contains an answer")
        begin = time.perf_counter()
        cpu_begin = time.process_time()
        logits = branch.forward(torch.tensor([ids], dtype=torch.long))[:, -1]
        prefill_seconds = time.perf_counter() - begin
        prefill_cpu_seconds = time.process_time() - cpu_begin
        retained = set(branch.cache.layers[0].positions.tolist())
        fact_fully_retained = set(range(fact["start"], fact["end"])).issubset(retained)
        generated = []
        begin = time.perf_counter()
        cpu_begin = time.process_time()
        for index in range(12):
            token = int(logits.argmax(-1).item())
            generated.append(token)
            if token == tokenizer.eos_token_id or index == 11:
                break
            logits = branch.forward(torch.tensor([[token]], dtype=torch.long))[:, -1]
        decode_seconds = time.perf_counter() - begin
        decode_cpu_seconds = time.process_time() - cpu_begin
        retained_after_decode = set(branch.cache.layers[0].positions.tolist())
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        exact = bool(re.search(rf"(?<!\d){re.escape(fact['code'])}(?!\d)", answer))
        results.append(
            {
                **fact,
                "question": question,
                "question_tokens": len(ids),
                "fact_fully_retained_after_question": fact_fully_retained,
                "fact_fully_retained_after_decode": set(range(fact["start"], fact["end"])).issubset(
                    retained_after_decode
                ),
                "answer": answer,
                "generated_token_ids": generated,
                "exact_code_present": exact,
                "prefill_seconds": prefill_seconds,
                "prefill_cpu_seconds": prefill_cpu_seconds,
                "decode_seconds": decode_seconds,
                "decode_cpu_seconds": decode_cpu_seconds,
                "next_absolute_position": branch.cache.get_seq_length(),
                "finite_logits": bool(torch.isfinite(logits).all()),
            }
        )
    return results


def native_short_controls(model, tokenizer, facts):
    results = []
    for fact in facts:
        content = (
            f"The code for {fact['label']} is {fact['code']}. "
            f"What is the code for {fact['label']}? Reply with only the digits."
        )
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            generated = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(generated[0, ids.shape[1] :], skip_special_tokens=True)
        results.append(
            {
                "label": fact["label"],
                "answer": answer,
                "exact_code_present": bool(
                    re.search(rf"(?<!\d){re.escape(fact['code'])}(?!\d)", answer)
                ),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=2400)
    parser.add_argument("--min-available-mib", type=float, default=1024)
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument(
        "--smoke", action="store_true", help="use a 256-token engineering smoke test"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.progress_output is not None and args.progress_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.progress_output}")
    if args.progress_output is not None and args.progress_output == args.output:
        raise ValueError("progress and final output must be different files")
    if args.max_seconds <= 0 or args.min_available_mib < 1024:
        raise ValueError("invalid resource budget")
    if digest(args.snapshot / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("weight checksum mismatch")
    if available_memory_mib() < args.min_available_mib + 2560:
        raise RuntimeError("insufficient available memory to load the model safely")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()
    targets = (128, 256) if args.smoke else TARGETS
    tokens, facts = build_stream(tokenizer, targets[-1])
    parity_tokens = torch.tensor([tokenizer.encode("Hello world. It is another day.")[:5]])
    parity = original_parity(model, parity_tokens)
    short_controls = native_short_controls(model, tokenizer, facts)
    extended = ExperimentalPositionAdapter(
        model, max_positions=11000, window=WINDOW, sink_tokens=SINKS
    )
    reference = IndependentNativeReference(model)
    cache_time = 0.0
    cache_calls = 0
    attention_time = {"reference": 0.0, "adapter": 0.0, "queries": 0.0}
    activity = "adapter"
    attention_calls = 0
    attention_starts = {}
    original_append = StreamingKVCache.append

    def attention_pre(module, _inputs):
        attention_starts[module] = time.perf_counter()

    def attention_post(module, _inputs, _output):
        nonlocal attention_calls
        attention_time[activity] += time.perf_counter() - attention_starts.pop(module)
        attention_calls += 1

    handles = []
    for layer in model.model.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(attention_pre))
        handles.append(layer.self_attn.register_forward_hook(attention_post))

    def timed_append(self, *a, **kw):
        nonlocal cache_time, cache_calls
        started = time.perf_counter()
        try:
            return original_append(self, *a, **kw)
        finally:
            cache_time += time.perf_counter() - started
            cache_calls += 1

    StreamingKVCache.append = timed_append
    started = time.perf_counter()
    cpu_start = time.process_time()
    progress = []
    probes = []
    comparisons = []
    reference_time = 0.0
    adapter_time = 0.0
    offset = 0
    status, reason = "completed", None
    error = None
    queries = []
    progress_file = (
        args.progress_output.open("x", encoding="utf-8")
        if args.progress_output is not None
        else None
    )
    prefill_wall = prefill_cpu = prefill_cache = prefill_attention = None
    try:
        for target in targets:
            while offset < target:
                if time.perf_counter() - started >= args.max_seconds:
                    status, reason = "incomplete", "wall-time budget reached"
                    break
                if available_memory_mib() < args.min_available_mib:
                    status, reason = "incomplete", "available-memory budget reached"
                    break
                end = min(offset + 64, target - 4) if offset < target - 4 else target
                chunk = torch.tensor([tokens[offset:end]], dtype=torch.long)
                activity = "reference"
                begin = time.perf_counter()
                expected_logits = reference.forward(chunk)
                reference_time += time.perf_counter() - begin
                activity = "adapter"
                begin = time.perf_counter()
                if end == target:
                    probe, logits = probe_suffix(model, extended, chunk)
                    probes.append(probe)
                else:
                    logits = extended.forward(
                        chunk, position_ids=torch.arange(offset, end, dtype=torch.long)[None, :]
                    )
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("nonfinite logits")
                adapter_time += time.perf_counter() - begin
                if logits.shape != expected_logits.shape:
                    raise AssertionError("independent native reference logit shape mismatch")
                differences = (logits - expected_logits).abs()
                comparisons.append(
                    {
                        "start": offset,
                        "end": end,
                        "max_abs_difference": float(differences.max()),
                        "mean_abs_difference": float(differences.mean()),
                        "out_of_tolerance": int(
                            (~torch.isclose(logits, expected_logits, atol=ATOL, rtol=RTOL)).sum()
                        ),
                        "values_compared": logits.numel(),
                    }
                )
                offset = end
                check_cache(extended, offset)
            if status != "completed":
                break
            progress.append(
                {
                    "length": offset,
                    "prefill_wall_seconds": time.perf_counter() - started,
                    "prefill_cpu_seconds": time.process_time() - cpu_start,
                    "cache_append_wall_seconds": cache_time,
                    "cache_append_calls": cache_calls,
                    "attention_wall_seconds_including_cache": dict(attention_time),
                    "attention_calls": attention_calls,
                    "native_reference_wall_seconds": reference_time,
                    "adapter_wall_seconds_including_suffix_probes": adapter_time,
                    "reference_max_abs_difference": max(
                        item["max_abs_difference"] for item in comparisons
                    ),
                    "reference_out_of_tolerance": sum(
                        item["out_of_tolerance"] for item in comparisons
                    ),
                    "cache_bytes": extended.cache.storage_bytes(),
                    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    "available_memory_mib": available_memory_mib(),
                }
            )
            if progress_file is not None:
                progress_file.write(json.dumps(progress[-1]) + "\n")
                progress_file.flush()
            print(f"completed {offset} tokens", flush=True)
        prefill_wall = time.perf_counter() - started
        prefill_cpu = time.process_time() - cpu_start
        prefill_cache = cache_time
        prefill_attention = dict(attention_time)
        activity = "queries"
        queries = query_facts(extended, tokenizer, facts) if status == "completed" else []
    except Exception as exc:
        status, reason, error = "failed", f"{type(exc).__name__}: {exc}", exc
    finally:
        if prefill_wall is None:
            prefill_wall = time.perf_counter() - started
            prefill_cpu = time.process_time() - cpu_start
            prefill_cache = cache_time
            prefill_attention = dict(attention_time)
        StreamingKVCache.append = original_append
        if progress_file is not None:
            progress_file.close()
        for handle in handles:
            handle.remove()
    result = {
        "status": status,
        "reason": reason,
        "smoke": args.smoke,
        "processed_tokens": offset,
        "model_revision": REVISION,
        "weights_sha256": WEIGHT_SHA256,
        "tokenizer_sha256": digest(args.snapshot / "tokenizer.json"),
        "config_sha256": digest(args.snapshot / "config.json"),
        "adapter_sha256": digest(ADAPTER_PATH),
        "streaming_cache_sha256": digest(
            Path(__file__).resolve().parents[2] / "zeta_scaling_toolkit/positions.py"
        ),
        "position_extension_sha256": digest(Path(__file__).with_name("position_extension.py")),
        "validation_sha256": digest(Path(__file__)),
        "stream_sha256": hashlib.sha256(
            json.dumps(tokens, separators=(",", ":")).encode()
        ).hexdigest(),
        "seed": SEED,
        "manifest": extended.manifest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "dtype": "float32",
            "threads": torch.get_num_threads(),
        },
        "budget": {"max_seconds": args.max_seconds, "min_available_mib": args.min_available_mib},
        "original_range_parity": parity,
        "native_short_fact_controls": short_controls,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "milestones": progress,
        "suffix_probes": probes,
        "independent_reference_chunks": comparisons,
        "native_reference_wall_seconds": reference_time,
        "adapter_wall_seconds_including_suffix_probes": adapter_time,
        "prefill_wall_seconds_including_probes": prefill_wall,
        "prefill_cpu_seconds_including_probes": prefill_cpu,
        "prefill_cache_append_wall_seconds_including_probes": prefill_cache,
        "prefill_attention_wall_seconds_including_cache_and_probes": prefill_attention,
        "fact_queries": queries,
        "final_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(f"status: {status}; result: {args.output}", flush=True)
    if error is not None:
        raise RuntimeError(f"validation failed: {reason}") from error


if __name__ == "__main__":
    main()
