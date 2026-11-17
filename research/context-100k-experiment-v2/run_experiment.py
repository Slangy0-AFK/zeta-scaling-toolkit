"""Run the pinned SmolLM2 100,000-token position-extension experiment."""

import argparse
import hashlib
import json
import os
import platform
import random
import re
import resource
import tempfile
import time
from pathlib import Path

import torch
import transformers
from compat_adapter import ExperimentalPositionAdapter, ZetaLlamaAdapter
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
WEIGHT_SHA256 = "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e"
TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
CONFIG_SHA256 = "224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee"
STAGES = (8193, 10000, 16384, 25000, 50000, 65537, 75000, 90000, 100000)
FACT_POSITIONS = (128, 2048, 8000, 16000, 25000, 50000, 75000, 90000)
WINDOW, SINKS, SEED = 64, 64, 20261117
MAX_POSITIONS = 100512
CHECKPOINT_INTERVAL = 2048
CHUNK_SIZE = 64
ATOL, RTOL = 2e-4, 1e-4
ROPE_ATOL = 5e-3
BASELINE_PREFIX_LENGTH = 8064
DEFAULT_CORPUS = Path.home() / ".cache/zeta-scaling-toolkit/tinyshakespeare.txt"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def token_digest(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def safe_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".tmp-", delete=False
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def available_memory_mib() -> float:
    with Path("/proc/meminfo").open(encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable is not reported by /proc/meminfo")


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def peak_gpu_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def backend_encode(tokenizer, text: str):
    return tokenizer.backend_tokenizer.encode(text, add_special_tokens=False)


def render_chat(tokenizer, messages, *, add_generation_prompt: bool) -> str:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return rendered


def find_token_span(offsets, char_start: int, char_end: int) -> tuple[int, int]:
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end and end > start
    ]
    if not indices:
        raise ValueError(f"no tokenizer tokens overlap character span {char_start}:{char_end}")
    return min(indices), max(indices) + 1


def find_source_offset(tokenizer, source: str, minimum: int, target_tokens: int) -> int:
    low, high = minimum, len(source)
    while low < high:
        middle = (low + high) // 2
        if len(backend_encode(tokenizer, source[:middle]).ids) < target_tokens:
            low = middle + 1
        else:
            high = middle
    while low < len(source) and not source[low].isspace():
        low += 1
    if low >= len(source):
        raise ValueError("the corpus is too short to place every fact")
    return low


def generate_facts(tokenizer) -> list[dict[str, object]]:
    rng = random.Random(SEED)
    identifiers = set()
    values = set()
    facts = []
    for index, target in enumerate(FACT_POSITIONS, 1):
        while True:
            identifier = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
            if identifier not in identifiers:
                identifiers.add(identifier)
                break
        while True:
            value = f"{rng.randrange(10**9, 10**10):010d}"
            if value not in values:
                values.add(value)
                break
        fact_text = (
            f"\n\nArchive record {index:02d}.\n"
            f"Fact identifier: {identifier}.\n"
            f"The exact value assigned to {identifier} is {value}.\n\n"
        )
        facts.append(
            {
                "index": index,
                "identifier": identifier,
                "value": value,
                "target_position": target,
                "text": fact_text,
                "token_count": len(backend_encode(tokenizer, fact_text).ids),
            }
        )
    return facts


def build_stream(tokenizer, corpus_path: Path) -> tuple[list[int], list[dict[str, object]], str]:
    if not corpus_path.is_file():
        raise FileNotFoundError(f"the local corpus is required: {corpus_path}")
    corpus_bytes = corpus_path.read_bytes()
    corpus = corpus_bytes.decode("utf-8")
    facts = generate_facts(tokenizer)
    marker = "ZETA_HEADER_MARKER"
    marker_rendered = render_chat(
        tokenizer, [{"role": "assistant", "content": marker}], add_generation_prompt=False
    )
    marker_start = marker_rendered.index(marker)
    marker_encoding = backend_encode(tokenizer, marker_rendered)
    header_token_count, _ = find_token_span(
        marker_encoding.offsets, marker_start, marker_start + len(marker)
    )
    source_offset = 0
    prior_inserted_tokens = 0
    for fact in facts:
        target_content_tokens = (
            int(fact["target_position"]) - header_token_count - prior_inserted_tokens
        )
        source_offset = find_source_offset(tokenizer, corpus, source_offset, target_content_tokens)
        fact["source_char_offset"] = source_offset
        prior_inserted_tokens += int(fact["token_count"])

    pieces = []
    source_offset = 0
    for fact in facts:
        current_offset = int(fact["source_char_offset"])
        pieces.extend((corpus[source_offset:current_offset], str(fact["text"])))
        source_offset = current_offset
    pieces.append(corpus[source_offset:])
    body = "".join(pieces)
    rendered = render_chat(
        tokenizer, [{"role": "assistant", "content": body}], add_generation_prompt=False
    )
    encoded = backend_encode(tokenizer, rendered)
    if len(encoded.ids) < STAGES[-1]:
        raise ValueError(
            f"tokenized corpus contains only {len(encoded.ids)} tokens; 100,000 are required"
        )
    stream_ids = encoded.ids[: STAGES[-1]]
    for fact in facts:
        char_start = rendered.index(str(fact["text"]))
        char_end = char_start + len(str(fact["text"]))
        token_start, token_end = find_token_span(encoded.offsets, char_start, char_end)
        if token_end > len(stream_ids):
            raise ValueError(f"fact {fact['identifier']} falls beyond the 100,000-token stream")
        fact["start"] = token_start
        fact["end"] = token_end
        fact["distance_from_stream_end"] = len(stream_ids) - token_end
    if any(facts[index]["start"] >= facts[index + 1]["start"] for index in range(len(facts) - 1)):
        raise AssertionError("fact positions are not strictly increasing")
    return stream_ids, facts, hashlib.sha256(corpus_bytes).hexdigest()


def compare_logits(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: mismatched logit shapes")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError(f"{name}: non-finite logits")
    delta = (actual - expected).abs()
    out_of_tolerance = int((~torch.isclose(actual, expected, atol=ATOL, rtol=RTOL)).sum())
    return {
        "name": name,
        "shape": list(actual.shape),
        "max_absolute_difference": float(delta.max()),
        "mean_absolute_difference": float(delta.mean()),
        "out_of_tolerance": out_of_tolerance,
        "atol": ATOL,
        "rtol": RTOL,
    }


def run_correctness_gates(model, tokenizer, device: torch.device) -> dict[str, object]:
    text_ids = backend_encode(tokenizer, "A careful observer records a stone arch at dawn.").ids
    if len(text_ids) < 8:
        raise RuntimeError("the pinned tokenizer produced too few parity tokens")
    tokens = torch.tensor([text_ids[:8], text_ids[-8:]], dtype=torch.long, device=device)
    positions = torch.arange(tokens.shape[1], dtype=torch.long, device=device)
    with torch.inference_mode():
        native = model(
            tokens,
            attention_mask=torch.ones_like(tokens),
            position_ids=positions[None, :].expand(tokens.shape[0], -1),
            use_cache=False,
        ).logits
    original = ZetaLlamaAdapter(model)
    actual = original.forward(tokens)
    parity = compare_logits("native prefill versus adapter", actual, native)
    if parity["out_of_tolerance"]:
        raise AssertionError(f"original-range parity failed: {parity}")

    sequential_cache = DynamicCache(config=model.config)
    sequential_outputs = []
    with torch.inference_mode():
        for index in range(tokens.shape[1]):
            output = model(
                tokens[:, index : index + 1],
                attention_mask=torch.ones(
                    (tokens.shape[0], index + 1), dtype=torch.long, device=device
                ),
                position_ids=torch.full(
                    (tokens.shape[0], 1), index, dtype=torch.long, device=device
                ),
                past_key_values=sequential_cache,
                use_cache=True,
            )
            sequential_outputs.append(output.logits)
    sequential = torch.cat(sequential_outputs, dim=1)
    sequential_parity = compare_logits("native sequential versus adapter", actual, sequential)
    if sequential_parity["out_of_tolerance"]:
        raise AssertionError(f"native sequential parity failed: {sequential_parity}")

    full_adapter = ZetaLlamaAdapter(model)
    full_logits = full_adapter.forward(tokens[:1])
    split_adapter = ZetaLlamaAdapter(model)
    split_logits = torch.cat(
        (split_adapter.forward(tokens[:1, :3]), split_adapter.forward(tokens[:1, 3:])), dim=1
    )
    continuation_exact = torch.equal(full_logits, split_logits)
    if not continuation_exact:
        raise AssertionError("incremental continuation differs from one-shot adapter prefill")

    changed = tokens[:1].clone()
    changed[0, -1] = (int(changed[0, -1]) + 1) % model.config.vocab_size
    changed_adapter = ZetaLlamaAdapter(model)
    changed_logits = changed_adapter.forward(changed)
    future_invariant = torch.equal(full_logits[:, :-1], changed_logits[:, :-1])
    if not future_invariant:
        raise AssertionError("changing a future token altered earlier adapter logits")

    bounded_ids = backend_encode(
        tokenizer,
        "The careful observer records a stone arch at dawn while another witness describes "
        "the river, the road, and the distant tower.",
    ).ids
    if len(bounded_ids) < 16:
        raise RuntimeError("the bounded-cache probe needs at least sixteen tokens")
    bounded_tokens = torch.tensor([bounded_ids[:16]], dtype=torch.long, device=device)
    bounded = ExperimentalPositionAdapter(
        model, max_positions=MAX_POSITIONS, window=4, sink_tokens=2
    )
    bounded_logits = bounded.forward(bounded_tokens)
    full_short = ZetaLlamaAdapter(model).forward(bounded_tokens)
    query = torch.arange(bounded_tokens.shape[1], device=device)
    key = torch.arange(bounded_tokens.shape[1], device=device)
    visible = (key[None, :] <= query[:, None]) & (
        (key[None, :] < 2) | (key[None, :] >= query[:, None] - 3)
    )
    mask = torch.zeros(
        (1, 1, len(query), len(key)), dtype=torch.float32, device=device
    ).masked_fill(~visible[None, None], torch.finfo(torch.float32).min)
    with torch.inference_mode():
        bounded_native = model(
            bounded_tokens,
            attention_mask=mask,
            position_ids=query[None, :],
            use_cache=False,
            output_attentions=True,
        )
    bounded_parity = compare_logits(
        "bounded adapter versus native dense sink/window mask",
        bounded_logits,
        bounded_native.logits,
    )
    if bounded_parity["out_of_tolerance"]:
        raise AssertionError(f"bounded-cache parity failed: {bounded_parity}")
    attentions = bounded_native.attentions
    if attentions is None or len(attentions) != model.config.num_hidden_layers:
        raise AssertionError("native attention probabilities were not returned")
    max_masked_probability = 0.0
    for probabilities in attentions:
        masked = probabilities.masked_select(~visible[None, None])
        if masked.numel():
            max_masked_probability = max(max_masked_probability, float(masked.abs().max()))
    if max_masked_probability != 0:
        raise AssertionError("masked keys received nonzero native attention probability")
    expected_retained = [0, 1, 12, 13, 14, 15]
    actual_retained = bounded.cache.layers[0].storage.positions.tolist()
    if actual_retained != expected_retained:
        raise AssertionError(f"incorrect sinks/window retention: {actual_retained}")
    cache_shapes = [
        list(layer.storage.keys.shape)
        for layer in bounded.cache.layers
        if layer.storage.keys is not None
    ]
    if len(cache_shapes) != model.config.num_hidden_layers or any(
        shape[0] != 1 or shape[-2] != len(expected_retained) for shape in cache_shapes
    ):
        raise AssertionError("bounded cache batch or sequence dimensions are incorrect")
    difference = float((full_short - bounded_logits).abs().max())
    if difference == 0:
        raise AssertionError("full and bounded cache paths did not exercise different contexts")

    head_dim = model.config.head_dim
    rope_positions = torch.tensor([[0, 8191, 8192, 65536, 100000]], device=device)
    rope_input = torch.linspace(-1, 1, head_dim, device=device).view(1, 1, 1, head_dim)
    rope_input = rope_input.expand(1, 1, len(rope_positions[0]), head_dim)
    cosine, sine = model.model.rotary_emb(rope_input, rope_positions)
    rotated = rope_input * cosine.unsqueeze(1) + torch.cat(
        (-rope_input[..., head_dim // 2 :], rope_input[..., : head_dim // 2]), dim=-1
    ) * sine.unsqueeze(1)
    base = float(model.config.rope_parameters["rope_theta"])
    frequencies = base ** (
        -torch.arange(0, head_dim, 2, dtype=torch.float64, device=device) / head_dim
    )
    angles = rope_positions.double().unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    first, second = rope_input.double().chunk(2, dim=-1)
    independent_rope = (
        rope_input.double() * angles.cos() + torch.cat((-second, first), dim=-1) * angles.sin()
    )
    rope_error = float((rotated.double() - independent_rope).abs().max())
    if not torch.isfinite(rotated).all() or rope_error > ROPE_ATOL:
        raise AssertionError(f"high-position RoPE check failed: max_abs={rope_error}")

    offset = ExperimentalPositionAdapter(
        model,
        max_positions=MAX_POSITIONS,
        window=3,
        initial_position=65_535,
    )
    high_logits = offset.forward(tokens[:1, :4])
    high_layer = offset.cache.layers[0].storage
    if high_layer.next_position != 65_539 or high_layer.positions.tolist() != [
        65_536,
        65_537,
        65_538,
    ]:
        raise AssertionError("high-position offset probe did not advance through position 65,538")
    if not torch.isfinite(high_logits).all():
        raise AssertionError("high-position offset probe returned non-finite logits")

    return {
        "status": "preflight_passed",
        "model_and_tokenizer_loading": {
            "status": "passed",
            "revision": REVISION,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
        },
        "original_range_native_adapter_parity": {"status": "passed", **parity},
        "native_sequential_parity": {"status": "passed", **sequential_parity},
        "incremental_cache_continuation": {
            "status": "passed",
            "bit_identical_to_single_prefill": continuation_exact,
        },
        "causal_future_token_invariance": {
            "status": "passed",
            "earlier_logits_bit_identical": future_invariant,
        },
        "full_vs_bounded_cache": {
            "status": "passed",
            "full_vs_bounded_max_abs_difference": difference,
            "bounded_native_reference": bounded_parity,
        },
        "attention_mask_and_eviction": {
            "status": "passed",
            "sink_positions": [0, 1],
            "recent_positions": [12, 13, 14, 15],
            "retained_positions": actual_retained,
            "max_masked_attention_probability": max_masked_probability,
            "batch_and_cache_shapes": cache_shapes,
        },
        "finite_logits_and_cache": {"status": "passed"},
        "high_position_rope": {
            "status": "passed",
            "positions": rope_positions[0].tolist(),
            "max_abs_error_vs_fp64": rope_error,
            "absolute_tolerance": ROPE_ATOL,
        },
        "synthetic_high_position_offset_probe": {
            "status": "passed",
            "history": "empty synthetic offset; not part of the 100K stream",
            "processed_positions": [65_535, 65_536, 65_537, 65_538],
            "next_position": high_layer.next_position,
            "retained_positions": high_layer.positions.tolist(),
        },
        "contiguous_staged_run_beyond_65537": {"status": "pending"},
        "tolerances": {"logit_atol": ATOL, "logit_rtol": RTOL},
    }


def expected_positions(end: int) -> list[int]:
    return list(range(min(end, SINKS))) + list(range(max(SINKS, end - WINDOW), end))


def check_cache(adapter: ExperimentalPositionAdapter, end: int) -> None:
    expected = expected_positions(end)
    config = adapter.model.config
    for layer in adapter.cache.layers:
        storage = layer.storage
        if storage.next_position != end or storage.positions is None:
            raise AssertionError("cache logical position did not advance correctly")
        if storage.positions.tolist() != expected:
            raise AssertionError("cache retained incorrect absolute positions")
        shape = (adapter.batch_size, config.num_key_value_heads, len(expected), config.head_dim)
        if storage.keys.shape != shape or storage.values.shape != shape:
            raise AssertionError("cache tensor shape changed")
        if not torch.isfinite(storage.keys).all() or not torch.isfinite(storage.values).all():
            raise AssertionError("nonfinite cache values")


def checkpoint_path(output_dir: Path, position: int) -> Path:
    return output_dir / "checkpoints" / f"position-{position:06d}.safetensors"


def save_checkpoint(
    adapter: ExperimentalPositionAdapter,
    path: Path,
    *,
    stream_sha256: str,
    model_sha256: str,
) -> None:
    tensors = {}
    for index, layer in enumerate(adapter.cache.layers):
        storage = layer.storage
        if storage.keys is None or storage.values is None or storage.positions is None:
            raise RuntimeError("cannot checkpoint an uninitialized cache")
        tensors[f"layer_{index}_keys"] = storage.keys.detach().cpu().contiguous()
        tensors[f"layer_{index}_values"] = storage.values.detach().cpu().contiguous()
    tensors["positions"] = adapter.cache.layers[0].storage.positions.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".checkpoint-", delete=False
        ) as stream:
            temp_path = Path(stream.name)
        save_file(
            tensors,
            str(temp_path),
            metadata={
                "next_position": str(adapter.cache.get_seq_length()),
                "stream_sha256": stream_sha256,
                "model_sha256": model_sha256,
                "model_revision": REVISION,
            },
        )
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def restore_checkpoint(
    adapter: ExperimentalPositionAdapter,
    path: Path,
    *,
    stream_sha256: str,
    model_sha256: str,
    device: torch.device,
) -> int:
    with safe_open(str(path), framework="pt", device=str(device)) as checkpoint:
        metadata = checkpoint.metadata()
    tensors = load_file(str(path), device=str(device))
    if metadata is None or (
        metadata.get("stream_sha256") != stream_sha256
        or metadata.get("model_sha256") != model_sha256
        or metadata.get("model_revision") != REVISION
    ):
        raise ValueError("checkpoint provenance does not match this experiment")
    end = int(metadata["next_position"])
    positions = tensors.pop("positions")
    if positions.tolist() != expected_positions(end):
        raise ValueError("checkpoint retained positions are inconsistent")
    for index, layer in enumerate(adapter.cache.layers):
        storage = layer.storage
        storage.keys = tensors.pop(f"layer_{index}_keys")
        storage.values = tensors.pop(f"layer_{index}_values")
        storage.positions = positions.clone()
        storage.next_position = end
        layer.keys, layer.values = storage.keys, storage.values
        layer.is_initialized = True
    adapter.batch_size = 1
    check_cache(adapter, end)
    return end


def append_progress(path: Path, entry: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".tmp-", delete=False
        ) as stream:
            temp_path = Path(stream.name)
            for entry in entries:
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_progress(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid progress record at line {line_number}") from error
    return entries


def has_completed_results(path: Path) -> bool:
    if not path.exists():
        return False
    results = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(results, dict) or results.get("status") not in (
        "completed",
        "incomplete",
        "failed",
    ):
        raise ValueError("existing resource results have an invalid status")
    return results["status"] == "completed"


def clone_adapter(source: ExperimentalPositionAdapter) -> ExperimentalPositionAdapter:
    first = source.cache.layers[0].storage
    clone = ExperimentalPositionAdapter(
        source.model,
        max_positions=source.max_positions,
        window=first.window,
        sink_tokens=first.sink_tokens,
    )
    clone.batch_size = source.batch_size
    for src, dst in zip(source.cache.layers, clone.cache.layers):
        dst.storage.next_position = src.storage.next_position
        dst.storage.keys = None if src.storage.keys is None else src.storage.keys.clone()
        dst.storage.values = None if src.storage.values is None else src.storage.values.clone()
        dst.storage.positions = (
            None if src.storage.positions is None else src.storage.positions.clone()
        )
        dst.keys, dst.values = dst.storage.keys, dst.storage.values
        dst.is_initialized = src.is_initialized
    return clone


def question_suffix(tokenizer, question: str) -> tuple[list[int], int]:
    close_ids = backend_encode(tokenizer, "<|im_end|>\n").ids
    rendered = render_chat(
        tokenizer, [{"role": "user", "content": question}], add_generation_prompt=True
    )
    user_start = rendered.rfind("<|im_start|>user\n")
    if user_start < 0:
        raise ValueError("the tokenizer chat template has no user-message marker")
    user_message = rendered[user_start:]
    encoded = backend_encode(tokenizer, user_message)
    question_start = user_message.index(question)
    question_token, _ = find_token_span(
        encoded.offsets, question_start, question_start + len(question)
    )
    return close_ids + encoded.ids, len(close_ids) + question_token


def greedy_answer(model, cache, tokenizer, suffix: list[int], start: int, device: torch.device):
    ids = torch.tensor([suffix], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(
            ids,
            attention_mask=torch.ones((1, start + len(suffix)), dtype=torch.long, device=device),
            position_ids=torch.arange(start, start + len(suffix), device=device)[None, :],
            past_key_values=cache,
            use_cache=True,
        )
    logits = output.logits[:, -1]
    if not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite logits during query prefill")
    generated = []
    decode_start = time.perf_counter()
    decode_cpu_start = time.process_time()
    for _ in range(12):
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        next_position = start + len(suffix) + len(generated) - 1
        with torch.inference_mode():
            output = model(
                torch.tensor([[token]], dtype=torch.long, device=device),
                attention_mask=torch.ones((1, next_position + 1), dtype=torch.long, device=device),
                position_ids=torch.tensor([[next_position]], dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
            )
        logits = output.logits[:, -1]
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite logits during query decoding")
    return generated, time.perf_counter() - decode_start, time.process_time() - decode_cpu_start


def short_fact_control(model, tokenizer, fact: dict[str, object], device: torch.device):
    question = (
        f"What is the exact value assigned to fact identifier {fact['identifier']}? "
        "Reply with only the value."
    )
    messages = [
        {"role": "assistant", "content": str(fact["text"])},
        {"role": "user", "content": question},
    ]
    rendered = render_chat(tokenizer, messages, add_generation_prompt=True)
    ids = backend_encode(tokenizer, rendered).ids
    if len(ids) > int(model.config.max_position_embeddings):
        raise ValueError("short fact control exceeds the native context configuration")
    cache = DynamicCache(config=model.config)
    generated, decode_seconds, decode_cpu = greedy_answer(model, cache, tokenizer, ids, 0, device)
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    exact = bool(re.search(rf"(?<!\d){re.escape(str(fact['value']))}(?!\d)", answer))
    return {
        "answer": answer,
        "exact_match": exact,
        "prompt_tokens": len(ids),
        "decode_seconds": decode_seconds,
        "decode_cpu_seconds": decode_cpu,
    }


def run_retention_queries(
    model,
    tokenizer,
    adapter: ExperimentalPositionAdapter,
    facts: list[dict[str, object]],
    stream_length: int,
    device: torch.device,
) -> list[dict[str, object]]:
    results = []
    for fact in facts:
        question = (
            f"What is the exact value assigned to fact identifier {fact['identifier']}? "
            "Reply with only the value."
        )
        suffix, question_offset = question_suffix(tokenizer, question)
        branch = clone_adapter(adapter)
        query_start = stream_length + question_offset
        query_begin = time.perf_counter()
        query_cpu_begin = time.process_time()
        logits = branch.forward(torch.tensor([suffix], dtype=torch.long, device=device))[:, -1]
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite logits during bounded query prefill")
        generated = []
        decode_begin = time.perf_counter()
        decode_cpu_begin = time.process_time()
        for _ in range(12):
            token = int(logits.argmax(dim=-1).item())
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break
            logits = branch.forward(torch.tensor([[token]], dtype=torch.long, device=device))[:, -1]
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite logits during bounded query decoding")
        decode_seconds = time.perf_counter() - decode_begin
        decode_cpu = time.process_time() - decode_cpu_begin
        query_seconds = time.perf_counter() - query_begin
        query_cpu = time.process_time() - query_cpu_begin
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        exact = bool(re.search(rf"(?<!\d){re.escape(str(fact['value']))}(?!\d)", answer))
        retained = set(branch.cache.layers[0].storage.positions.tolist())
        fact_positions = set(range(int(fact["start"]), int(fact["end"])))
        fully_retained = fact_positions.issubset(retained)
        control = short_fact_control(model, tokenizer, fact, device)
        if exact and not fully_retained:
            interpretation = (
                "correct_after_direct_fact_eviction; indirect persistence or guessing is unresolved"
            )
        elif not exact and fully_retained:
            interpretation = "fact span retained; generation did not return the exact value"
        elif not exact and control["exact_match"]:
            interpretation = (
                "fact evicted; short-prompt control passed, long-range retrieval failed"
            )
        elif not exact:
            interpretation = "fact evicted; short-prompt control also failed"
        else:
            interpretation = "fact retained and exact value returned"
        results.append(
            {
                **fact,
                "question": question,
                "answer": answer,
                "generated_token_ids": generated,
                "exact_match": exact,
                "distance_from_query_start_tokens": query_start - int(fact["end"]),
                "question_start_position": query_start,
                "fact_fully_retained_after_query": fully_retained,
                "retained_fact_positions": sorted(fact_positions & retained),
                "short_prompt_control": control,
                "interpretation": interpretation,
                "query_seconds": query_seconds,
                "query_cpu_seconds": query_cpu,
                "decode_seconds": decode_seconds,
                "decode_cpu_seconds": decode_cpu,
                "final_cache_next_position": branch.cache.get_seq_length(),
                "finite_logits": True,
            }
        )
        del branch
    return results


def clone_dynamic_cache(source: DynamicCache, config) -> DynamicCache:
    clone = DynamicCache(config=config)
    for src, dst in zip(source.layers, clone.layers):
        if src.keys is None or src.values is None:
            raise RuntimeError("native baseline cache is incomplete")
        dst.keys = src.keys.clone()
        dst.values = src.values.clone()
        dst.is_initialized = True
    return clone


def run_native_baseline(
    model,
    tokenizer,
    stream_ids: list[int],
    facts: list[dict[str, object]],
    device: torch.device,
) -> list[dict[str, object]]:
    prefix_length = min(BASELINE_PREFIX_LENGTH, len(stream_ids))
    cache = DynamicCache(config=model.config)
    offset = 0
    while offset < prefix_length:
        end = min(offset + CHUNK_SIZE, prefix_length)
        chunk = torch.tensor([stream_ids[offset:end]], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model(
                chunk,
                attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
                position_ids=torch.arange(offset, end, dtype=torch.long, device=device)[None, :],
                past_key_values=cache,
                use_cache=True,
            )
        if not torch.isfinite(output.logits).all():
            raise RuntimeError("native baseline produced nonfinite prefix logits")
        offset = end

    results = []
    for fact in facts:
        if int(fact["end"]) > prefix_length:
            results.append(
                {
                    "identifier": fact["identifier"],
                    "fact_position": fact["start"],
                    "status": "not_in_native_prefix",
                    "native_context_limit": int(model.config.max_position_embeddings),
                    "native_prefix_tokens": prefix_length,
                }
            )
            continue
        question = (
            f"What is the exact value assigned to fact identifier {fact['identifier']}? "
            "Reply with only the value."
        )
        suffix, _ = question_suffix(tokenizer, question)
        total_prompt = prefix_length + len(suffix)
        if total_prompt > int(model.config.max_position_embeddings):
            results.append(
                {
                    "identifier": fact["identifier"],
                    "fact_position": fact["start"],
                    "status": "query_exceeds_native_context_limit",
                    "native_context_limit": int(model.config.max_position_embeddings),
                    "native_prefix_tokens": prefix_length,
                    "query_tokens": len(suffix),
                    "total_prompt_tokens": total_prompt,
                }
            )
            continue
        branch_cache = clone_dynamic_cache(cache, model.config)
        generated, decode_seconds, decode_cpu = greedy_answer(
            model, branch_cache, tokenizer, suffix, prefix_length, device
        )
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        exact = bool(re.search(rf"(?<!\d){re.escape(str(fact['value']))}(?!\d)", answer))
        results.append(
            {
                "identifier": fact["identifier"],
                "fact_position": fact["start"],
                "fact_end": fact["end"],
                "distance_from_query_start_tokens": prefix_length + len(suffix) - int(fact["end"]),
                "status": "completed",
                "answer": answer,
                "expected_value": fact["value"],
                "exact_match": exact,
                "native_context_limit": int(model.config.max_position_embeddings),
                "native_prefix_tokens": prefix_length,
                "query_tokens": len(suffix),
                "total_prompt_tokens": total_prompt,
                "precision": str(next(model.parameters()).dtype),
                "device": str(device),
                "decode_seconds": decode_seconds,
                "decode_cpu_seconds": decode_cpu,
            }
        )
        del branch_cache
    return results


def record_environment(device: torch.device, corpus_path: Path) -> dict[str, object]:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": __import__("tokenizers").__version__,
        "safetensors": __import__("safetensors").__version__,
        "huggingface_hub": __import__("huggingface_hub").__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(),
        "logical_cpu_count": os.cpu_count(),
        "corpus_path": str(corpus_path),
        "available_memory_mib_at_start": available_memory_mib(),
    }


def checkpoint_entry(
    end: int,
    started: float,
    last_checkpoint: float,
    adapter: ExperimentalPositionAdapter,
    device: torch.device,
    stream_sha256: str,
    path: Path,
) -> dict[str, object]:
    elapsed = time.perf_counter() - started
    return {
        "processed_tokens": end,
        "first_position": 0,
        "last_position": end - 1,
        "elapsed_seconds": elapsed,
        "interval_seconds": time.perf_counter() - last_checkpoint,
        "tokens_per_second": end / elapsed if elapsed else None,
        "peak_rss_mib": peak_rss_mib(),
        "peak_gpu_memory_mib": peak_gpu_memory_mib(device),
        "available_memory_mib": available_memory_mib(),
        "cache_bytes": adapter.cache.storage_bytes(),
        "retained_positions": expected_positions(end),
        "is_requested_stage": end in STAGES,
        "checkpoint": str(path),
        "stream_sha256": stream_sha256,
    }


def write_final_report(
    path: Path,
    status: str,
    processed_tokens: int,
    facts: list[dict[str, object]],
    retention: list[dict[str, object]],
    native: list[dict[str, object]],
    resources: dict[str, object],
) -> None:
    by_position = []
    for fact in facts:
        result = next(
            (entry for entry in retention if entry["identifier"] == fact["identifier"]), None
        )
        by_position.append(
            f"| {fact['start']} | {fact['distance_from_stream_end']} | "
            f"{'yes' if result and result['exact_match'] else 'no' if result else 'not run'} | "
            f"{'yes' if result and result['fact_fully_retained_after_query'] else 'no' if result else 'not run'} |"
        )
    native_completed = [entry for entry in native if entry.get("status") == "completed"]
    lines = [
        "# Final report",
        "",
        f"**Status: {status}.**",
        "",
        f"Processed {processed_tokens:,} of 100,000 contiguous tokenizer-produced tokens.",
        "The 100,000-token processing result is separate from useful-context retention.",
        "",
        "## Fact retrieval",
        "",
        "| Fact start position | Distance from stream end | Exact match | Entire fact span retained after query |",
        "|---:|---:|:---:|:---:|",
        *by_position,
        "",
        "The values were randomly generated from a recorded deterministic seed. Questions did not contain expected values. "
        "A correct answer after eviction is not sufficient to distinguish indirect persistence from guessing.",
        "",
        "## Native baseline",
        "",
        f"The native model's configured context length is 8,192 tokens. The matched native prefix run used "
        f"{resources.get('native_prefix_tokens', 'not recorded')} tokens, with query prompts kept within that configured "
        "limit where possible. It is not a matched 100,000-token baseline.",
        f"Completed native fact queries: {len(native_completed)} of {len(native)}.",
        "The experimental path processes a single 100,000-token stream; the native baseline is only a same-stream prefix.",
        "",
        "## Resources",
        "",
        f"- Total wall time: {resources.get('total_elapsed_seconds', 'not completed')} seconds",
        f"- Stream prefill wall time: {resources.get('prefill_seconds', 'not completed')} seconds",
        f"- Stream throughput: {resources.get('tokens_per_second', 'not completed')} tokens/second",
        f"- Peak resident memory: {resources.get('peak_rss_mib', 'not recorded')} MiB",
        f"- Peak GPU memory: {resources.get('peak_gpu_memory_mib', 'not applicable')} MiB",
        f"- Final bounded KV cache: {resources.get('final_cache_bytes', 'not recorded')} bytes",
        "",
        "## Scientific conclusion",
        "",
        "No compute or latency savings are claimed: there is no matched native 100K full-history baseline. "
        "Processing tokens, retaining useful information, reducing reprocessed tokens, and reducing compute are distinct claims. "
        "No numeric context-retention pass threshold was supplied, so this report does not assign CONTEXT RETENTION PASS.",
        "",
        "All raw per-stage and per-fact records are retained in `progress.jsonl`, `retention-results.jsonl`, "
        "`native-baseline.jsonl`, and `resource-results.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--max-seconds", type=float, default=43_200)
    parser.add_argument("--min-available-mib", type=float, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.min_available_mib < 1024:
        raise ValueError("invalid runtime or memory budget")
    snapshot = args.snapshot.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    resource_path = output_dir / "resource-results.json"
    retention_path = output_dir / "retention-results.jsonl"
    native_path = output_dir / "native-baseline.jsonl"
    correctness_path = output_dir / "CORRECTNESS_RESULTS.json"
    stream_path = output_dir / "stream.safetensors"
    generated_paths = (progress_path, resource_path, retention_path, native_path, stream_path)
    if not args.resume and any(path.exists() for path in generated_paths):
        raise FileExistsError(
            "experiment outputs already exist; use --resume or a new output directory"
        )
    if (
        args.resume
        and not progress_path.exists()
        and not (correctness_path.exists() and stream_path.exists())
    ):
        raise FileNotFoundError("cannot resume without preflight artifacts or progress.jsonl")
    if args.resume and has_completed_results(resource_path):
        print(f"experiment already has final results: {resource_path}", flush=True)
        return
    for filename, expected in (
        ("model.safetensors", WEIGHT_SHA256),
        ("tokenizer.json", TOKENIZER_SHA256),
        ("config.json", CONFIG_SHA256),
    ):
        actual = digest(snapshot / filename)
        if actual != expected:
            raise ValueError(f"{filename} SHA-256 mismatch: {actual}")
    if available_memory_mib() < args.min_available_mib + 2560:
        raise RuntimeError("insufficient available memory to load the model safely")

    started = time.perf_counter()
    cpu_started = time.process_time()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = record_environment(device, args.corpus)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    tokenizer_validation = backend_encode(tokenizer, "tokenizer compatibility probe")
    wrapped = render_chat(
        tokenizer,
        [{"role": "assistant", "content": "tokenizer compatibility probe"}],
        add_generation_prompt=False,
    )
    if (
        backend_encode(tokenizer, wrapped).ids
        != tokenizer.apply_chat_template(
            [{"role": "assistant", "content": "tokenizer compatibility probe"}],
            tokenize=True,
            add_generation_prompt=False,
        )["input_ids"]
    ):
        raise RuntimeError("backend tokenization differs from the native chat template")
    if not tokenizer_validation.ids:
        raise RuntimeError("tokenizer compatibility probe returned no tokens")

    model_load_start = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            snapshot,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        .to(device)
        .eval()
    )
    model_load_seconds = time.perf_counter() - model_load_start
    if model.config.model_type != "llama" or int(model.config.max_position_embeddings) != 8192:
        raise RuntimeError("the loaded checkpoint does not match the recorded native Llama context")
    model_sha256 = digest(snapshot / "model.safetensors")
    tokenizer_sha256 = digest(snapshot / "tokenizer.json")
    stream_ids, facts, corpus_sha256 = build_stream(tokenizer, args.corpus)
    stream_sha256 = token_digest(stream_ids)
    if len(stream_ids) != STAGES[-1]:
        raise AssertionError("constructed stream does not contain exactly 100,000 tokens")

    progress_entries = load_progress(progress_path)
    if args.resume:
        correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
        if correctness.get("status") not in ("preflight_passed", "all_gates_passed"):
            raise ValueError("the recorded correctness gates do not permit resumption")
        if correctness.get("stream_sha256") != stream_sha256:
            raise ValueError("the deterministic input stream differs from the recorded run")
        if not stream_path.is_file():
            raise FileNotFoundError("the recorded stream token file is missing")
        with safe_open(str(stream_path), framework="pt", device="cpu") as stream_file:
            stream_metadata = stream_file.metadata()
        if (
            stream_metadata is None
            or stream_metadata.get("stream_sha256") != stream_sha256
            or stream_metadata.get("tokenizer_sha256") != tokenizer_sha256
        ):
            raise ValueError("stored stream provenance does not match the current run")
        stored = load_file(str(stream_path), device="cpu")["input_ids"].tolist()
        if stored != stream_ids:
            raise ValueError("stored tokenizer IDs differ from the rebuilt input stream")
        preflight = correctness
    else:
        preflight = run_correctness_gates(model, tokenizer, device)
        preflight.update(
            {
                "model_id": MODEL_ID,
                "model_revision": REVISION,
                "weights_sha256": model_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "stream_sha256": stream_sha256,
                "corpus_sha256": corpus_sha256,
                "source_commit": "ee3a7c8f5702b73ae8e2c2dc2d73930b09efa2ed",
            }
        )
        safe_write_json(correctness_path, preflight)
        save_file(
            {"input_ids": torch.tensor(stream_ids, dtype=torch.int64)},
            str(stream_path),
            metadata={
                "stream_sha256": stream_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "model_revision": REVISION,
                "token_count": str(len(stream_ids)),
            },
        )
    if args.preflight_only:
        print(
            f"preflight status: {preflight['status']}; stream sha256: {stream_sha256}",
            flush=True,
        )
        return
    environment["model_load_seconds"] = model_load_seconds
    environment["model_max_position_embeddings"] = int(model.config.max_position_embeddings)
    environment["model_revision"] = REVISION
    environment["model_sha256"] = model_sha256
    environment["tokenizer_sha256"] = tokenizer_sha256
    environment["corpus_sha256"] = corpus_sha256
    environment["stream_sha256"] = stream_sha256
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    adapter = ExperimentalPositionAdapter(
        model, max_positions=MAX_POSITIONS, window=WINDOW, sink_tokens=SINKS
    )
    offset = 0
    previous_checkpoint_time = time.perf_counter()
    if args.resume and progress_entries:
        last_entry = progress_entries[-1]
        offset = int(last_entry["processed_tokens"])
        path = Path(str(last_entry["checkpoint"]))
        if not path.is_absolute():
            path = output_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint is missing: {path}")
        restored = restore_checkpoint(
            adapter,
            path,
            stream_sha256=stream_sha256,
            model_sha256=model_sha256,
            device=device,
        )
        if restored != offset:
            raise ValueError("checkpoint position differs from progress.jsonl")
    prefill_started = time.perf_counter()
    status = "running"
    reason = None
    stage_targets = [stage for stage in STAGES if stage > offset]
    next_stage_index = 0
    next_periodic = ((offset // CHECKPOINT_INTERVAL) + 1) * CHECKPOINT_INTERVAL
    error = None
    try:
        while offset < STAGES[-1]:
            if time.perf_counter() - prefill_started >= args.max_seconds:
                status, reason = "incomplete", "experiment wall-time budget reached"
                break
            available = available_memory_mib()
            if available < args.min_available_mib:
                status, reason = "incomplete", "available-memory budget reached"
                break
            while (
                next_stage_index < len(stage_targets) and stage_targets[next_stage_index] <= offset
            ):
                next_stage_index += 1
            next_target = STAGES[-1]
            if next_periodic > offset:
                next_target = min(next_target, next_periodic)
            if next_stage_index < len(stage_targets):
                next_target = min(next_target, stage_targets[next_stage_index])
            end = min(offset + CHUNK_SIZE, next_target)
            chunk = torch.tensor(stream_ids[offset:end], dtype=torch.long, device=device)[None, :]
            position_ids = torch.arange(offset, end, dtype=torch.long, device=device)[None, :]
            logits = adapter.forward(chunk, position_ids=position_ids)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"nonfinite stream logits at positions {offset}:{end}")
            offset = end
            check_cache(adapter, offset)
            requested_stage = offset in STAGES
            periodic_checkpoint = offset == next_periodic
            if requested_stage or periodic_checkpoint:
                path = checkpoint_path(output_dir, offset)
                save_checkpoint(
                    adapter, path, stream_sha256=stream_sha256, model_sha256=model_sha256
                )
                entry = checkpoint_entry(
                    offset,
                    prefill_started,
                    previous_checkpoint_time,
                    adapter,
                    device,
                    stream_sha256,
                    path.relative_to(output_dir),
                )
                append_progress(progress_path, entry)
                progress_entries.append(entry)
                previous_checkpoint_time = time.perf_counter()
                print(
                    f"checkpoint {offset:,}/{STAGES[-1]:,}: "
                    f"{entry['tokens_per_second']:.3f} tokens/s",
                    flush=True,
                )
                if offset == 65_537:
                    preflight["contiguous_staged_run_beyond_65537"] = {
                        "status": "passed",
                        "processed_contiguously_from_position_zero": True,
                        "processed_tokens": offset,
                        "last_position": offset - 1,
                        "cache_positions": expected_positions(offset),
                        "checkpoint": str(path.relative_to(output_dir)),
                    }
                    preflight["status"] = "all_gates_passed"
                    safe_write_json(correctness_path, preflight)
                if requested_stage:
                    next_stage_index += 1
                if periodic_checkpoint:
                    next_periodic += CHECKPOINT_INTERVAL
    except Exception as exc:
        status = "failed"
        reason = f"{type(exc).__name__}: {exc}"
        error = exc
    prefill_seconds = time.perf_counter() - prefill_started
    if status == "running":
        status = "completed"
    retention = []
    native = []
    if status == "completed":
        if args.resume and retention_path.exists() and native_path.exists():
            retention = [json.loads(line) for line in retention_path.read_text().splitlines()]
            native = [json.loads(line) for line in native_path.read_text().splitlines()]
            query_seconds = 0.0
            native_seconds = 0.0
        else:
            query_started = time.perf_counter()
            retention = run_retention_queries(model, tokenizer, adapter, facts, offset, device)
            query_seconds = time.perf_counter() - query_started
            native_started = time.perf_counter()
            native = run_native_baseline(model, tokenizer, stream_ids, facts, device)
            native_seconds = time.perf_counter() - native_started
            write_jsonl(retention_path, retention)
            write_jsonl(native_path, native)
        evaluation = {
            "query_seconds": query_seconds,
            "native_baseline_seconds": native_seconds,
            "native_prefix_tokens": min(BASELINE_PREFIX_LENGTH, offset),
        }
    else:
        evaluation = {"query_seconds": None, "native_baseline_seconds": None}
    total_elapsed = time.perf_counter() - started
    resource_results = {
        "status": status,
        "reason": reason,
        "processed_tokens": offset,
        "requested_tokens": STAGES[-1],
        "model_load_seconds": model_load_seconds,
        "prefill_seconds": prefill_seconds,
        "tokens_per_second": offset / prefill_seconds if prefill_seconds else None,
        "evaluation": evaluation,
        "total_elapsed_seconds": total_elapsed,
        "total_cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_mib": peak_rss_mib(),
        "peak_gpu_memory_mib": peak_gpu_memory_mib(device),
        "final_cache_bytes": adapter.cache.storage_bytes(),
        "final_cache_positions": expected_positions(offset),
        "environment": environment,
        "device": str(device),
        "precision": str(next(model.parameters()).dtype),
        "stream_sha256": stream_sha256,
        "corpus_sha256": corpus_sha256,
        "experiment_budget": {
            "max_seconds": args.max_seconds,
            "minimum_available_memory_mib": args.min_available_mib,
        },
        "retention_exact_matches": sum(bool(item["exact_match"]) for item in retention),
        "retention_fact_count": len(retention),
        "native_completed_queries": sum(item.get("status") == "completed" for item in native),
    }
    safe_write_json(resource_path, resource_results)
    final_status = (
        "100K PROCESSING PASS"
        if status == "completed" and len(retention) == len(facts)
        else "PARTIAL"
        if offset > 0
        else "BLOCKED"
        if status != "failed"
        else "NO-GO"
    )
    write_final_report(
        output_dir / "FINAL_REPORT.md",
        final_status,
        offset,
        facts,
        retention,
        native,
        {
            **resource_results,
            "native_prefix_tokens": evaluation.get("native_prefix_tokens"),
        },
    )
    print(f"experiment status: {final_status}; output: {output_dir}", flush=True)
    if error is not None:
        raise RuntimeError(f"experiment failed: {reason}") from error
    if status == "incomplete":
        raise RuntimeError(f"experiment is resumable but incomplete: {reason}")


if __name__ == "__main__":
    main()
