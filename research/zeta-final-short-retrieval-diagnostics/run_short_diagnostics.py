"""Run isolated retrieval diagnostics over verified prefixes of the pinned stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "research/zeta100k-fresh-20261117-5bc2985bbe99"
SCRIPT_DIR = REPO / "research/context-100k-experiment-v2"
SNAPSHOT = Path(
    "/home/codespace/.cache/zeta-smollm-hf/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/"
    "snapshots/a10cc1512eabd3dde888204e902eca88bddb4951"
)
STREAM_PATH = RUN_DIR / "stream.safetensors"
FACTS_PATH = RUN_DIR / "retention-results.jsonl"
EXPECTED_STREAM_SHA256 = "c9c43fb35050bc31f8d60e2a9bd600ae960f96d7a7ac1618b41505b2955da322"
MODEL_SHA256 = "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e"
TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
CONFIG_SHA256 = "224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee"
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
SEED = 20261117
WINDOW = 64
SINKS = 64
MAX_POSITIONS = 100512
NEGATIVE_IDENTIFIER = "ZZZZZZZZ"

sys.path.insert(0, str(SCRIPT_DIR))
import run_experiment as original  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_original_facts() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in FACTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def positions(adapter: original.ExperimentalPositionAdapter) -> list[int]:
    storage = adapter.cache.layers[0].storage
    if storage.positions is None:
        return []
    return storage.positions.tolist()


def exact_value(answer: str, value: str) -> bool:
    return bool(re.search(rf"(?<!\d){re.escape(value)}(?!\d)", answer))


def evaluate_query(
    model,
    tokenizer,
    adapter: original.ExperimentalPositionAdapter,
    question: str,
    stream_length: int,
    device: torch.device,
    *,
    expected_value: str | None,
    fact: dict[str, object] | None = None,
) -> dict[str, object]:
    suffix, question_offset = original.question_suffix(tokenizer, question)
    branch = original.clone_adapter(adapter)
    before_positions = positions(branch)
    query_start = stream_length + question_offset
    query_started = time.perf_counter()
    query_cpu_started = time.process_time()
    logits = branch.forward(torch.tensor([suffix], dtype=torch.long, device=device))[:, -1]
    query_positions = positions(branch)
    query_cache_bytes = branch.cache.storage_bytes()
    if not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite logits after retrieval question prefill")

    generated: list[int] = []
    decode_started = time.perf_counter()
    decode_cpu_started = time.process_time()
    for _ in range(12):
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        logits = branch.forward(torch.tensor([[token]], dtype=torch.long, device=device))[:, -1]
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite logits during retrieval decoding")
    decode_seconds = time.perf_counter() - decode_started
    decode_cpu_seconds = time.process_time() - decode_cpu_started
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    final_positions = positions(branch)
    record: dict[str, object] = {
        "question": question,
        "answer": answer,
        "generated_token_ids": generated,
        "question_suffix_token_count": len(suffix),
        "question_start_position": query_start,
        "query_input_end_position": stream_length + len(suffix),
        "cache_positions_before_question": before_positions,
        "cache_positions_after_question_before_generation": query_positions,
        "cache_bytes_after_question_before_generation": query_cache_bytes,
        "cache_positions_after_generation": final_positions,
        "final_cache_next_position": branch.cache.get_seq_length(),
        "decode_seconds": decode_seconds,
        "decode_cpu_seconds": decode_cpu_seconds,
        "query_seconds": time.perf_counter() - query_started,
        "query_cpu_seconds": time.process_time() - query_cpu_started,
        "finite_logits": True,
    }
    if expected_value is not None:
        record["expected_value"] = expected_value
        record["exact_match"] = exact_value(answer, expected_value)
    if fact is not None:
        fact_positions = set(range(int(fact["start"]), int(fact["end"])))
        retained = set(query_positions)
        retained_fact_positions = sorted(fact_positions & retained)
        record.update(
            {
                "identifier": fact["identifier"],
                "fact_target_position": fact["target_position"],
                "fact_start_position": fact["start"],
                "fact_end_position_exclusive": fact["end"],
                "fact_token_count": fact["token_count"],
                "distance_fact_end_to_query_start_tokens": query_start - int(fact["end"]),
                "distance_fact_end_to_question_end_tokens": (
                    stream_length + len(suffix) - int(fact["end"])
                ),
                "retained_fact_positions_at_query": retained_fact_positions,
                "fact_fully_retained_at_query": fact_positions.issubset(retained),
                "fact_evicted_at_query": not fact_positions.issubset(retained),
            }
        )
    del branch
    return record


def run(tokens: int, output_path: Path) -> None:
    if tokens not in (8192, 16384):
        raise ValueError("only the requested 8,192- and 16,384-token diagnostics are allowed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.jsonl")
    if output_path.exists() or progress_path.exists():
        raise FileExistsError("diagnostic output already exists; choose a fresh output path")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    random.seed(SEED)
    started = time.perf_counter()
    cpu_started = time.process_time()
    available_at_start = original.available_memory_mib()
    if available_at_start < 3584:
        raise RuntimeError("less than the original 3,584 MiB model-load gate is available")

    asset_hashes = {
        "weights_sha256": sha256(SNAPSHOT / "model.safetensors"),
        "tokenizer_sha256": sha256(SNAPSHOT / "tokenizer.json"),
        "config_sha256": sha256(SNAPSHOT / "config.json"),
        "corpus_sha256": sha256(Path(original.DEFAULT_CORPUS)),
        "stream_file_sha256": sha256(STREAM_PATH),
    }
    if asset_hashes["weights_sha256"] != MODEL_SHA256:
        raise ValueError("pinned model weights hash mismatch")
    if asset_hashes["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise ValueError("pinned tokenizer hash mismatch")
    if asset_hashes["config_sha256"] != CONFIG_SHA256:
        raise ValueError("pinned model config hash mismatch")
    if asset_hashes["corpus_sha256"] != CORPUS_SHA256:
        raise ValueError("pinned corpus hash mismatch")

    stream = load_file(str(STREAM_PATH), device="cpu")["input_ids"].tolist()
    stream_hash = original.token_digest(stream)
    if len(stream) != 100000 or stream_hash != EXPECTED_STREAM_SHA256:
        raise ValueError("saved source stream token count or SHA-256 mismatch")
    facts = load_original_facts()
    if any(fact["identifier"] == NEGATIVE_IDENTIFIER for fact in facts):
        raise ValueError("negative-control identifier collides with a planted fact")
    included_facts = [fact for fact in facts if int(fact["end"]) <= tokens]
    if not included_facts:
        raise ValueError("no complete original fact falls within this stream prefix")

    tokenizer_load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    tokenizer_load_seconds = time.perf_counter() - tokenizer_load_started
    model_load_started = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            SNAPSHOT,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        .to(torch.device("cpu"))
        .eval()
    )
    model_load_seconds = time.perf_counter() - model_load_started
    if model.config.model_type != "llama" or int(model.config.max_position_embeddings) != 8192:
        raise RuntimeError("loaded model does not match the recorded native Llama configuration")

    adapter = original.ExperimentalPositionAdapter(
        model, max_positions=MAX_POSITIONS, window=WINDOW, sink_tokens=SINKS
    )
    prefill_started = time.perf_counter()
    prefill_cpu_started = time.process_time()
    for offset in range(0, tokens, original.CHUNK_SIZE):
        chunk_ids = torch.tensor(
            stream[offset : min(offset + original.CHUNK_SIZE, tokens)],
            dtype=torch.long,
            device="cpu",
        )[None, :]
        logits = adapter.forward(chunk_ids)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"nonfinite stream logits at prefix offset {offset}")
        processed = min(offset + original.CHUNK_SIZE, tokens)
        if processed % 2048 == 0 or processed == tokens:
            original.check_cache(adapter, processed)
            elapsed = time.perf_counter() - prefill_started
            progress = {
                "processed_tokens": processed,
                "last_position": processed - 1,
                "elapsed_seconds": elapsed,
                "tokens_per_second": processed / elapsed if elapsed else None,
                "available_memory_mib": original.available_memory_mib(),
                "cache_bytes": adapter.cache.storage_bytes(),
                "cache_positions": positions(adapter),
                "finite_logits_and_cache": True,
            }
            with progress_path.open("a", encoding="utf-8") as progress_file:
                progress_file.write(json.dumps(progress, sort_keys=True) + "\n")
                progress_file.flush()
            print(
                f"prefill {processed}/{tokens}: {progress['tokens_per_second']:.3f} tokens/s",
                flush=True,
            )
    original.check_cache(adapter, tokens)
    prefill_seconds = time.perf_counter() - prefill_started
    prefill_cpu_seconds = time.process_time() - prefill_cpu_started
    prefix_cache_positions = positions(adapter)

    results = []
    for fact in included_facts:
        question = str(fact["question"])
        expected_question = (
            f"What is the exact value assigned to fact identifier {fact['identifier']}? "
            "Reply with only the value."
        )
        if question != expected_question:
            raise ValueError(f"recorded question changed for {fact['identifier']}")
        retrieval = evaluate_query(
            model,
            tokenizer,
            adapter,
            question,
            tokens,
            torch.device("cpu"),
            expected_value=str(fact["value"]),
            fact=fact,
        )
        positive = original.short_fact_control(model, tokenizer, fact, torch.device("cpu"))
        retrieval["short_prompt_positive_control"] = positive
        results.append(retrieval)

    negative_question = (
        f"What is the exact value assigned to fact identifier {NEGATIVE_IDENTIFIER}? "
        "Reply with only the value."
    )
    negative = evaluate_query(
        model,
        tokenizer,
        adapter,
        negative_question,
        tokens,
        torch.device("cpu"),
        expected_value=None,
    )
    known_values = [str(fact["value"]) for fact in included_facts]
    negative["identifier_not_planted"] = NEGATIVE_IDENTIFIER
    negative["matches_any_planted_value"] = any(
        exact_value(str(negative["answer"]), value) for value in known_values
    )
    negative["negative_control_pass"] = not negative["matches_any_planted_value"]

    output = {
        "diagnostic": f"{tokens}_token_retrieval",
        "status": "completed",
        "stream_prefix_tokens": tokens,
        "facts_included": len(included_facts),
        "retrieval_exact_matches": sum(bool(item["exact_match"]) for item in results),
        "retrieval_exact_match_score": (
            f"{sum(bool(item['exact_match']) for item in results)}/{len(results)}"
        ),
        "short_prompt_positive_controls_passed": sum(
            bool(item["short_prompt_positive_control"]["exact_match"]) for item in results
        ),
        "negative_control": negative,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "model_native_context_limit": int(model.config.max_position_embeddings),
        "tokenizer_revision": REVISION,
        "configuration": {
            "seed": SEED,
            "chunk_size": original.CHUNK_SIZE,
            "sink_positions": SINKS,
            "recent_positions": WINDOW,
            "precision": "float32",
            "device": "cpu",
            "position_adapter": "ExperimentalPositionAdapter, native unscaled RoPE",
            "max_positions": MAX_POSITIONS,
            "attention_implementation": "eager",
        },
        "asset_hashes": asset_hashes,
        "stream_sha256": stream_hash,
        "stream_source": "read-only prefix of the verified completed-run token stream",
        "prefix_cache_positions_after_prefill": prefix_cache_positions,
        "prefix_cache_bytes_after_prefill": adapter.cache.storage_bytes(),
        "cache_integrity_pass": True,
        "facts": results,
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "available_memory_mib_at_start": available_at_start,
            "available_memory_mib_at_end": original.available_memory_mib(),
            "peak_process_rss_mib": original.peak_rss_mib(),
        },
        "timing": {
            "tokenizer_load_seconds": tokenizer_load_seconds,
            "model_load_seconds": model_load_seconds,
            "stream_prefill_seconds": prefill_seconds,
            "stream_prefill_cpu_seconds": prefill_cpu_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "total_process_cpu_seconds": time.process_time() - cpu_started,
        },
        "errors_or_numerical_anomalies": [],
        "runner_source_sha256": sha256(Path(__file__)),
        "adapter_source_sha256": sha256(SCRIPT_DIR / "compat_adapter.py"),
        "experiment_source_sha256": sha256(SCRIPT_DIR / "run_experiment.py"),
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "diagnostic": output["diagnostic"],
                "status": output["status"],
                "score": output["retrieval_exact_match_score"],
                "positive_controls": (
                    f"{output['short_prompt_positive_controls_passed']}/{output['facts_included']}"
                ),
                "negative_control_pass": negative["negative_control_pass"],
                "total_wall_seconds": output["timing"]["total_wall_seconds"],
                "peak_process_rss_mib": output["environment"]["peak_process_rss_mib"],
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=(8192, 16384), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.tokens, args.output.resolve())


if __name__ == "__main__":
    main()
