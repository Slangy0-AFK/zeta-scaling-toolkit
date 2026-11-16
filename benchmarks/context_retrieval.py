"""Controlled synthetic associative retrieval benchmark.

The smoke phase validates task/model wiring only. It does not train a model
and its outputs are not experiment results.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import random
import struct
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    ExperimentManifest,
    compute_source_digest,
)

if __package__:
    from .context_successor import candidate as candidate_settings
    from .length_generalization import SequenceLM, bootstrap_interval, positive_int
else:
    from context_successor import candidate as candidate_settings
    from length_generalization import SequenceLM, bootstrap_interval, positive_int


PROTOCOL_VERSION = 1
CONTEXT_LENGTHS = (256, 512, 1024, 2048, 4096)
INITIAL_METHODS = ("rope-log-full", "rope-full", "alibi", "rope-log-window")
COMPARISON_METHODS = ("rope-log-full", "rope-full", "alibi")
CANDIDATES = (0, 1)
TRAINING_LENGTH = 256
TRAINING_PAIR_COUNT = 16
TRAIN_POOL_SIZE = 4096
SANITY_SEEDS = tuple(range(200, 210))
TUNING_SEEDS = tuple(range(300, 310))
REPORT_SEEDS = tuple(range(400, 410))
POSITION_BUCKETS = ("early", "middle", "late")
DISTANCE_BUCKETS = (
    (0, 64, "0_63"),
    (64, 256, "64_255"),
    (256, 1024, "256_1023"),
    (1024, 4096, "1024_4095"),
)

NOISE_START = 0
NOISE_COUNT = 32
KEY_START = NOISE_START + NOISE_COUNT
KEY_COUNT = 64
VALUE_START = KEY_START + KEY_COUNT
VALUE_COUNT = 64
PAIR_MARKER = VALUE_START + VALUE_COUNT
QUERY_MARKER = PAIR_MARKER + 1
VOCABULARY_SIZE = QUERY_MARKER + 1


@dataclass(frozen=True)
class RetrievalExample:
    tokens: tuple[int, ...]
    target: int
    split: str
    seed: int
    sample_index: int
    context_length: int
    pair_count: int
    target_key: int
    target_position: int
    query_position: int
    target_distance: int
    target_position_bucket: str
    pair_order: tuple[int, ...]

    @property
    def example_id(self) -> str:
        token_bytes = struct.pack(f"<{len(self.tokens)}H", *self.tokens)
        target_bytes = struct.pack("<H", self.target)
        return hashlib.sha256(token_bytes + target_bytes).hexdigest()


def _derived_seed(seed: int, split: str, sample_index: int) -> int:
    material = f"zeta-retrieval-v{PROTOCOL_VERSION}:{split}:{seed}:{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _position_bucket(position: int, context_length: int) -> str:
    fraction = position / (context_length - 2)
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def _pair_slots(context_length: int, pair_count: int) -> list[int]:
    if context_length < 4 or pair_count < 3 or pair_count > min(KEY_COUNT, VALUE_COUNT):
        raise ValueError("context must fit at least three pairs and pair_count must fit the vocab")
    query_start = context_length - 2
    if pair_count * 3 + 2 > context_length:
        raise ValueError("context is too short for the requested number of pairs")
    last_start = query_start - 3
    slots = [round(index * last_start / (pair_count - 1)) for index in range(pair_count)]
    if any(right - left < 3 for left, right in zip(slots, slots[1:])):
        raise ValueError("pair slots overlap; reduce pair_count or increase context length")
    return slots


def generate_example(
    context_length: int,
    pair_count: int,
    seed: int,
    *,
    split: str = "train",
    sample_index: int = 0,
    target_bucket: str | None = None,
) -> RetrievalExample:
    """Generate an exact-length query-conditioned retrieval example."""
    if not isinstance(seed, int) or seed < 0 or sample_index < 0:
        raise ValueError("seed and sample_index must be nonnegative integers")
    if target_bucket is not None and target_bucket not in POSITION_BUCKETS:
        raise ValueError(f"target_bucket must be one of {POSITION_BUCKETS}")
    slots = _pair_slots(context_length, pair_count)
    rng = random.Random(_derived_seed(seed, split, sample_index))
    tokens = [rng.randrange(NOISE_START, NOISE_START + NOISE_COUNT) for _ in range(context_length)]

    keys = rng.sample(range(KEY_START, KEY_START + KEY_COUNT), pair_count)
    values = rng.sample(range(VALUE_START, VALUE_START + VALUE_COUNT), pair_count)
    pairs = list(zip(keys, values))
    rng.shuffle(pairs)

    slot_buckets = [_position_bucket(start + 2, context_length) for start in slots]
    selected_bucket = target_bucket or rng.choice(POSITION_BUCKETS)
    eligible_slots = [
        index for index, bucket in enumerate(slot_buckets) if bucket == selected_bucket
    ]
    if not eligible_slots:
        raise ValueError(f"no pair slot available in target position bucket {selected_bucket}")
    target_slot = rng.choice(eligible_slots)

    for start, (key, value) in zip(slots, pairs):
        tokens[start : start + 3] = [PAIR_MARKER, key, value]
    query_start = context_length - 2
    target_key, target = pairs[target_slot]
    tokens[query_start:] = [QUERY_MARKER, target_key]
    example = RetrievalExample(
        tokens=tuple(tokens),
        target=target,
        split=split,
        seed=seed,
        sample_index=sample_index,
        context_length=context_length,
        pair_count=pair_count,
        target_key=target_key,
        target_position=slots[target_slot] + 2,
        query_position=context_length - 1,
        target_distance=context_length - 1 - (slots[target_slot] + 2),
        target_position_bucket=selected_bucket,
        pair_order=tuple(key for key, _ in pairs),
    )
    validate_example(example)
    return example


def stored_pairs(tokens: Sequence[int]) -> list[tuple[int, int, int]]:
    """Return stored (start, key, value) triples from a serialized input."""
    pairs = []
    for position, token in enumerate(tokens[:-2]):
        if token == PAIR_MARKER:
            if position + 2 >= len(tokens) - 2:
                raise ValueError("truncated stored pair")
            pairs.append((position, tokens[position + 1], tokens[position + 2]))
    return pairs


def oracle_target(tokens: Sequence[int]) -> int:
    """Resolve the query by matching its key against the stored pairs."""
    if len(tokens) < 5 or tokens[-2] != QUERY_MARKER:
        raise ValueError("input must end in a query marker and query key")
    pairs = stored_pairs(tokens)
    keys = [key for _, key, _ in pairs]
    values = [value for _, _, value in pairs]
    if len(set(keys)) != len(keys) or len(set(values)) != len(values):
        raise ValueError("stored keys and values must be unique")
    mapping = dict(zip(keys, values))
    query_key = tokens[-1]
    if keys.count(query_key) != 1 or query_key not in mapping:
        raise ValueError("query key must identify exactly one stored pair")
    return mapping[query_key]


def validate_example(example: RetrievalExample) -> None:
    if len(example.tokens) != example.context_length:
        raise ValueError("serialized input length does not match context_length")
    if example.tokens[-2:] != (QUERY_MARKER, example.target_key):
        raise ValueError("query must occupy the final two context positions")
    pairs = stored_pairs(example.tokens)
    keys = [key for _, key, _ in pairs]
    values = [value for _, _, value in pairs]
    if len(pairs) != example.pair_count or len(set(keys)) != example.pair_count:
        raise ValueError("each stored pair must have a unique key")
    if len(set(values)) != example.pair_count:
        raise ValueError("each stored pair must have a unique value")
    if any(not KEY_START <= key < KEY_START + KEY_COUNT for key in keys):
        raise ValueError("stored keys must use the key token class")
    if any(not VALUE_START <= value < VALUE_START + VALUE_COUNT for value in values):
        raise ValueError("stored values must use the value token class")
    if tuple(key for _, key, _ in pairs) != example.pair_order:
        raise ValueError("pair order metadata does not match the serialized context")
    if keys.count(example.target_key) != 1:
        raise ValueError("query key must occur exactly once among stored keys")
    if example.tokens.count(example.target_key) != 2:
        raise ValueError("query key must occur once in storage and once in the query")
    if example.tokens.count(example.target) != 1 or example.target not in values:
        raise ValueError("answer must occur exactly once, in its stored pair")
    if oracle_target(example.tokens) != example.target:
        raise ValueError("oracle lookup does not match the generated label")
    target_start = next(start for start, key, _ in pairs if key == example.target_key)
    if (
        example.target_position != target_start + 2
        or example.query_position != example.context_length - 1
        or example.target_distance != example.query_position - example.target_position
    ):
        raise ValueError("target and query position metadata is inconsistent")
    stored_positions = {position for position, _, _ in pairs}
    for position, token in enumerate(example.tokens[:-2]):
        if position in stored_positions:
            continue
        if position - 1 in stored_positions or position - 2 in stored_positions:
            continue
        if not NOISE_START <= token < NOISE_START + NOISE_COUNT:
            raise ValueError("non-pair context positions must contain unrelated noise")
    actual_bucket = _position_bucket(example.target_position, example.context_length)
    if actual_bucket != example.target_position_bucket:
        raise ValueError("target position bucket metadata is inconsistent")


def permute_distractors(example: RetrievalExample, seed: int) -> RetrievalExample:
    """Reorder non-target pairs without moving the target pair or query."""
    rng = random.Random(seed)
    pairs = stored_pairs(example.tokens)
    target_pair = next(pair for pair in pairs if pair[1] == example.target_key)
    original_distractors = [pair for pair in pairs if pair[1] != example.target_key]
    distractors = original_distractors.copy()
    rng.shuffle(distractors)
    if len(distractors) > 1 and distractors == original_distractors:
        distractors = distractors[1:] + distractors[:1]
    distractor_iter = iter(distractors)
    rebuilt = list(example.tokens)
    for start, key, value in pairs:
        if key == example.target_key:
            continue
        new_key, new_value = next(distractor_iter)[1:]
        rebuilt[start : start + 3] = [PAIR_MARKER, new_key, new_value]
    target_start, target_key, target_value = target_pair
    rebuilt[target_start : target_start + 3] = [PAIR_MARKER, target_key, target_value]
    result = RetrievalExample(
        tokens=tuple(rebuilt),
        target=example.target,
        split=example.split,
        seed=example.seed,
        sample_index=example.sample_index,
        context_length=example.context_length,
        pair_count=example.pair_count,
        target_key=example.target_key,
        target_position=example.target_position,
        query_position=example.query_position,
        target_distance=example.target_distance,
        target_position_bucket=example.target_position_bucket,
        pair_order=tuple(key for _, key, _ in stored_pairs(rebuilt)),
    )
    validate_example(result)
    return result


def ablate_stored_pairs(example: RetrievalExample, seed: int = 0) -> tuple[int, ...]:
    """Remove all stored pairs while preserving query, length, and noise-only input."""
    tokens = list(example.tokens)
    for start, _, _ in stored_pairs(tokens):
        for offset in range(3):
            tokens[start + offset] = NOISE_START + ((seed + start + offset) % NOISE_COUNT)
    return tuple(tokens)


def random_answer(example: RetrievalExample, seed: int = 0) -> int:
    rng = random.Random(_derived_seed(seed, "random-control", int(example.example_id[:16], 16)))
    return rng.randrange(VALUE_START, VALUE_START + VALUE_COUNT)


def generate_dataset(
    split: str,
    seed: int,
    sample_count: int,
    context_length: int,
    pair_count: int,
) -> list[RetrievalExample]:
    if sample_count < 3 or sample_count % 3:
        raise ValueError("sample_count must be a positive multiple of three")
    examples = [
        generate_example(
            context_length,
            pair_count,
            seed,
            split=split,
            sample_index=index,
            target_bucket=POSITION_BUCKETS[index % 3],
        )
        for index in range(sample_count)
    ]
    if len({example.example_id for example in examples}) != sample_count:
        raise ValueError("generated dataset contains duplicate examples")
    return examples


def fingerprint_examples(examples: Iterable[RetrievalExample]) -> str:
    identifiers = sorted(example.example_id for example in examples)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("cannot fingerprint duplicate examples")
    return hashlib.sha256("".join(identifiers).encode()).hexdigest()


def example_registry_records(
    split: str, examples: Iterable[RetrievalExample]
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "example_id": example.example_id,
            "seed": example.seed,
            "sample_index": example.sample_index,
            "context_length": example.context_length,
            "pair_count": example.pair_count,
            "pair_order": list(example.pair_order),
            "target_position": example.target_position,
            "query_position": example.query_position,
            "target_distance": example.target_distance,
            "target_position_bucket": example.target_position_bucket,
        }
        for example in examples
    ]


def write_example_registry(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically extend the split-ID registry, rejecting cross-split reuse."""
    incoming = list(records)
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {record["example_id"]: record for record in existing}
    if len(by_id) != len(existing):
        raise ValueError("example registry contains duplicate IDs")
    for record in incoming:
        previous = by_id.get(record["example_id"])
        if previous is not None and previous["split"] != record["split"]:
            raise ValueError(
                f"example leakage across {previous['split']} and {record['split']} splits"
            )
        if previous is not None and previous != record:
            raise ValueError("example ID metadata differs from existing registry")
        by_id[record["example_id"]] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(by_id[key], sort_keys=True) + "\n" for key in sorted(by_id)),
        encoding="utf-8",
    )
    temporary.replace(path)


def metric_counts(
    examples: Sequence[RetrievalExample], predictions: Sequence[int]
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("each example must have exactly one prediction")

    def summarize(indices: Sequence[int]) -> dict[str, int | float]:
        correct = sum(predictions[index] == examples[index].target for index in indices)
        total = len(indices)
        return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}

    overall = summarize(list(range(len(examples))))
    groups: dict[str, dict[str, list[int]]] = {
        "target_position": defaultdict(list),
        "target_distance": defaultdict(list),
        "distractor_count": defaultdict(list),
    }
    for index, example in enumerate(examples):
        groups["target_position"][example.target_position_bucket].append(index)
        groups["target_distance"][_distance_bucket(example.target_distance)].append(index)
        groups["distractor_count"][str(example.pair_count - 1)].append(index)
    return {
        "overall": overall,
        "by_target_position": {
            bucket: summarize(indices)
            for bucket, indices in sorted(groups["target_position"].items())
        },
        "by_target_distance": {
            bucket: summarize(indices)
            for bucket, indices in sorted(groups["target_distance"].items())
        },
        "by_distractor_count": {
            bucket: summarize(indices)
            for bucket, indices in sorted(groups["distractor_count"].items())
        },
    }


def _distance_bucket(distance: int) -> str:
    for lower, upper, label in DISTANCE_BUCKETS:
        if lower <= distance < upper:
            return label
    return f"{DISTANCE_BUCKETS[-1][2]}_plus"


def evaluate_predictions(
    examples: Sequence[RetrievalExample], predictions: Sequence[int]
) -> dict[str, Any]:
    if any(
        type(prediction) is not int or not 0 <= prediction < VOCABULARY_SIZE
        for prediction in predictions
    ):
        raise ValueError("predictions must be vocabulary token IDs")
    return metric_counts(examples, predictions)


def evaluate_model(
    model: SequenceLM,
    examples: Sequence[RetrievalExample],
    *,
    batch_size: int = 8,
    ablate_pairs: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            rows = [
                ablate_stored_pairs(example, seed=example.sample_index)
                if ablate_pairs
                else example.tokens
                for example in batch
            ]
            inputs = torch.tensor(rows, dtype=torch.long, device=device)
            logits = model.forward_last(inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite retrieval evaluation logits")
            predictions.extend(int(token) for token in logits.argmax(dim=-1).cpu().tolist())
    return evaluate_predictions(examples, predictions)


def evaluate_random_control(examples: Sequence[RetrievalExample], seed: int = 0) -> dict[str, Any]:
    return evaluate_predictions(examples, [random_answer(example, seed) for example in examples])


def _source_digests() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[1]
    toolkit_digest = compute_source_digest(list((root / "zeta_scaling_toolkit").rglob("*.py")))
    benchmark_digest = compute_source_digest(list((root / "benchmarks").rglob("*.py")))
    return toolkit_digest, benchmark_digest


def _manifest(
    configuration: dict[str, Any],
    *,
    seed: int,
    training_split_fingerprint: str,
    evaluation_split_fingerprint: str,
) -> tuple[dict[str, Any], str]:
    toolkit_digest, benchmark_digest = _source_digests()
    manifest = ExperimentManifest.from_environment(
        toolkit_source_digest=toolkit_digest,
        benchmark_source_digest=benchmark_digest,
        seed=seed,
        experiment_config={
            **configuration,
            "training_split_fingerprint": training_split_fingerprint,
            "evaluation_split_fingerprint": evaluation_split_fingerprint,
        },
    )
    manifest.training_budget_steps = configuration["steps"]
    manifest.training_batch_size = configuration["batch_size"]
    manifest_dict = manifest.as_dict()
    digest = hashlib.sha256(
        json.dumps(manifest_dict, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest_dict, digest


def _save_model(model: SequenceLM, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary
    )
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _experiment_lock(records_path: Path):
    lock_path = records_path.with_name(records_path.stem + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"experiment is already locked: {records_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _experiment_identity(
    configuration: dict[str, Any], records_path: Path, checkpoint_dir: Path
) -> dict[str, Any]:
    manifest, _ = _manifest(
        configuration,
        seed=0,
        training_split_fingerprint="experiment",
        evaluation_split_fingerprint="experiment",
    )
    return {
        "configuration": configuration,
        "configuration_sha256": _json_digest(configuration),
        "manifest": manifest,
        "manifest_sha256": _json_digest(manifest),
        "records_path": str(records_path.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
    }


def _open_experiment(
    records_path: Path,
    registry_path: Path,
    checkpoint_dir: Path,
    configuration: dict[str, Any],
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    status_path = records_path.with_name(records_path.stem + ".experiment.json")
    identity = _experiment_identity(configuration, records_path, checkpoint_dir)
    if resume:
        if not status_path.is_file():
            raise ValueError(f"cannot resume unmanaged experiment: {status_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("identity") != identity:
            raise ValueError("experiment source, configuration, environment, or paths differ")
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"experiment checkpoint directory is missing: {checkpoint_dir}")
        return status_path, status
    summary_path = records_path.with_name(records_path.stem + ".summary.json")
    for path in (records_path, registry_path, checkpoint_dir, status_path, summary_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"new experiment requires unused output paths: {path}")
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    status = {"identity": identity, "phases": {}}
    _atomic_json(status_path, status)
    return status_path, status


def _build_training_pool() -> tuple[torch.Tensor, torch.Tensor, list[RetrievalExample]]:
    examples = [
        generate_example(
            TRAINING_LENGTH,
            TRAINING_PAIR_COUNT,
            1,
            split="train",
            sample_index=index,
            target_bucket=POSITION_BUCKETS[index % 3],
        )
        for index in range(TRAIN_POOL_SIZE)
    ]
    inputs = torch.tensor([example.tokens for example in examples], dtype=torch.long)
    targets = torch.tensor([example.target for example in examples], dtype=torch.long)
    return inputs, targets, examples


def train_model(
    method: str,
    candidate: int,
    seed: int,
    steps: int,
    batch_size: int,
    training_inputs: torch.Tensor,
    training_targets: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> SequenceLM:
    if method not in INITIAL_METHODS:
        raise ValueError(f"unsupported initial method: {method}")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    settings = candidate_settings(method, candidate)
    torch.manual_seed(seed)
    training_variant = "rope-full" if method in ("rope-linear", "rope-ntk") else method
    model = SequenceLM(
        VOCABULARY_SIZE,
        training_variant,
        False,
        TRAINING_LENGTH,
        settings["tau0"],
        10000.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    scheduler = AsymptoticLRScheduler(optimizer, floor_ratio=0.1, total_steps=steps)
    generator = torch.Generator().manual_seed(seed + 50000)
    model.train()
    for _ in range(steps):
        indices = torch.randint(len(training_inputs), (batch_size,), generator=generator)
        inputs = training_inputs[indices].to(device)
        targets = training_targets[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_last(inputs)
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite retrieval loss for {method}/{seed}")
        loss.backward()
        optimizer.step()
        scheduler.step()
    if method in ("rope-linear", "rope-ntk"):
        model.variant = method
        for block in model.blocks:
            block.variant = method
            block.rope_scale = settings["rope_scale"]
    return model.eval()


def _evaluation_specs() -> tuple[tuple[int, int], ...]:
    primary = tuple((length, TRAINING_PAIR_COUNT) for length in CONTEXT_LENGTHS)
    distractor_sweep = tuple((4096, count) for count in (8, 32, 64))
    return primary + distractor_sweep


def _append_records(path: Path, records: list[dict[str, Any]]) -> None:
    existing = _read_records(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            for record in [*existing, *records]:
                file.write(json.dumps(record, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_result_file(path: Path, configuration: dict[str, Any]) -> list[dict[str, Any]]:
    records = _read_records(path)
    if any(record.get("configuration") != configuration for record in records):
        raise ValueError(f"existing results use an incompatible configuration: {path}")
    current_source_digests = _source_digests()
    keys = [
        (record["phase"], record.get("method"), record.get("candidate"), record.get("seed"))
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"result file contains duplicate phase/method/candidate/seed rows: {path}")
    for record in records:
        content = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != _json_digest(content):
            raise ValueError("record hash mismatch")
        checkpoint = Path(record["checkpoint_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"recorded checkpoint is missing: {checkpoint}")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["checkpoint_sha256"]:
            raise ValueError(f"recorded checkpoint hash mismatch: {checkpoint}")
        manifest_hash = hashlib.sha256(
            json.dumps(record["manifest"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if manifest_hash != record["manifest_sha256"]:
            raise ValueError("recorded experiment manifest hash mismatch")
        manifest_config = record["manifest"]["experiment_config"]
        if (
            manifest_config.get("checkpoint_sha256") != record["checkpoint_sha256"]
            or manifest_config.get("configuration_sha256") != _json_digest(configuration)
            or any(manifest_config.get(key) != value for key, value in configuration.items())
            or any(
                manifest_config.get(key) != record[key] for key in ("method", "candidate", "phase")
            )
            or record["manifest"]["seed"] != record["seed"]
            or manifest_config.get("candidate_settings")
            != candidate_settings(record["method"], record["candidate"])
            or record["candidate_settings"] != manifest_config.get("candidate_settings")
            or record["training_split_fingerprint"]
            != manifest_config.get("training_split_fingerprint")
        ):
            raise ValueError("checkpoint, configuration, or trial identity does not match manifest")
        recorded_sources = (
            record["manifest"]["toolkit_source_digest"],
            record["manifest"]["benchmark_source_digest"],
        )
        if recorded_sources != current_source_digests:
            raise ValueError("current source fingerprints differ from recorded experiment")
        manifest_base_config = {
            key: value
            for key, value in manifest_config.items()
            if key not in ("training_split_fingerprint", "evaluation_split_fingerprint")
        }
        _, current_manifest_hash = _manifest(
            manifest_base_config,
            seed=record["seed"],
            training_split_fingerprint=manifest_config["training_split_fingerprint"],
            evaluation_split_fingerprint=manifest_config["evaluation_split_fingerprint"],
        )
        if current_manifest_hash != record["manifest_sha256"]:
            raise ValueError("current provenance differs from recorded experiment manifest")
    return records


def _eligible_candidates(
    records: Sequence[dict[str, Any]], methods: Sequence[str]
) -> dict[str, list[int]]:
    eligible: dict[str, list[int]] = {}
    for method in methods:
        eligible[method] = []
        for candidate in CANDIDATES:
            trials = [
                record
                for record in records
                if record["method"] == method and record["candidate"] == candidate
            ]
            if len(trials) != len(SANITY_SEEDS) or {r["seed"] for r in trials} != set(SANITY_SEEDS):
                raise ValueError(f"sanity records incomplete for {method} candidate {candidate}")
            if (
                sum(
                    record["evaluation_metrics"]["256:16"]["model"]["overall"]["accuracy"] >= 0.95
                    for record in trials
                )
                >= 8
            ):
                eligible[method].append(candidate)
    missing = [method for method in methods if not eligible.get(method)]
    if missing:
        raise ValueError(f"training-length retrieval gate failed for methods: {missing}")
    return eligible


def _select_candidates(
    records: Sequence[dict[str, Any]], eligible: dict[str, list[int]]
) -> dict[str, int]:
    selected = {}
    for method, candidates in eligible.items():
        if not candidates:
            continue
        means = {}
        for candidate in candidates:
            trials = [
                record
                for record in records
                if record["method"] == method and record["candidate"] == candidate
            ]
            if len(trials) != len(TUNING_SEEDS) or {r["seed"] for r in trials} != set(TUNING_SEEDS):
                raise ValueError(f"tuning records incomplete for {method} candidate {candidate}")
            means[candidate] = sum(
                record["evaluation_metrics"]["4096:16"]["model"]["overall"]["accuracy"]
                for record in trials
            ) / len(trials)
        selected[method] = max(candidates, key=lambda index: means[index])
    if any(method not in selected for method in COMPARISON_METHODS):
        raise ValueError("candidate selection lacks a required full-attention comparison method")
    return selected


def _record_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": "query-dependent-associative-retrieval",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "train_length": TRAINING_LENGTH,
        "train_pair_count": TRAINING_PAIR_COUNT,
        "training_seed": 1,
        "training_pool_size": TRAIN_POOL_SIZE,
        "context_lengths": list(CONTEXT_LENGTHS),
        "pair_counts": [8, 16, 32, 64],
        "eval_examples_per_seed": args.eval_examples,
        "methods": list(args.methods),
        "candidates": list(CANDIDATES),
        "sanity_seeds": list(SANITY_SEEDS),
        "tuning_seeds": list(TUNING_SEEDS),
        "report_seeds": list(REPORT_SEEDS),
        "device": args.device,
        "precision": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "random_control_seed": 0,
        "eligibility_accuracy": 0.95,
        "eligibility_seeds_required": 8,
        "eligibility_seed_count": 10,
    }


def _record_metrics(
    model: SequenceLM, examples: list[RetrievalExample], eval_batch_size: int
) -> dict[str, Any]:
    return {
        "model": evaluate_model(model, examples, batch_size=eval_batch_size),
        "no_retrieval": evaluate_model(
            model, examples, batch_size=eval_batch_size, ablate_pairs=True
        ),
        "random_answer": evaluate_random_control(examples),
        "evaluation_examples": len(examples),
        "evaluation_denominator": len(examples),
        "example_fingerprint": fingerprint_examples(examples),
    }


def _run_training_phase(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    records_path: Path,
    registry_path: Path,
    checkpoint_dir: Path,
) -> None:
    existing = _validate_result_file(records_path, configuration)
    completed = {
        (record["phase"], record.get("method"), record.get("candidate"), record.get("seed"))
        for record in existing
    }
    registry_records: list[dict[str, Any]] = []
    train_inputs, train_targets, train_examples = _build_training_pool()
    registry_records.extend(example_registry_records("train", train_examples))
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    write_example_registry(registry_path, registry_records)
    training_fingerprint = fingerprint_examples(train_examples)

    if args.phase == "sanity":
        seeds = SANITY_SEEDS
        candidates_by_method = {method: list(CANDIDATES) for method in args.methods}
        eval_specs = ((TRAINING_LENGTH, TRAINING_PAIR_COUNT),)
        split = "sanity"
    else:
        prior_sanity = [record for record in existing if record["phase"] == "sanity"]
        eligible = _eligible_candidates(prior_sanity, COMPARISON_METHODS)
        if args.phase == "tune":
            seeds = TUNING_SEEDS
            candidates_by_method = dict(eligible)
            if "rope-log-window" in args.methods:
                candidates_by_method["rope-log-window"] = [0]
            eval_specs = ((4096, TRAINING_PAIR_COUNT),)
            split = "tune"
        else:
            tuning = [record for record in existing if record["phase"] == "tune"]
            selected = _select_candidates(tuning, eligible)
            seeds = REPORT_SEEDS
            candidates_by_method = {method: [candidate] for method, candidate in selected.items()}
            if "rope-log-window" in args.methods:
                candidates_by_method["rope-log-window"] = [0]
            eval_specs = _evaluation_specs()
            split = "report"

    for seed in seeds:
        first_length, first_pair_count = eval_specs[0]
        first_examples = generate_dataset(
            split, seed, args.eval_examples, first_length, first_pair_count
        )
        first_fingerprint = fingerprint_examples(first_examples)
        for method in args.methods:
            for candidate in candidates_by_method.get(method, []):
                key = (args.phase, method, candidate, seed)
                if key in completed:
                    continue
                checkpoint = checkpoint_dir / f"{args.phase}-{method}-{candidate}-{seed}.pt"
                sidecar = checkpoint.with_suffix(".manifest.json")
                if checkpoint.exists() != sidecar.exists():
                    raise FileExistsError(
                        f"unverifiable orphan checkpoint or manifest: {checkpoint}"
                    )
                if checkpoint.exists():
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                else:
                    model = train_model(
                        method,
                        candidate,
                        seed,
                        args.steps,
                        args.batch_size,
                        train_inputs,
                        train_targets,
                        torch.device(args.device),
                    )
                    checkpoint_hash = _save_model(model, checkpoint)
                    metadata = None
                manifest_config = {
                    **configuration,
                    "configuration_sha256": _json_digest(configuration),
                    "checkpoint_sha256": checkpoint_hash,
                    "method": method,
                    "candidate": candidate,
                    "candidate_settings": candidate_settings(method, candidate),
                    "phase": args.phase,
                }
                manifest_dict, manifest_hash = _manifest(
                    manifest_config,
                    seed=seed,
                    training_split_fingerprint=training_fingerprint,
                    evaluation_split_fingerprint=first_fingerprint,
                )
                expected_metadata = {
                    "checkpoint_sha256": checkpoint_hash,
                    "manifest": manifest_dict,
                    "manifest_sha256": manifest_hash,
                }
                if metadata is None:
                    _atomic_json(sidecar, expected_metadata)
                elif metadata != expected_metadata:
                    raise ValueError(f"checkpoint manifest or provenance mismatch: {checkpoint}")
                else:
                    settings = candidate_settings(method, candidate)
                    model = SequenceLM(
                        VOCABULARY_SIZE, method, False, TRAINING_LENGTH, settings["tau0"], 10000.0
                    ).to(torch.device(args.device))
                    model.load_state_dict(
                        torch.load(checkpoint, map_location="cpu", weights_only=True)
                    )
                    model.eval()
                current_record = {
                    "protocol_version": PROTOCOL_VERSION,
                    "phase": args.phase,
                    "configuration": configuration,
                    "method": method,
                    "candidate": candidate,
                    "candidate_settings": candidate_settings(method, candidate),
                    "seed": seed,
                    "training_length": TRAINING_LENGTH,
                    "training_split_fingerprint": training_fingerprint,
                    "evaluation_split": split,
                    "evaluation_metrics": {},
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "manifest": manifest_dict,
                    "manifest_sha256": manifest_hash,
                    "environment": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform(),
                        "threads": torch.get_num_threads(),
                    },
                }
                for context_length, pair_count in eval_specs:
                    examples = (
                        first_examples
                        if (context_length, pair_count) == eval_specs[0]
                        else generate_dataset(
                            split, seed, args.eval_examples, context_length, pair_count
                        )
                    )
                    write_example_registry(registry_path, example_registry_records(split, examples))
                    current_record["evaluation_metrics"][f"{context_length}:{pair_count}"] = {
                        **_record_metrics(model, examples, args.eval_batch_size),
                        "evaluation_split_fingerprint": fingerprint_examples(examples),
                    }
                current_record["record_sha256"] = _json_digest(current_record)
                _append_records(records_path, [current_record])


def _phase_records(records: Sequence[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    return [record for record in records if record["phase"] == phase]


def _paired_summary(report_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_method = {method: {} for method in INITIAL_METHODS}
    for record in report_records:
        by_method.setdefault(record["method"], {})[record["seed"]] = record
    comparisons: dict[str, Any] = {}
    for target in ("rope-log-full", "rope-log-window"):
        for baseline in ("rope-full", "alibi"):
            cells = {}
            for metric_key in sorted(
                set.intersection(
                    *[
                        set(record["evaluation_metrics"])
                        for method in (target, baseline)
                        for record in by_method.get(method, {}).values()
                    ]
                )
                if by_method.get(target) and by_method.get(baseline)
                else set()
            ):

                def flatten_metrics(metric: dict[str, Any]) -> dict[str, float]:
                    result = {"overall": metric["overall"]["accuracy"]}
                    for group_name in (
                        "by_target_position",
                        "by_target_distance",
                        "by_distractor_count",
                    ):
                        result.update(
                            {
                                f"{group_name}:{group}": value["accuracy"]
                                for group, value in metric[group_name].items()
                            }
                        )
                    return result

                target_metrics = {
                    seed: flatten_metrics(
                        by_method[target][seed]["evaluation_metrics"][metric_key]["model"]
                    )
                    for seed in REPORT_SEEDS
                }
                baseline_metrics = {
                    seed: flatten_metrics(
                        by_method[baseline][seed]["evaluation_metrics"][metric_key]["model"]
                    )
                    for seed in REPORT_SEEDS
                }
                common_groups = set.intersection(
                    *(
                        set(target_metrics[seed]) & set(baseline_metrics[seed])
                        for seed in REPORT_SEEDS
                    )
                )
                for group in sorted(common_groups):
                    differences = [
                        target_metrics[seed][group] - baseline_metrics[seed][group]
                        for seed in REPORT_SEEDS
                    ]
                    lower, upper = bootstrap_interval(differences)
                    cells[f"{metric_key}:{group}"] = {
                        "paired_mean_difference": sum(differences) / len(differences),
                        "paired_95pct_seed_bootstrap_ci": [lower, upper],
                        "target_wins": sum(value > 0 for value in differences),
                        "seed_count": len(differences),
                        "per_seed_differences": dict(zip(REPORT_SEEDS, differences)),
                    }
            comparisons[f"{target}_vs_{baseline}"] = cells
    return {
        "protocol_version": PROTOCOL_VERSION,
        "report_seed_count": len(REPORT_SEEDS),
        "comparisons": comparisons,
        "report_records": [
            {
                "method": row["method"],
                "candidate": row["candidate"],
                "seed": row["seed"],
                "evaluation_metrics": row["evaluation_metrics"],
            }
            for row in report_records
        ],
    }


def _run_smoke(output_path: Path, device: torch.device = torch.device("cpu")) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {output_path}")
    torch.manual_seed(0)
    examples = generate_dataset("smoke", 7, 3, TRAINING_LENGTH, TRAINING_PAIR_COUNT)
    oracle_metrics = evaluate_predictions(examples, [oracle_target(e.tokens) for e in examples])
    if oracle_metrics["overall"]["accuracy"] != 1.0:
        raise RuntimeError("deterministic retrieval oracle failed")
    models = {}
    for method in INITIAL_METHODS:
        settings = candidate_settings(method, 0)
        training_variant = "rope-full" if method in ("rope-linear", "rope-ntk") else method
        model = (
            SequenceLM(
                VOCABULARY_SIZE,
                training_variant,
                False,
                TRAINING_LENGTH,
                settings["tau0"],
                10000.0,
            )
            .to(device)
            .eval()
        )
        with torch.no_grad():
            logits = model.forward_last(
                torch.tensor([examples[0].tokens], dtype=torch.long, device=device)
            )
        if logits.shape != (1, VOCABULARY_SIZE) or not torch.isfinite(logits).all():
            raise RuntimeError(f"smoke forward failed for {method}")
        model_metrics = evaluate_model(model, examples, batch_size=1)
        ablated_metrics = evaluate_model(model, examples, batch_size=1, ablate_pairs=True)
        models[method] = {
            "output_shape": list(logits.shape),
            "finite": True,
            "evaluation_denominator": model_metrics["overall"]["total"],
            "ablated_denominator": ablated_metrics["overall"]["total"],
        }
    result = {
        "smoke_only": True,
        "training_performed": False,
        "research_result": False,
        "protocol_version": PROTOCOL_VERSION,
        "examples": len(examples),
        "context_length": TRAINING_LENGTH,
        "pair_count": TRAINING_PAIR_COUNT,
        "oracle_accuracy": oracle_metrics["overall"]["accuracy"],
        "random_control": evaluate_random_control(examples),
        "methods_forwarded": models,
        "example_fingerprint": fingerprint_examples(examples),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"smoke_only": True, "training_performed": False, "output": str(output_path)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("smoke", "sanity", "tune", "report", "summary"), required=True
    )
    parser.add_argument("--steps", type=positive_int, default=3000)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--eval-examples", type=positive_int, default=300)
    parser.add_argument("--eval-batch-size", type=positive_int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--methods", nargs="+", choices=INITIAL_METHODS, default=list(INITIAL_METHODS)
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--resume", action="store_true", help="Resume an existing managed experiment"
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")
    if args.output is None:
        args.output = (
            Path("benchmarks/context-retrieval-smoke.json")
            if args.phase == "smoke"
            else Path("benchmarks/context-retrieval-v1.jsonl")
        )
    if args.eval_examples % 3:
        parser.error("--eval-examples must be divisible by three for balanced target positions")
    if set(COMPARISON_METHODS) - set(args.methods):
        parser.error(f"--methods must include {', '.join(COMPARISON_METHODS)}")
    torch.set_num_threads(1)
    if args.phase == "smoke":
        if args.resume:
            parser.error("--resume is not applicable to smoke")
        _run_smoke(args.output, torch.device(args.device))
        return

    if args.output.suffix != ".jsonl":
        parser.error("experiment output must have a .jsonl extension")
    if not args.resume and args.phase != "sanity":
        parser.error("a new experiment must start with sanity; use --resume for later phases")
    configuration = _record_configuration(args)
    records_path = args.output.resolve()
    registry_path = records_path.with_name(records_path.stem + ".examples.jsonl")
    checkpoint_dir = (
        args.checkpoint_dir or records_path.with_name(records_path.stem + "-checkpoints")
    ).resolve()

    with _experiment_lock(records_path):
        status_path, status = _open_experiment(
            records_path, registry_path, checkpoint_dir, configuration, args.resume
        )
        status["phases"][args.phase] = "running"
        _atomic_json(status_path, status)
        try:
            if args.phase in ("sanity", "tune", "report"):
                _run_training_phase(
                    args, configuration, records_path, registry_path, checkpoint_dir
                )
            else:
                records = _validate_result_file(records_path, configuration)
                report_records = _phase_records(records, "report")
                expected_methods = set(configuration["methods"])
                if (
                    len(report_records) != len(expected_methods) * len(REPORT_SEEDS)
                    or {record["method"] for record in report_records} != expected_methods
                ):
                    raise ValueError(
                        "summary requires complete reporting records for every initial method and seed"
                    )
                summary_path = records_path.with_name(records_path.stem + ".summary.json")
                if summary_path.exists():
                    raise FileExistsError(f"refusing to overwrite summary: {summary_path}")
                _atomic_json(summary_path, _paired_summary(report_records))
                print(f"Wrote {summary_path}")
        except BaseException:
            status["phases"][args.phase] = "incomplete"
            _atomic_json(status_path, status)
            raise
        status["phases"][args.phase] = "completed"
        _atomic_json(status_path, status)


if __name__ == "__main__":
    main()
