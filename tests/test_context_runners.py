import json
import sys

import pytest

from benchmarks import context_char, context_char_tuned, context_needle_tuned, context_successor


def test_character_control_runner_resumes_completed_seed(tmp_path, monkeypatch):
    output = tmp_path / "characters.jsonl"
    monkeypatch.setattr(context_char, "dataset_info", lambda path: ("abc" * 8000, "test"))
    calls = []

    class Model:
        def set_attention_evaluation(self, mode):
            calls.append(mode)

    monkeypatch.setattr(context_char, "train_char", lambda *args: Model())
    monkeypatch.setattr(context_char, "evaluate_char_contexts", lambda *args: {"result": 1})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_char",
            "--steps",
            "1",
            "--variants",
            "rope-full",
            "--seeds",
            "42",
            "--lengths",
            "256",
            "--output",
            str(output),
        ],
    )
    context_char.main()
    context_char.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["metrics"] == {"full": {"result": 1}, "window": {"result": 1}}
    assert calls == ["full", "window"]


def test_successor_runner_resumes_completed_candidate(tmp_path, monkeypatch):
    output = tmp_path / "successor.jsonl"
    calls = []
    monkeypatch.setattr(context_successor, "train_successor", lambda *args: calls.append(args))
    monkeypatch.setattr(context_successor, "evaluate_successor", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_successor",
            "--tuning",
            "--steps",
            "1",
            "--methods",
            "rope-full",
            "--seeds",
            "100",
            "--candidates",
            "0",
            "--output",
            str(output),
        ],
    )
    context_successor.main()
    context_successor.main()
    assert len(output.read_text().splitlines()) == 1
    assert len(calls) == 1


def test_successor_runner_keeps_original_defaults_and_resumes_with_candidate_two(
    tmp_path, monkeypatch
):
    output = tmp_path / "successor.jsonl"
    monkeypatch.setattr(context_successor, "train_successor", lambda *args: None)
    monkeypatch.setattr(context_successor, "evaluate_successor", lambda *args, **kwargs: {})
    args = [
        "context_successor",
        "--tuning",
        "--steps",
        "1",
        "--methods",
        "alibi",
        "--seeds",
        "100",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", args)
    context_successor.main()
    assert [json.loads(line)["candidate"] for line in output.read_text().splitlines()] == [0, 1]

    monkeypatch.setattr(sys, "argv", [*args, "--candidates", "0", "1", "2"])
    context_successor.main()
    context_successor.main()
    assert [json.loads(line)["candidate"] for line in output.read_text().splitlines()] == [0, 1, 2]


@pytest.mark.parametrize("tuning", [True, False])
def test_successor_runner_rejects_candidate_three_for_unsupported_methods(
    tmp_path, monkeypatch, tuning
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_successor",
            *(["--tuning"] if tuning else []),
            "--methods",
            "alibi",
            "--candidates",
            "3",
            "--output",
            str(tmp_path / "successor.jsonl"),
        ],
    )
    with pytest.raises(SystemExit):
        context_successor.main()


def test_successor_runner_records_candidate_three(tmp_path, monkeypatch):
    output = tmp_path / "successor.jsonl"
    monkeypatch.setattr(context_successor, "train_successor", lambda *args: None)
    monkeypatch.setattr(context_successor, "evaluate_successor", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_successor",
            "--tuning",
            "--steps",
            "1",
            "--methods",
            "rope-log-window",
            "--seeds",
            "100",
            "--candidates",
            "3",
            "--output",
            str(output),
        ],
    )
    context_successor.main()
    record = json.loads(output.read_text())
    assert record["candidate"] == 3
    assert record["settings"]["tau0"] == 2.0


def test_character_tuning_runner_resumes_completed_candidate(tmp_path, monkeypatch):
    output = tmp_path / "char-tuning.jsonl"
    monkeypatch.setattr(context_char_tuned, "dataset_info", lambda path: ("abc" * 1000, "test"))
    calls = []
    monkeypatch.setattr(context_char_tuned, "train_char", lambda *args: calls.append(args))
    monkeypatch.setattr(context_char_tuned, "context_benefit", lambda *args: {"1024": 2.0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_char_tuned",
            "--tuning",
            "--steps",
            "1",
            "--methods",
            "rope-full",
            "--seeds",
            "100",
            "--candidates",
            "0",
            "--output",
            str(output),
        ],
    )
    context_char_tuned.main()
    context_char_tuned.main()
    assert len(output.read_text().splitlines()) == 1
    assert len(calls) == 1


def test_needle_runner_resumes_only_after_sanity_gate(tmp_path, monkeypatch):
    output = tmp_path / "needle.jsonl"
    sanity = tmp_path / "sanity.jsonl"
    sanity.write_text(
        "\n".join(
            json.dumps(
                {
                    "seed": seed,
                    "variant": "rope-full",
                    "train_length": 256,
                    "steps": 3000,
                    "batch_size": 16,
                    "eval_queries": 200,
                    "correct": 180,
                    "total": 200,
                }
            )
            for seed in range(42, 52)
        )
    )
    calls = []
    monkeypatch.setattr(context_needle_tuned, "train_needle", lambda *args: calls.append(args))
    monkeypatch.setattr(context_needle_tuned, "evaluate_needle_sanity", lambda *args: (140, 200))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_needle_tuned",
            "--tuning",
            "--steps",
            "1",
            "--methods",
            "rope-full",
            "--seeds",
            "100",
            "--candidates",
            "0",
            "--sanity-records",
            str(sanity),
            "--output",
            str(output),
        ],
    )
    context_needle_tuned.main()
    context_needle_tuned.main()
    assert len(output.read_text().splitlines()) == 1
    assert len(calls) == 1
