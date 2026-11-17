import importlib.util
import json
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "research/independent-retrieval-v1/runner.py"
spec = importlib.util.spec_from_file_location("independent_retrieval_runner", RUNNER_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("length", [20, 50, 100, 250, 500])
def test_generator_lengths_uniqueness_and_seed_independence(length):
    turns = runner.generate_conversation(length, 123, "validation")
    assert len(turns) == length
    assert turns == runner.generate_conversation(length, 123, "validation")
    assert turns != runner.generate_conversation(length, 123, "report")
    facts = [turn for turn in turns if turn["kind"] == "fact"]
    queries = [turn for turn in turns if turn["kind"] == "query"]
    assert len({turn["key"] for turn in queries}) == 7
    assert len({turn["key"] for turn in facts}) == len(facts) - 1
    assert len({turn["value"] for turn in facts}) == len(facts)
    assert len(queries) == 7
    assert sum(runner.oracle(turns, turns.index(query)) == runner.MISSING for query in queries) == 1


def test_counterfactual_lookup_and_no_query_answer_leakage():
    turns = runner.generate_conversation(50, 17, "validation")
    queries = [index for index, turn in enumerate(turns) if turn["kind"] == "query"]
    answers = [runner.oracle(turns, index) for index in queries]
    present = [answer for answer in answers if answer != runner.MISSING]
    assert len(set(present)) == 6
    assert all(
        runner.oracle(turns, index) not in runner.serialize(turns[index])
        for index in queries
        if runner.oracle(turns, index) != runner.MISSING
    )
    index = queries[0]
    other_index = next(other for other in queries if answers[queries.index(other)] != answers[0])
    counterfactual = [dict(turn) for turn in turns]
    counterfactual[index]["key"] = turns[other_index]["key"]
    assert runner.oracle(counterfactual, index) == answers[queries.index(other_index)]


def test_oracle_resolves_updates_and_shuffled_facts():
    turns = runner.generate_conversation(20, 31, "validation")
    facts = [turn for turn in turns if turn["kind"] == "fact"]
    update = facts[-1]
    query = {"kind": "query", "key": update["key"]}
    assert runner.oracle([*facts, query], len(facts)) == update["value"]
    first = facts[0]
    query = {"kind": "query", "key": first["key"]}
    reversed_facts = list(reversed(facts[:-1]))
    assert runner.oracle([*reversed_facts, query], len(reversed_facts)) == first["value"]


def test_controls_and_denominators_are_paired_and_no_oracle_as_prediction():
    turns = runner.generate_conversation(20, 54, "validation")
    rows = runner.evaluate(turns, 54)
    metrics = runner.summarize(rows)
    assert metrics["full_history_symbolic"]["correct"] == 7
    assert metrics["full_history_symbolic"]["total"] == 7
    assert metrics["full_history_symbolic"]["positive_total"] == 6
    assert metrics["full_history_symbolic"]["negative_total"] == 1
    assert all(row["system"] != "oracle" for row in rows)
    differences = runner.paired_differences(rows, "fixed_window_symbolic", "full_history_symbolic")
    assert len(differences) == 7
    assert sum(differences) < 0
    with pytest.raises(ValueError, match="same nonempty query set"):
        runner.paired_differences(rows[:-1], "fixed_window_symbolic", rows[-1]["system"])


def test_validation_records_incomplete_system_and_refuses_overwrite(tmp_path):
    output = tmp_path / "run"
    runner.run_validation(output)
    result = json.loads((output / "RESULTS.json").read_text())
    assert result["status"] == "completed_validation_only"
    assert result["scientific_comparison"] is False
    assert result["queries"] == 14
    assert result["hypotheses"] == dict.fromkeys(("H1", "H2", "H3"), "inconclusive")
    assert json.loads((output / "status.json").read_text())["status"] == "completed_validation_only"
    before = (output / "RESULTS.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_validation(output)
    assert (output / "RESULTS.json").read_bytes() == before
