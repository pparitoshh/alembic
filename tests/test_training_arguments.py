"""Real Arrow roundtrip regression; no teacher, model or cluster calls."""
import copy
import json

import pytest

from distillkit.records import training_example


def trace(calls):
    messages = [{"role": "user", "content": "Synthetic diagnostic request."}]
    for name, arguments in calls:
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": arguments}}
            ]},
            {"role": "tool", "content": "{}"},
        ])
    messages.append({"role": "assistant", "content": "Synthetic final answer."})
    return {"messages": messages}


def arguments(example):
    return [call["function"]["arguments"] for message in example["messages"]
            for call in message.get("tool_calls") or []]


def test_training_adapter_preserves_original_and_already_serialized_arguments():
    serialized = '{"job_id": "42"}'
    row = trace([("job_status", serialized), ("read_job_log", {
        "job_id": "42", "stream": "stdout", "tail_lines": 20,
    })])
    before = copy.deepcopy(row)
    adapted = training_example(row, "SYS")
    assert row == before
    assert arguments(adapted)[0] == serialized
    assert json.loads(arguments(adapted)[1]) == before["messages"][3]["tool_calls"][0]["function"]["arguments"]
    adapted["messages"][-1]["content"] = "changed copy"
    assert row == before


def test_different_tools_do_not_gain_arguments_across_arrow_turns_and_rows():
    datasets = pytest.importorskip("datasets")
    rows = [
        trace([("job_status", {"job_id": "42"}),
               ("read_job_log", {"job_id": "42", "stream": "stdout", "tail_lines": 20})]),
        trace([("gpu_availability", {})]),
        {"question": "Synthetic prose?", "answer": "Synthetic prose."},
    ]
    adapted = [training_example(row, "SYS") for row in rows]
    roundtrip = datasets.Dataset.from_list(adapted).to_list()
    assert [arguments(row) for row in roundtrip] == [arguments(row) for row in adapted]
    assert json.loads(arguments(roundtrip[0])[0]) == {"job_id": "42"}
    assert json.loads(arguments(roundtrip[1])[0]) == {}
    assert arguments(roundtrip[2]) == []


def test_old_dict_representation_demonstrates_arrow_field_pollution():
    datasets = pytest.importorskip("datasets")
    row = trace([("job_status", {"job_id": "42"}),
                 ("read_job_log", {"job_id": "42", "stream": "stdout", "tail_lines": 20})])
    old = datasets.Dataset.from_list([row])[0]
    assert arguments(old)[0] == {"job_id": "42", "stream": None, "tail_lines": None}
    corrected = datasets.Dataset.from_list([training_example(row, "SYS")])[0]
    assert json.loads(arguments(corrected)[0]) == {"job_id": "42"}


def test_nested_and_heterogeneous_argument_values_survive_arrow():
    datasets = pytest.importorskip("datasets")
    values = [
        {"payload": {"a": [1, "x", None], "label": "δ"}},
        {"payload": "different type", "enabled": False},
    ]
    examples = [training_example(trace([("synthetic", value)]), "SYS") for value in values]
    assert [json.loads(arguments(row)[0]) for row in datasets.Dataset.from_list(examples)] == values


def test_nonfinite_argument_is_not_serialized_as_invalid_json():
    with pytest.raises(ValueError):
        training_example(trace([("synthetic", {"value": float("nan")})]), "SYS")
