"""Production generation + real Config/IO, fake teacher; no inference or command execution."""
import copy
import hashlib
import json

import pytest

from distillkit import generate
from distillkit.io import JsonlAppender, read_jsonl, write_jsonl
from distillkit.teacher import Completion
from test_scenario_plan import planned, write_plan
from test_source_registry import setup  # dependency of the real Config/registry fixture


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def teacher_fixture(monkeypatch, questions):
    """Only the model boundary is replaced; production prompts, schemas and appenders run."""
    questions = iter(questions)
    requests = []
    raw_responses = []

    class Teacher:
        def __init__(self, cfg):
            self.model = cfg.model

        def map(self, fn, items):
            return [fn(item) for item in items]

        def chat_json(self, system, user, schema):
            requests.append(("question", [system, user], {}))
            raw = json.dumps({"question": next(questions)}, ensure_ascii=False, indent=2)
            raw_responses.append(raw)
            return schema.model_validate_json(raw), raw

        def complete(self, messages, **kwargs):
            requests.append(("answer", copy.deepcopy(messages), kwargs))
            # Explicitly synthetic token data tests the existing writer only.
            lp = {"tokens": ["token_id:1"], "logprobs": [-0.1], "top": [[["token_id:1", -0.1]]]}
            return Completion("Synthetic fixture response; not model inference.",
                              logprobs=lp if kwargs.get("top_logprobs") else None)

    monkeypatch.setattr(generate, "Teacher", Teacher)
    return requests, raw_responses


@pytest.mark.parametrize("task", ["concept", "howto", "debug", "script"])
def test_scoped_prose_rules_reach_actual_answer_request(planned, monkeypatch, task):
    cfg, _, entry = planned
    entry["task"] = task
    write_plan(cfg, [entry])
    cfg.teacher.top_logprobs = 1
    requests, _ = teacher_fixture(monkeypatch, ["What does the synthetic scanner count establish?"])
    generate.run(cfg)
    answers = [r for r in requests if r[0] == "answer"]
    assert len(answers) == 1
    _, messages, kwargs = answers[0]
    assert generate.PROMPT_VERSION in messages[0]["content"]
    prompt = messages[1]["content"]
    for rule in ("Answer only the user's requested scope", "not an unsolicited full program",
                 "API constants", "Preserve a supplied local-computation placeholder",
                 "Preserve the source's qualifications", "Code-fence language labels must match",
                 "not as shell assignments"):
        assert rule in prompt
    assert "The synthetic scanner reports a count." in prompt
    assert entry["brief"] not in prompt  # scenario is question-planning input, not an answer key
    assert kwargs["top_logprobs"] == 1
    rows = read_jsonl(cfg.run_dir / "generated.jsonl")
    assert len(rows) == 1 and rows[0]["prompt_version"] == generate.PROMPT_VERSION
    assert rows[0]["messages"] == [
        {"role": "user", "content": "What does the synthetic scanner count establish?"},
        {"role": "assistant", "content": "Synthetic fixture response; not model inference."}]
    assert len(read_jsonl(cfg.run_dir / "teacher_logprobs.jsonl.gz")) == 1
    assert not (cfg.run_dir / "question_rejections.jsonl").exists()


@pytest.mark.parametrize("mode", ["prose", "call", "ask", "none"])
def test_question_context_rules_reach_every_mode_without_changing_tool_answer_rules(planned, monkeypatch, mode):
    cfg, _, entry = planned
    entry["mode"] = mode
    write_plan(cfg, [entry])
    question = "Please check my job's current state." if mode == "ask" else "What does the synthetic scanner count establish?"
    requests, _ = teacher_fixture(monkeypatch, [question])
    generate.run(cfg)
    prompt = requests[0][1][1]
    assert "student's entire user context" in prompt
    assert "Keep required code context minimal but complete" in prompt
    assert "close every code fence" in prompt
    assert "ask a smaller question whose necessary context fits" in prompt
    assert "preserve the intentionally missing required user input" in prompt
    row = read_jsonl(cfg.run_dir / "generated.jsonl")[0]
    assert row["question"] == question and row["mode"] == mode
    if mode != "prose":
        assert requests[1][2]["tools"] == generate.SCHEMAS
        assert "Answer only the user's requested scope" not in requests[1][1][0]["content"]
        assert "Never invent job ids or other arguments" in requests[1][1][0]["content"]
    if mode == "ask":
        # An intentionally missing identifier remains a legitimate clarification input.
        assert "job_id" not in question and not (cfg.run_dir / "question_rejections.jsonl").exists()


@pytest.mark.parametrize("question", [
    "Explain the difference between `local` and `global` indices.",
    "Explain the inline example ``value `with` backticks`` without running it.",
    "Explain this complete C fragment:\n```c\nint value = 3;\n```",
    "Explain this complete shell fragment:\n```bash\nprintf '%s\\n' hello\n```",
    'Explain this C string literal:\n```c\nprintf("```");\n```',
    "Explain this shell string literal:\n```bash\nprintf '%s\\n' '```'\n```",
    "Explain this C comment:\n```c\n// A literal ``` marker is part of the comment.\nint value = 3;\n```",
    "Explain this prose-joined opening:```c\nint value = 3;\n```",
    "Explain this literal marker inside a larger block:\n````text\n```\n````",
    "Explain this complete alternate fence:\n~~~text\nexample `value`\n~~~",
    "Explain the escaped literal marker \\``` without treating it as code.",
])
def test_inline_backticks_and_complete_fences_pass_real_generation(planned, monkeypatch, question):
    cfg, _, _ = planned
    requests, _ = teacher_fixture(monkeypatch, [question])
    generate.run(cfg)
    assert [r[0] for r in requests] == ["question", "answer"]
    assert read_jsonl(cfg.run_dir / "generated.jsonl")[0]["question"] == question
    assert not (cfg.run_dir / "question_rejections.jsonl").exists()


@pytest.mark.parametrize("question", [
    "Explain this truncated code:```c#include<stdio.h>void f(){ printf(",
    "Explain this unfinished block:\n```c\nint value = 3;\n~~~",
    "Explain this too-short closing marker:\n````c\nint value = 3;\n```",
    "Explain this unfinished tilde block:\n~~~text\nvalue",
    'Explain this unclosed C block:\n```c\nprintf("```");',
    "Explain this unclosed block ending in a comment:\n```c\nint value = 3;\n// ```",
    "Explain this joined opening ending in a comment:```c\nint value = 3;\n// ```",
    'Explain this joined opening ending in an unfinished string:```c\nprintf("```',
    "Explain this non-standalone closing marker:```c\nint value = 3;```",
])
def test_valid_json_unmatched_fence_is_preserved_and_never_answered(planned, monkeypatch, capsys, question):
    cfg, _, _ = planned
    cfg.teacher.top_logprobs = 1
    original_question = "  " + question + "\n "
    requests, raw = teacher_fixture(monkeypatch, [original_question])
    generate.run(cfg)
    assert [r[0] for r in requests] == ["question"]
    assert not (cfg.run_dir / "generated.jsonl").exists()
    assert not (cfg.run_dir / "teacher_logprobs.jsonl.gz").exists()
    candidate = read_jsonl(cfg.run_dir / "questions.jsonl")[0]
    check = candidate["question_validation"]
    assert candidate["question"] == question
    assert check["status"] == "rejected" and check["reason"] == "unmatched_question_code_fence"
    assert check["raw_question"] == original_question
    assert check["raw_question_sha256"] == hashlib.sha256(original_question.encode()).hexdigest()
    assert check["raw_response"] == raw[0]
    assert check["raw_response_sha256"] == hashlib.sha256(raw[0].encode()).hexdigest()
    rejection = read_jsonl(cfg.run_dir / "question_rejections.jsonl")[0]
    assert rejection["candidate"] == candidate
    assert rejection["candidate_sha256"] == digest(candidate)
    assert rejection["id"] == candidate["id"]
    assert rejection["check_version"] == generate.QUESTION_CHECK_VERSION
    log = capsys.readouterr().out
    assert "1 rejected-question candidates" in log and "invalid JSON" not in log
    before = {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
    generate.run(cfg)
    assert [r[0] for r in requests] == ["question"]  # no silent regenerate or answer
    assert before == {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}


def test_interruption_after_question_capture_recovers_audit_without_new_request(planned, monkeypatch):
    cfg, _, _ = planned
    question = "Explain the incomplete fragment:```c\nvalue ="
    requests, raw = teacher_fixture(monkeypatch, [question])

    class InterruptedAppender(JsonlAppender):
        def append(self, row):
            if self.path.name == "question_rejections.jsonl":
                raise RuntimeError("synthetic interruption before rejection audit append")
            super().append(row)

    monkeypatch.setattr(generate, "JsonlAppender", InterruptedAppender)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        generate.run(cfg)
    before = (cfg.run_dir / "questions.jsonl").read_bytes()
    monkeypatch.setattr(generate, "JsonlAppender", JsonlAppender)
    generate.run(cfg)
    assert [r[0] for r in requests] == ["question"]
    assert before == (cfg.run_dir / "questions.jsonl").read_bytes()
    assert read_jsonl(cfg.run_dir / "question_rejections.jsonl")[0]["candidate"]["question_validation"]["raw_response"] == raw[0]


@pytest.mark.parametrize("annotation", [None, {"status": "accepted"}])
def test_cached_legacy_question_cannot_bypass_screen_with_missing_or_false_annotation(setup, monkeypatch, annotation):
    cfg, _ = setup
    cfg.seeds.registry = None  # compatibility path: legacy grid, no frozen scenario manifest
    job = generate.question_jobs(cfg)[0]
    question = {**job, "question": "Explain the incomplete snippet:```c\nvalue =",
                "scenario": {}, "prompt_version": "legacy-fixture", "mock_version": generate.MOCK_VERSION}
    if annotation is not None:
        question["question_validation"] = annotation
    write_jsonl(cfg.run_dir / "questions.jsonl", [question])
    before = (cfg.run_dir / "questions.jsonl").read_bytes()
    requests, _ = teacher_fixture(monkeypatch, [])
    generate.run(cfg)
    assert requests == [] and not (cfg.run_dir / "generated.jsonl").exists()
    assert before == (cfg.run_dir / "questions.jsonl").read_bytes()
    rejection = read_jsonl(cfg.run_dir / "question_rejections.jsonl")[0]
    assert rejection["reason"] == "unmatched_question_code_fence"
    assert rejection["candidate"] == question  # no historical question/annotation repair
    assert "raw_response" not in rejection["candidate"].get("question_validation", {})  # unavailable, never fabricated


def test_mixed_plan_counts_rejected_candidate_separately_from_missing_json(planned, monkeypatch, capsys):
    cfg, _, entry = planned
    other = {**entry, "scenario_id": "separate-procedure", "task": "howto",
             "brief": "Ask for the next observation needed to interpret the synthetic scanner count."}
    write_plan(cfg, [entry, other])
    cfg.generate.answers_per_question = 2
    requests, _ = teacher_fixture(monkeypatch, ["Explain the incomplete fragment:```c\nvalue =",
                                               "What observation would establish a cause?"])
    generate.run(cfg)
    assert len(read_jsonl(cfg.run_dir / "questions.jsonl")) == 2
    assert len(read_jsonl(cfg.run_dir / "question_rejections.jsonl")) == 1
    rows = read_jsonl(cfg.run_dir / "generated.jsonl")
    assert len(rows) == 2 and {r["sample"] for r in rows} == {0, 1}
    assert all(r["scenario_id"] == "separate-procedure" for r in rows)
    manifest = json.loads((cfg.run_dir / "generation_manifest.json").read_text())
    assert manifest["inputs"]["planned_questions"] == 2 and manifest["inputs"]["planned_answers"] == 4
    assert [r[0] for r in requests] == ["question", "question", "answer", "answer"]
    assert "invalid JSON" not in capsys.readouterr().out


def test_orphan_rejection_artifact_requires_fresh_manifest_before_client(planned, monkeypatch):
    cfg, _, _ = planned
    write_jsonl(cfg.run_dir / "question_rejections.jsonl", [{"id": "previous-attempt"}])
    monkeypatch.setattr(generate, "Teacher", lambda *_: pytest.fail("client constructed before stale-artifact rejection"))
    with pytest.raises(ValueError, match="artifacts exist without a matching manifest"):
        generate.run(cfg)
    assert not (cfg.run_dir / "generation_manifest.json").exists()


def test_check_version_change_requires_fresh_run_before_client(planned, monkeypatch):
    cfg, _, _ = planned
    teacher_fixture(monkeypatch, ["What does the synthetic scanner count establish?"])
    generate.run(cfg)
    before = {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
    monkeypatch.setattr(generate, "QUESTION_CHECK_VERSION", "synthetic-next-check-version")
    monkeypatch.setattr(generate, "Teacher", lambda *_: pytest.fail("client constructed before manifest rejection"))
    with pytest.raises(ValueError, match="generation manifest differs"):
        generate.run(cfg)
    assert before == {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
