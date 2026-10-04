"""GGUF eval mode: answers from the exported quants on llama-server (faked: no server, no network)."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from distillkit import evaluate, gguf_eval
from distillkit.config import load_config

ROOT = Path(__file__).parent.parent


class FakeTokenizer:
    def apply_chat_template(self, messages, tools=None, **kw):
        return f"<sys>{messages[0]['content']}<tools>{len(tools or [])}<user>{messages[1]['content']}<assistant>"


class FakeJudge:
    """Prefers an answer "s" (the student, or a quant that kept the student's answer)."""

    def __init__(self, cfg, name="judge"):
        self.cfg = cfg

    def chat_json(self, system, prompt, model, temperature=None):
        from distillkit.schemas import JudgeVerdict

        a = prompt.split("Answer A:\n", 1)[1].split("\n", 1)[0]
        v = JudgeVerdict(reasoning="r", verdict="A" if a == "s" else "B")
        return v, v.model_dump_json()

    def map(self, fn, items):
        return [fn(i) for i in items]


@pytest.fixture
def fake_server(monkeypatch):
    """llama-server replaced by a function of the prompt; records the prompts it got."""
    prompts = []

    @contextmanager
    def server(cmd_for_port, log, timeout=600):
        yield "http://fake"

    def complete(url, prompt, max_tokens):
        prompts.append(prompt)
        return "s"

    monkeypatch.setattr(gguf_eval, "_tokenizer", lambda model: FakeTokenizer())
    monkeypatch.setattr(gguf_eval, "llama_server", server)
    monkeypatch.setattr(gguf_eval, "complete", complete)
    return prompts


def _cfg(tmp_path, *overrides):
    cfg = load_config(ROOT / "configs/tools_pilot.yaml", [
        f"run_dir={tmp_path / 'run'}", f"eval.file={ROOT / 'data/eval/eval_all.jsonl'}",
        f"eval.tool_file={ROOT / 'data/eval/eval_tools.jsonl'}", *overrides,
    ])
    (cfg.run_dir / "export").mkdir(parents=True, exist_ok=True)
    return cfg


def test_server_cmd_is_cpu_only_and_local(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "llama-server").touch()
    cmd = gguf_eval.server_cmd(tmp_path, Path("m.gguf"), 8123, 8192, 6, 0)
    assert cmd[0] == str(tmp_path / "bin" / "llama-server")
    args = dict(zip(cmd[1::2], cmd[2::2]))
    assert args["--host"] == "127.0.0.1" and args["--port"] == "8123"
    assert (args["-c"], args["-t"], args["-ngl"], args["-np"]) == ("8192", "6", "0", "1")


def test_comparisons():
    assert evaluate.comparisons(["base"]) == []
    assert evaluate.comparisons(["base", "student"]) == [("student", "base")]
    # no student (e.g. a base-model export): quantization loss is measured against base
    assert evaluate.comparisons(["base", "gguf_Q4_K_M"]) == [("gguf_Q4_K_M", "base")]
    assert evaluate.comparisons(["base", "student", "gguf_Q4_K_M"]) == [
        ("student", "base"), ("gguf_Q4_K_M", "student"), ("gguf_Q4_K_M", "base")]


def test_gguf_answers_use_the_eval_prompt_and_are_cached(tmp_path, monkeypatch, fake_server):
    monkeypatch.setattr(evaluate, "generate_answers", lambda cfg, q, tools=None: {"base": ["b"] * len(q)})
    cfg = _cfg(tmp_path, "eval.gguf=[Q4_K_M]")
    with pytest.raises(SystemExit, match="model-Q4_K_M.gguf not found"):
        evaluate.run_answers(cfg)
    gguf = cfg.run_dir / "export" / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"x")

    answers = evaluate.run_answers(cfg)
    n_prose, n_tools = sum(1 for _ in open(cfg.eval.file)), sum(1 for _ in open(cfg.eval.tool_file))
    assert len(answers["gguf_Q4_K_M"]) == len(answers["base"]) == n_prose + n_tools
    # same rendered prompt as the HF eval: system prompt, tool schemas on the tool slice only
    assert fake_server[0].startswith(f"<sys>{cfg.student.system_prompt}<tools>0<user>")
    assert fake_server[-1].startswith(f"<sys>{cfg.student.system_prompt}<tools>8<user>")

    evaluate.run_answers(cfg)
    assert len(fake_server) == n_prose + n_tools  # reused
    os.utime(gguf, (1, 1))  # re-exported
    evaluate.run_answers(cfg)
    assert len(fake_server) == 2 * (n_prose + n_tools)


def test_gguf_judged_against_its_source_and_base(tmp_path, monkeypatch, fake_server):
    monkeypatch.setattr(evaluate, "generate_answers", lambda cfg, q, tools=None: {"base": ["b"] * len(q), "student": ["s"] * len(q)})
    monkeypatch.setattr(evaluate, "Teacher", FakeJudge)
    cfg = _cfg(tmp_path, "eval.gguf=[Q4_K_M]")
    (cfg.run_dir / "export" / "model-Q4_K_M.gguf").write_bytes(b"x")
    s = evaluate.run(cfg)
    assert s["student_vs_base_win_rate"] == 1.0
    assert s["gguf_Q4_K_M_vs_student_win_rate"] == 0.5  # same answers as the student: no quantization loss
    assert s["gguf_Q4_K_M_vs_base_win_rate"] == 1.0
    assert {"check_pass_rate", "tool_decision_acc"} <= set(s["gguf_Q4_K_M"])
    row = json.loads((cfg.run_dir / "eval_outputs.jsonl").read_text().splitlines()[0])
    assert set(row["judge"]) == {"student_vs_base", "gguf_Q4_K_M_vs_student", "gguf_Q4_K_M_vs_base"}
