import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from distillkit import verify
from distillkit.config import Config, TrainCfg, load_config
from distillkit.generate import question_jobs
from distillkit.io import JsonlAppender, read_jsonl, write_jsonl
from distillkit.records import final_answer, messages, prose_row, training_example
from distillkit.teacher import _compact_logprobs

ROOT = Path(__file__).parent.parent
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_validates(path):
    assert load_config(path).run_dir.is_dir()  # created under the test's temp dir (conftest chdir)


def _cfg_dict(tmp_path) -> dict:
    import yaml

    d = yaml.safe_load((ROOT / "configs/toy_qdora.yaml").read_text())
    d["run_dir"] = str(tmp_path / "run")
    d["seeds"]["dir"] = str(ROOT / "data/seeds")
    d["eval"]["file"] = str(ROOT / "data/eval/eval_all.jsonl")
    return d


def test_unknown_key_is_rejected(tmp_path):
    d = _cfg_dict(tmp_path)
    d["train"]["learning_rat"] = 1e-4
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_precision_defaults():
    assert TrainCfg().bf16 and TrainCfg().dtype == "bfloat16"
    assert TrainCfg(fp16=True).bf16 is False and TrainCfg(fp16=True).dtype == "float16"
    with pytest.raises(ValidationError):
        TrainCfg(fp16=True, bf16=True)


def test_legacy_row_and_tool_trace():
    legacy = {"question": "q?", "answer": "a."}
    assert messages(legacy)[-1] == {"role": "assistant", "content": "a."}
    assert final_answer(legacy) == "a."

    trace = {
        "question": "Is job 42 running?",
        "tools": [{"name": "job_status", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": "Is job 42 running?"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "job_status", "arguments": {"job_id": "42"}}}]},
            {"role": "tool", "content": '{"state": "RUNNING"}'},
            {"role": "assistant", "content": "Yes, job 42 is running."},
        ],
    }
    assert final_answer(trace) == "Yes, job 42 is running."
    ex = training_example(trace, "SYS")
    assert ex["messages"][0] == {"role": "system", "content": "SYS"}
    assert json.loads(ex["tools"])[0] == {"type": "function", "function": trace["tools"][0]}  # what vLLM/Ollama send
    assert training_example(legacy, "SYS")["tools"] is None


@pytest.mark.parametrize("name", ["rows.jsonl", "rows.jsonl.gz"])
def test_appender_resumes_after_truncated_write(tmp_path, name):
    path = tmp_path / name
    out = JsonlAppender(path)
    out.append({"id": 1})
    out.append({"id": 2})
    # simulate a job killed mid-write: a partial last line
    if name.endswith(".gz"):
        with gzip.open(path, "at") as f:
            f.write('{"id": 3, "tex')
    else:
        with open(path, "a") as f:
            f.write('{"id": 3, "tex')
    assert [r["id"] for r in out.existing()] == [1, 2]
    out.append({"id": 3})
    assert [r["id"] for r in read_jsonl(path)] == [1, 2, 3]


def test_compact_logprobs():
    tok = lambda t, lp, top=(): SimpleNamespace(token=t, logprob=lp, top_logprobs=[SimpleNamespace(token=a, logprob=b) for a, b in top])
    lp = SimpleNamespace(content=[tok("token_id:1", -0.1, [("token_id:1", -0.1), ("token_id:7", -2.3)]), tok("token_id:2", -0.5)])
    assert _compact_logprobs(lp) == {
        "tokens": ["token_id:1", "token_id:2"],
        "logprobs": [-0.1, -0.5],
        "top": [[["token_id:1", -0.1], ["token_id:7", -2.3]], []],
    }
    assert _compact_logprobs(None) is None


def test_question_jobs_are_stable_and_unique(tmp_path):
    cfg = Config.model_validate(_cfg_dict(tmp_path))
    a, b = question_jobs(cfg), question_jobs(cfg)
    assert [j["id"] for j in a] == [j["id"] for j in b]
    assert len({j["id"] for j in a}) == len(a)
    assert not {j["doc_id"] for j in a} & set(cfg.seeds.eval_docs)  # eval docs never seed training questions


def test_verify_on_messages_rows(tmp_path):
    cfg = Config.model_validate(_cfg_dict(tmp_path))
    cfg.run_dir.mkdir()
    meta = {"doc_id": "d", "chunk_id": "d#0", "persona": "p", "task": "howto"}
    rows = [
        {"id": "b", **prose_row(meta, "How do I set a time limit for my job?", "```bash\n#SBATCH --time=01:00:00\n```")},
        {"id": "a", **prose_row(meta, "How do I ask for two GPUs on one node?", "```bash\n#SBATCH --gpu-count=2\n```")},
        {"id": "c", "question": "How do I set a time limit for my job?", "answer": "Use --time."},  # legacy dup
    ]
    write_jsonl(cfg.run_dir / "generated.jsonl", rows)
    verify.run(cfg)
    kept = read_jsonl(cfg.run_dir / "verified.jsonl")
    rejected = {r["id"]: r["reject_reason"] for r in read_jsonl(cfg.run_dir / "rejected.jsonl")}
    assert [r["id"] for r in kept] == ["b", "c"]  # same question text: several answers per question are kept
    assert rejected == {"a": "bad_flags"}
