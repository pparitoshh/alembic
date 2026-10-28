"""Tests for research/train_multistep_tools/build_tool_kd.py.

Hermetic: a small fake tokenizer (no model download, no GPU, no network) and tmp dirs only. The last test
imports research/curiosity_b300/train_kd.py copy-free with stubbed torch and transformers, so its real
loader code reads the part files the build wrote.
"""
import importlib.util
import json
import re
import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "research/train_multistep_tools/build_tool_kd.py"
TRAIN_KD = ROOT / "research/curiosity_b300/train_kd.py"

_spec = importlib.util.spec_from_file_location("build_tool_kd", BUILD)
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)

SYSTEM = "You are an HPC assistant. Use tools for live state."
IM_END = 1 << 20
WORD = re.compile(r"\w+|[^\w\s]")

TOOLS = [
    {"type": "function", "function": {"name": "job_status", "description": "job state",
                                      "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "gpu_availability", "description": "gpu load",
                                      "parameters": {"type": "object", "properties": {"partition": {"type": "string"}}}}},
]


class FakeTok:
    """Stands in for the student tokenizer: one id per word or punctuation char, deterministic."""

    def __init__(self):
        self.vocab, self.calls = {}, []

    def _ids(self, text):
        out = []
        for w in WORD.findall(text):
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab) + 1
            out.append(self.vocab[w])
        return out

    def apply_chat_template(self, msgs, tools=None, add_generation_prompt=True, tokenize=True, **kw):
        ids = []
        for m in msgs:
            ids += self._ids("<" + m["role"] + ">")
            ids += self._ids(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                ids += self._ids(tc["function"]["name"])
        if add_generation_prompt:
            ids += self._ids("<assistant>")
        self.calls.append({"messages": msgs, "tools": tools, "ids": ids})
        return ids

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": self._ids(text)}

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return IM_END


def make_row(target=None):
    return {
        "id": "traj7#t3",
        "family": "hpc",
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "Is job 42 done, and how busy is the gpu partition?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"type": "function", "function": {"name": "job_status", "arguments": {"job_id": "42"}}}]},
            {"role": "user", "content": "job_status: job 42 is RUNNING"},
            {"role": "assistant", "content": target if target is not None else
                'The job runs. Checking the load.\n<tool_call>\n{"name": "gpu_availability", "arguments": {"partition": "gpu"}}\n</tool_call>'},
        ],
    }


def run_build(tmp_path, rows, tok, *extra):
    src = tmp_path / "src.jsonl"
    with open(src, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "parts"
    bk.main([str(src), str(out), "--onehot", "--system", SYSTEM, *extra], tok=tok)
    return out


def read_part(out_dir):
    z = np.load(out_dir / "part0.npz")
    return {k: z[k] for k in ("prompt", "out", "topk_ids", "topk_lp", "prompt_off", "out_off")}


def test_prompt_out_split(tmp_path):
    tok = FakeTok()
    row = make_row()
    d = read_part(run_build(tmp_path, [row], tok, "--tool-prefix", ""))
    call = tok.calls[0]
    assert call["messages"][0] == {"role": "system", "content": SYSTEM}
    assert call["messages"][1:] == row["messages"][:-1]  # the history without the target, system first
    assert d["prompt"].tolist() == call["ids"]
    assert d["out"].tolist() == tok(row["messages"][-1]["content"])["input_ids"] + [IM_END]
    assert d["prompt_off"].tolist() == [0, len(d["prompt"])]
    assert d["out_off"].tolist() == [0, len(d["out"])]
    assert d["prompt"].dtype == np.int32 and d["out"].dtype == np.int32
    checking, running = tok.vocab["Checking"], tok.vocab["RUNNING"]
    assert checking in d["out"].tolist() and checking not in d["prompt"].tolist()
    assert running in d["prompt"].tolist() and running not in d["out"].tolist()


def test_prefix_rename(tmp_path):
    tok = FakeTok()
    tok("job_status gpu_availability mcp__hpc__job_status mcp__hpc__gpu_availability")  # seed the four ids
    d = read_part(run_build(tmp_path, [make_row()], tok))
    call = tok.calls[0]
    assert [t["function"]["name"] for t in call["tools"]] == ["mcp__hpc__job_status", "mcp__hpc__gpu_availability"]
    msgs = call["messages"]
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "mcp__hpc__job_status"
    assert msgs[3]["content"] == "mcp__hpc__job_status: job 42 is RUNNING"
    prompt, out = d["prompt"].tolist(), d["out"].tolist()
    assert tok.vocab["mcp__hpc__job_status"] in prompt and tok.vocab["mcp__hpc__gpu_availability"] in out
    for bare in ("job_status", "gpu_availability"):
        assert tok.vocab[bare] not in prompt and tok.vocab[bare] not in out


def test_tool_prefix_option(tmp_path):
    tok = FakeTok()
    run_build(tmp_path, [make_row()], tok, "--tool-prefix", "tools.")
    call = tok.calls[0]
    assert call["tools"][0]["function"]["name"] == "tools.job_status"
    assert call["messages"][3]["content"].startswith("tools.job_status:")


def test_onehot_arrays(tmp_path):
    d = read_part(run_build(tmp_path, [make_row()], tok=FakeTok()))
    n = len(d["out"])
    ti, tl = d["topk_ids"], d["topk_lp"]
    assert ti.shape == (n, 20) and tl.shape == (n, 20)
    assert ti.dtype == np.int32 and tl.dtype == np.float16
    assert (ti[:, 0] == d["out"]).all()
    assert (tl[:, 0] == 0).all()
    assert (tl[:, 1:] == np.float16(-1e4)).all()


def test_fill_topk_ranks_like_teacher_gen():
    LP = namedtuple("LP", "rank logprob")
    pos = [{11: LP(2, -3.0), 7: LP(0, -0.5), 9: LP(1, -1.25), 13: LP(3, -6e4)}]
    ti, tl = bk.fill_topk(pos, k=3)
    assert ti.tolist() == [[7, 9, 11]]
    assert tl.tolist() == [[-0.5, -1.25, -3.0]]
    ti, tl = bk.fill_topk(pos, k=4)
    assert tl.tolist() == [[-0.5, -1.25, -3.0, -1e4]]  # below the floor, clamped as teacher_gen clamps
    ti, tl = bk.fill_topk(pos + [{}], k=20)
    assert ti[1].tolist() == [0] * 20 and tl[1, 0] == np.float16(-1e4)


def test_reject_rules(tmp_path, capsys):
    tok = FakeTok()
    name_in_text = make_row(
        target='I will now call job_status for you.\n<tool_call>\n{"name": "gpu_availability", "arguments": {}}\n</tool_call>')
    bad_json = make_row(target='<tool_call>\n{"name": gpu_availability}\n</tool_call>')
    out_dir = run_build(tmp_path, [make_row(), name_in_text, bad_json], tok)
    printed = capsys.readouterr().out
    assert "'tool_name_text': 1" in printed and "'bad_json': 1" in printed
    d = read_part(out_dir)
    assert len(d["out_off"]) == 2 and d["out_off"].tolist() == [0, len(d["out"])]  # only the good row is kept
    meta = [json.loads(l) for l in open(out_dir / "part0.jsonl")]
    assert [m["id"] for m in meta] == ["traj7#t3"]
    assert meta[0]["n_prompt"] == len(d["prompt"]) and meta[0]["n_out"] == len(d["out"])


def test_reject_over_max_len(tmp_path):
    d = read_part(run_build(tmp_path, [make_row()], FakeTok(), "--max-len", "30"))
    assert d["prompt_off"].tolist() == [0] and d["out_off"].tolist() == [0]
    assert d["topk_ids"].shape == (0, 20) and d["topk_lp"].dtype == np.float16


def test_max_len_default_comes_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_LEN", "30")
    d = read_part(run_build(tmp_path, [make_row()], FakeTok()))
    assert d["out_off"].tolist() == [0]


def test_default_system_matches_teacher_gen_config():
    assert bk.default_system().startswith("You are an HPC assistant")


def load_train_kd(monkeypatch, data_dir):
    """Import train_kd.py with torch and transformers stubbed (no GPU, no model) and EPOCHS=0, so only the
    real data code runs: the part scan, the offsets and the load() function."""
    torch = ModuleType("torch")
    dist = ModuleType("torch.distributed")
    dist.init_process_group = MagicMock()
    dist.get_rank, dist.get_world_size = (lambda: 1), (lambda: 1)
    dist.broadcast = dist.all_reduce = dist.barrier = dist.destroy_process_group = MagicMock()
    cuda = ModuleType("torch.cuda")
    cuda.set_device = MagicMock()
    nn, fn = ModuleType("torch.nn"), ModuleType("torch.nn.functional")
    optim = ModuleType("torch.optim")
    optim.AdamW = MagicMock()
    optim.lr_scheduler = ModuleType("torch.optim.lr_scheduler")
    optim.lr_scheduler.LambdaLR = MagicMock()
    model = MagicMock()
    model.parameters.return_value = iter([])
    model.to.return_value = model
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = MagicMock()
    transformers.AutoModelForCausalLM.from_pretrained.return_value = model
    transformers.AutoTokenizer = MagicMock()
    torch.device = MagicMock()
    torch.float32 = "float32"  # only passed through to the stubbed from_pretrained
    torch.distributed, torch.cuda, torch.nn, torch.optim = dist, cuda, nn, optim
    for name, mod in [("torch", torch), ("torch.distributed", dist), ("torch.cuda", cuda), ("torch.nn", nn),
                      ("torch.nn.functional", fn), ("torch.optim", optim), ("torch.optim.lr_scheduler", optim.lr_scheduler),
                      ("transformers", transformers)]:
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("EPOCHS", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train_kd.py", "student", str(data_dir), str(data_dir.parent / "unused_out")])
    spec = importlib.util.spec_from_file_location("train_kd_under_test", TRAIN_KD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_kd_loader_reads_the_part_files(tmp_path, monkeypatch):
    tok = FakeTok()
    replay = {"id": "rep42",  # replay row as the join in build.sh writes it: system inside, no tools field
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": "What is sbatch?"},
                           {"role": "assistant", "content": "sbatch puts a script in the queue."}]}
    out_dir = run_build(tmp_path, [make_row(), make_row(target="Both checks pass, the partition is quiet."), replay], tok)
    mod = load_train_kd(monkeypatch, out_dir)
    assert len(mod.items) == 3
    z = np.load(out_dir / "part0.npz")
    assert sum(p1 - p0 for _, p0, p1, _, _ in mod.items) == len(z["prompt"])
    assert sum(o1 - o0 for _, _, _, o0, o1 in mod.items) == len(z["out"])
    for f, p0, p1, o0, o1 in mod.items:
        d = mod.load(f)
        assert set(d) == {"prompt", "out", "topk_ids", "topk_lp"}
        assert d["prompt"][p0:p1].tolist() == z["prompt"][p0:p1].tolist()
        assert d["out"][o0:o1][-1] == IM_END  # every target ends with <|im_end|>
        assert len(d["topk_ids"][o0:o1]) == len(d["topk_lp"][o0:o1]) == o1 - o0
    meta = [json.loads(l) for l in open(out_dir / "part0.jsonl")]
    assert [m["id"] for m in meta] == ["traj7#t3", "traj7#t3", "rep42"]
