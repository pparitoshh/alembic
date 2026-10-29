"""KD data build for multi-turn tool examples (training fix 4, step 5).

Input rows are the examples research/agent_train/build_agent_data.py writes: {"id", "tools", "messages"},
and the last message is the assistant target (text plus tool-call blocks). For every kept row the script
builds one train_kd.py sequence:

- prompt ids: tok.apply_chat_template([system] + messages[:-1], tools=tools, add_generation_prompt=True,
  tokenize=True) with the system prompt of the evaluation (the same SYSTEM as teacher_gen.py; pass it with
  --system, or the default loader reads configs/rag65_seed42.json as teacher_gen.py does);
- out ids: the tokens of the target plus the <|im_end|> token;
- top-20 ids and log-probs per out token: one vLLM pass over prompt+out with prompt_logprobs=20
  (--teacher TEACHER_DIR, TP env as teacher_gen.py), or a one-hot distribution for pure SFT through the
  same loss (--onehot: no teacher, no GPU).

Output is partN.npz + partN.jsonl in exactly the format research/curiosity_b300/train_kd.py reads, so it
trains on them without changes.

Tool names appear as dsh shows them at inference (--tool-prefix, default mcp__hpc__): the script renames the
schemas, the tool-call blocks of the target and of the history, and bare tool names in the history text.

An example is rejected (counted and printed) when the target names a tool outside a tool-call block, when
a tool-call body does not parse as JSON, or when prompt+out is longer than train_kd.py's MAX_LEN
(--max-len, default from the MAX_LEN env, else 12288).

usage: build_tool_kd.py SRC.jsonl OUT_DIR (--teacher DIR | --onehot) --tokenizer DIR
       [--part N] [--nparts M] [--tool-prefix mcp__hpc__] [--system TEXT] [--max-len N]
"""
import argparse
import copy
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

K = 20  # teacher top-k, as teacher_gen.py
TC_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ROOT = Path(__file__).resolve().parents[2]


class Reject(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def default_system():
    """The system prompt of the evaluation, read the way teacher_gen.py reads it."""
    sys.path.insert(0, str(ROOT / "src"))
    from distillkit.config import load_config

    return load_config(ROOT / "configs/rag65_seed42.json").student.system_prompt


def _fn(tool):
    return tool.get("function", tool)  # OpenAI {"type": "function", "function": ...} or a bare schema


def tool_names(tools):
    return [n for t in (tools or []) if isinstance(n := _fn(t).get("name"), str)]


def rename_tools(tools, prefix):
    out = []
    for t in tools or []:
        t = copy.deepcopy(t)
        f = _fn(t)
        if isinstance(f.get("name"), str) and not f["name"].startswith(prefix):
            f["name"] = prefix + f["name"]
        out.append(t)
    return out


def rename_call_blocks(text, prefix):
    """Prefix the tool name in every tool-call block; the text between the blocks stays byte-identical."""
    def sub(m):
        try:
            body = json.loads(m.group(1))
        except json.JSONDecodeError:
            raise Reject("bad_json")
        if isinstance(body, dict) and isinstance(body.get("name"), str) and not body["name"].startswith(prefix):
            body["name"] = prefix + body["name"]
        return "<tool_call>\n" + json.dumps(body, ensure_ascii=False) + "\n</tool_call>"

    return TC_BLOCK.sub(sub, text)


def rename_history(msgs, names, prefix):
    out = []
    for m in msgs:
        m = copy.deepcopy(m)
        if isinstance(m.get("content"), str):
            m["content"] = rename_call_blocks(m["content"], prefix)
            for n in names:  # renamed block bodies carry the prefix already, so the boundary skips them
                if not n.startswith(prefix):
                    m["content"] = re.sub(r"\b" + re.escape(n) + r"\b", prefix + n, m["content"])
        for tc in m.get("tool_calls") or []:
            f = _fn(tc)
            if isinstance(f.get("name"), str) and not f["name"].startswith(prefix):
                f["name"] = prefix + f["name"]
        out.append(m)
    return out


def build_row(row, tok, system, prefix, max_len):
    """(prompt_ids, out_ids, renamed_target) for one example row; Reject on the rules of the module doc."""
    msgs = row.get("messages") or []
    if not msgs or not isinstance(msgs[-1].get("content"), str) or not msgs[-1]["content"].strip():
        raise Reject("no_target")
    names = tool_names(row.get("tools"))
    target = rename_call_blocks(msgs[-1]["content"], prefix)
    outside = TC_BLOCK.sub(" ", target)
    if any(re.search(r"\b" + re.escape(n) + r"\b", outside) for n in set(names + [prefix + n for n in names])):
        raise Reject("tool_name_text")
    history = rename_history([m for m in msgs[:-1] if m.get("role") != "system"], names, prefix)
    ids = tok.apply_chat_template([{"role": "system", "content": system}] + history,
                                  tools=rename_tools(row.get("tools"), prefix) or None,
                                  add_generation_prompt=True, tokenize=True)
    prompt = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    enc = tok(target, add_special_tokens=False)
    out = list(enc["input_ids"] if hasattr(enc, "keys") else enc.input_ids) + [tok.convert_tokens_to_ids("<|im_end|>")]
    if len(prompt) + len(out) > max_len:
        raise Reject("too_long")
    return prompt, out, target


def fill_topk(lps_by_pos, k=K):
    """topk_ids/topk_lp for the out tokens from vLLM prompt_logprobs dicts (one dict per out-token position),
    ranked and clamped exactly as teacher_gen.py ranks them."""
    ti = np.zeros((len(lps_by_pos), k), np.int32)
    tl = np.full((len(lps_by_pos), k), -1e4, np.float16)
    for t, lps in enumerate(lps_by_pos):
        if not lps:
            continue
        for j, (tid, lp) in enumerate(sorted(lps.items(), key=lambda kv: kv[1].rank)[:k]):
            ti[t, j], tl[t, j] = tid, max(lp.logprob, -1e4)
    return ti, tl


def onehot_topk(out_ids, k=K):
    """All teacher mass on the target token: pure SFT through the KD loss."""
    ti = np.zeros((len(out_ids), k), np.int32)
    tl = np.full((len(out_ids), k), -1e4, np.float16)
    ti[:, 0] = out_ids
    tl[:, 0] = 0.0
    return ti, tl


def score_topk(pairs, teacher, k=K):
    """Score prompt+out for every kept row in one vLLM pass; position len(prompt)+j predicts out token j."""
    from vllm import LLM, SamplingParams  # heavy import: only when a teacher runs

    llm = LLM(teacher, tensor_parallel_size=int(os.environ.get("TP", 4)), max_model_len=16384,
              gpu_memory_utilization=0.90, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=k)
    res = llm.generate([{"prompt_token_ids": list(p) + list(o)} for p, o in pairs], sp)
    return [fill_topk(g.prompt_logprobs[len(p): len(p) + len(o)], k) for (p, o), g in zip(pairs, res)]


def write_part(out_dir, part, P, O, TI, TL, meta):
    po, oo = np.cumsum([0] + [len(x) for x in P]), np.cumsum([0] + [len(x) for x in O])
    np.savez(f"{out_dir}/part{part}.npz.tmp.npz",
             prompt=np.concatenate(P) if P else np.array([], np.int32),
             out=np.concatenate(O) if O else np.array([], np.int32),
             topk_ids=np.concatenate(TI) if TI else np.zeros((0, K), np.int32),
             topk_lp=np.concatenate(TL) if TL else np.full((0, K), -1e4, np.float16),
             prompt_off=po, out_off=oo)
    with open(f"{out_dir}/part{part}.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    os.replace(f"{out_dir}/part{part}.npz.tmp.npz", f"{out_dir}/part{part}.npz")


def main(argv=None, tok=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--tokenizer", default=None, help="student model dir; its chat template renders the prompts")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--teacher")
    g.add_argument("--onehot", action="store_true", help="no teacher: all mass on the target token (pure SFT)")
    ap.add_argument("--tool-prefix", default="mcp__hpc__")
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-len", type=int, default=int(os.environ.get("MAX_LEN", 12288)))
    a = ap.parse_args(argv)
    if tok is None:
        if not a.tokenizer:
            ap.error("--tokenizer is required (the student model dir)")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    system = a.system or default_system()
    rows = [json.loads(l) for i, l in enumerate(open(a.src)) if i % a.nparts == a.part]
    pairs, meta, rej = [], [], Counter()
    for row in rows:
        try:
            prompt, out, target = build_row(row, tok, system, a.tool_prefix, a.max_len)
        except Reject as e:
            rej[e.reason] += 1
            continue
        pairs.append((prompt, out))
        meta.append({"id": row.get("id"), "domain": row.get("domain") or row.get("family"), "text": target,
                     "n_prompt": len(prompt), "n_out": len(out)})
    os.makedirs(a.out, exist_ok=True)
    if pairs:
        TI, TL = zip(*score_topk(pairs, a.teacher)) if a.teacher else zip(*(onehot_topk(o) for _, o in pairs))
    else:
        TI, TL = (), ()
    write_part(a.out, a.part, [np.asarray(p, np.int32) for p, _ in pairs], [np.asarray(o, np.int32) for _, o in pairs],
               list(TI), list(TL), meta)
    counts = {r: rej.get(r, 0) for r in ("tool_name_text", "bad_json", "too_long", "no_target")}
    print(f"TOOL KD part {a.part}: {len(meta)} of {len(rows)} examples kept, "
          f"{sum(len(o) for _, o in pairs)} answer tokens, rejected {counts}")


if __name__ == "__main__":
    main()
