"""Round 49 data: verified agent trajectories from real Slurm (research/agent_train/collect.py) become one training
example per assistant turn, exactly as research/tools38/build_tool_data.py does for the simulated workflows: the
prompt is the conversation so far (user turn with retrieved excerpts, earlier tool calls and the REAL tool results),
the target is the assistant turn (text plus <tool_call> blocks). Each example carries the tool schemas the harness
offers at inference (search_docs + the 9 live Slurm tools), which train_kd.py renders into the prompt. Replay rows
(non-tool rows of round 38's data, with Qwen3-235B scores; KD_MIXED=1) keep the HPC answering.

usage: build_agent_data.py OUT_DIR REPLAY_SRC TRAJ.jsonl... [--upsample N] [--n-replay N]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from distillkit.assistant import SEARCH_TOOL  # noqa: E402
from distillkit.slurm_live import openai_tools  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("replay")
ap.add_argument("traj", nargs="+")
ap.add_argument("--upsample", type=int, default=2)
ap.add_argument("--n-replay", type=int, default=12000)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
TOOLS = [SEARCH_TOOL] + openai_tools()


def call_text(tc):
    f = tc["function"]
    args = f["arguments"] if isinstance(f["arguments"], str) else json.dumps(f["arguments"], ensure_ascii=False)
    return '<tool_call>\n{"name": "' + f["name"] + '", "arguments": ' + args + "}\n</tool_call>"


def clean(m):
    """History message as the chat template expects it (ids are an API detail, not part of the rendered prompt)."""
    m = {k: v for k, v in m.items() if k in ("role", "content", "tool_calls")}
    if m.get("tool_calls"):
        m["tool_calls"] = [{"type": "function", "function": {"name": tc["function"]["name"], "arguments": (
            json.loads(tc["function"]["arguments"] or "{}") if isinstance(tc["function"]["arguments"], str)
            else tc["function"]["arguments"])}} for tc in m["tool_calls"]]
    return m


rows, fams, n_traj = [], {}, 0
for path in a.traj:
    for line in open(path):
        t = json.loads(line)
        if not t.get("pass"):
            continue
        n_traj += 1
        fams[t["family"]] = fams.get(t["family"], 0) + 1
        msgs = [clean(m) for m in t["messages"] if m["role"] != "system"]  # train_kd.py adds the system prompt
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            target = (m.get("content") or "").strip()
            if m.get("tool_calls"):
                target = ((target + "\n") if target else "") + "\n".join(call_text(tc) for tc in m["tool_calls"])
            if not target:
                continue
            ex = {"id": f"{t['id']}#t{i}", "question": t["question"], "tools": TOOLS, "family": t["family"],
                  "messages": msgs[:i] + [{"role": "assistant", "content": target}]}
            rows += [ex] * a.upsample
rng = random.Random(49)
rep = [r for r in map(json.loads, open(a.replay)) if not r.get("tools")]
rng.shuffle(rep)
rep = rep[:a.n_replay]
data = rows + rep
rng.shuffle(data)
with open(out / "data_src.jsonl", "w") as f:
    for ex in data:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"INPUT READY: {n_traj} verified trajectories {fams} -> {len(rows) // a.upsample} assistant turns x{a.upsample}, "
      f"{len(rep)} replay rows, {len(data)} training rows")
