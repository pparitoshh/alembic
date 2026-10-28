"""Measure first-tool accuracy of the student model with two system prompts.
Calls Ollama /api/chat only. It never runs a tool. Standard library plus distillkit."""
import http.client
import json
import math
import time
import urllib.request
from pathlib import Path

from distillkit.assistant import SYSTEM_PROMPT
from distillkit.slurm_live import openai_tools

HERE = Path(__file__).parent
URL = "http://127.0.0.1:11434/api/chat"
MODEL = "hpc-assistant-vkkm-live:latest"
PROMPTS = {
    "base": SYSTEM_PROMPT,
    "extra": SYSTEM_PROMPT + "\nPrefer a specific tool. Use run_readonly_command only when no tool fits.",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ask(system, question, tools):
    body = {"model": MODEL, "stream": False, "options": {"temperature": 0}, "tools": tools,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                msg = json.load(r).get("message", {})
            break
        except (OSError, http.client.HTTPException):
            if attempt == 7:
                raise
            time.sleep(20)
    calls = [c["function"]["name"] for c in msg.get("tool_calls") or []]
    return calls or ["TEXT"]


def main():
    questions = [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines() if line.strip()]
    tools = openai_tools()
    print(f"tool count: {len(tools)}", flush=True)
    assert len(tools) == 19, "expected 19 tools: set PYTHONPATH to a src with the full tool list"
    out = {}
    for pname, system in PROMPTS.items():
        rows = []
        for q in questions:
            got = ask(system, q["question"], tools)
            exp = q["expected_tool"]
            first = got[0]
            ok = (first == "TEXT") if exp == "none" else (first == exp)
            rows.append({"id": q["id"], "expected": exp, "got": got, "called": first != "TEXT", "correct": ok})
        n = len(rows)
        called = sum(r["called"] for r in rows)
        good = sum(r["correct"] for r in rows)
        lo, hi = wilson(good, n)
        summ = (f"{pname}: tool called {called}/{n}, first tool correct {good}/{n} "
                f"({good / n:.1%}, Wilson 95% [{lo:.1%}, {hi:.1%}])")
        print(f"\n== {pname} ==\n{'id':5} {'expected':22} {'got':40} ok")
        for r in rows:
            print(f"{r['id']:5} {r['expected']:22} {','.join(r['got']):40} {'Y' if r['correct'] else 'N'}")
        print(summ)
        out[pname] = {"summary": summ, "rows": rows}
    (HERE / "eval_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
