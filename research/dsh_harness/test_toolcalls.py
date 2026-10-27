"""Check that a model returns structured tool_calls (not text in the Hermes `<tool_call>` format).

Standard library only. Run with PYTHONPATH=src. Never runs a tool call: it only reads
message.tool_calls[].function.name from the reply.

Example:
    PYTHONPATH=src .venv/bin/python3 research/dsh_harness/test_toolcalls.py --model hpc-assistant-vkkm-live:latest
"""
import argparse
import json
import socket
import sys
import urllib.request

from distillkit.assistant import SYSTEM_PROMPT
from distillkit.slurm_live import openai_tools

# question -> tool we want the model to call first
QUESTIONS = [
    ("Show my queue", "list_queue"),
    ("What is the status of job 12345?", "job_status"),
    ("Are any GPUs free right now?", "gpu_availability"),
    ("Which partitions does this cluster have?", "partition_info"),
    ("Show the log of job 12345", "read_job_log"),
    ("How much CPU time did job 12345 use?", "job_accounting"),
    ("List my pending jobs", "list_queue"),
    ("Why is job 12345 still pending?", "job_status"),
]


def chat(base_url, model, question):
    """One chat completion. Returns the reply message dict."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": question}],
        "tools": openai_tools(),
        "stream": False,
        "temperature": 0,
        "max_tokens": 1500,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]


def names(message):
    """The tool names in the reply (structured tool_calls only, never text)."""
    return [tc["function"]["name"] for tc in (message.get("tool_calls") or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    called = first_ok = total = 0
    for run in range(1, args.runs + 1):
        for question, want in QUESTIONS:
            total += 1
            try:
                got = names(chat(args.base_url, args.model, question))
            except (TimeoutError, socket.timeout):
                print(f"run {run}: {question!r} ERROR timeout")
                continue
            hit = bool(got) and got[0] == want
            if got:
                called += 1
            if hit:
                first_ok += 1
            print(f"run {run}: {question!r} want={want} got={got[0] if got else None}")
    print(f"tool called {called}/{total}, first tool correct {first_ok}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
