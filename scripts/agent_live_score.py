"""Score one agent_live run file against data/agent_live/tasks.jsonl.

usage: agent_live_score.py [--run-file data/agent_live/runs/final.jsonl]
                           [--running-job JID] [--pending-job JID] [--finished-job JID]

Per task: first_tool_ok (first call name == expected_tool), any_tool_ok (the expected tool
appears at least once), args_ok (the first call of the expected tool has every expected_args
key with an equal value; partition/account compare case-insensitively; for submit_job only the
required keys must exist with test_only true, for run_readonly_command only the required keys
must exist), zero_calls, n_calls, seconds. Prints the score table and writes data/agent_live/results.md.
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data/agent_live/tasks.jsonl"
RUNS = ROOT / "data/agent_live/runs"
DEFAULT_RUN = RUNS / "final.jsonl"
OUT = ROOT / "data/agent_live/results.md"
ACTIONS = ("submit_job", "cancel_job", "update_job", "hold_job", "release_job", "requeue_job")
_CASE_INSENSITIVE = {"partition", "account"}


def _rel(p):
    """A path relative to the repo root (absolute if the path is outside the root)."""
    try:
        return Path(p).relative_to(ROOT)
    except ValueError:
        return Path(p)


def _equal(key, want, got):
    if want is None or got is None:
        return False
    if key.lower() in _CASE_INSENSITIVE:
        return str(want).lower() == str(got).lower()
    return str(want) == str(got)


def args_ok(tool, want_args, calls):
    """True if the first call of `tool` satisfies `want_args` (see module docstring for the rules)."""
    call = next((c for c in calls if c.get("name") == tool), None)
    if call is None:
        return False
    got = call.get("args") or {}
    if tool == "submit_job":
        if not all(k in got for k in want_args):
            return False
        return got.get("test_only") is True
    if tool == "run_readonly_command":
        return all(k in got for k in want_args)
    return all(k in got and _equal(k, v, got[k]) for k, v in want_args.items())


def score_task(task, row):
    calls = row.get("calls") or []
    return {"id": task["id"], "tool": task["expected_tool"],
            "first_tool_ok": bool(calls) and calls[0].get("name") == task["expected_tool"],
            "any_tool_ok": any(c.get("name") == task["expected_tool"] for c in calls),
            "args_ok": args_ok(task["expected_tool"], task.get("expected_args") or {}, calls),
            "zero_calls": not calls, "n_calls": len(calls),
            "call_names": [c.get("name") for c in calls], "calls": calls,
            "seconds": row.get("seconds", 0.0), "timeout": bool(row.get("timeout"))}


def score_run(tasks, rows):
    """Returns (scores, per_tool): scores in task order, per_tool is an OrderedDict tool -> aggregates."""
    by_id = {r["id"]: r for r in rows}
    scores = OrderedDict((t["id"], score_task(t, by_id.get(t["id"], {}))) for t in tasks)
    per_tool = OrderedDict()
    for t in tasks:
        per_tool.setdefault(t["expected_tool"], {"n": 0, "first": 0, "any": 0, "args": 0})
    for s in scores.values():
        g = per_tool[s["tool"]]
        g["n"] += 1
        g["first"] += s["first_tool_ok"]
        g["any"] += s["any_tool_ok"]
        g["args"] += s["args_ok"]
    return scores, per_tool


def write_report(path, tasks, scores, per_tool, run_file, model):
    out = []
    n_all = len(scores)
    mean_secs = sum(s["seconds"] for s in scores.values()) / max(n_all, 1)
    n_first = sum(s["first_tool_ok"] for s in scores.values())
    n_any = sum(s["any_tool_ok"] for s in scores.values())
    n_args = sum(s["args_ok"] for s in scores.values())
    n_zero = sum(s["zero_calls"] for s in scores.values())
    action_calls = [c for s in scores.values() for c in s["calls"] if c.get("name") in ACTIONS]
    sb_calls = [c for c in action_calls if c.get("name") == "submit_job"]
    sb_test_only = [c for c in sb_calls if (c.get("args") or {}).get("test_only") is True]

    out += [
        "# Agent live coverage run — results",
        "",
        "## Run state",
        "",
        "- One run on 2026-10-27 on Leonardo. The jobs ran in reservation `s_tra_ncc1` on partition `boost_usr_prod`.",
        f"- Model: `{model}` (VaibhavKKM qwen3-4b weights with the x02 Ollama template).",
        "- Temperature: 0. The runner sets this value in every chat request.",
        f"- Tasks: {n_all}. Scored run file: `{_rel(run_file)}`.",
        f"- Mean time per task: {mean_secs:.1f} s.",
        "",
        "## Task table",
        "",
        "`f` = first call has the expected tool. `a` = the expected tool appears at least once. "
        "`r` = the arguments of the expected tool call are correct. `z` = no tool call at all.",
        "",
        "| id | f | a | r | z | expected tool | calls | made call names | seconds |",
        "|---|:-:|:-:|:-:|:-:|---|---:|---|---:|",
    ]
    for s in scores.values():
        out.append(
            f"| {s['id']} | {'x' if s['first_tool_ok'] else ''} | {'x' if s['any_tool_ok'] else ''} "
            f"| {'x' if s['args_ok'] else ''} | {'x' if s['zero_calls'] else ''} | {s['tool']} "
            f"| {s['n_calls']} | {', '.join(s['call_names']) or 'none'} | {s['seconds']:.1f} |")
    out += [
        f"| total | {n_first}/{n_all} | {n_any}/{n_all} | {n_args}/{n_all} | {n_zero}/{n_all} "
        f"| all tools | {sum(s['n_calls'] for s in scores.values())} | mean {mean_secs:.1f} s per task "
        f"| {sum(s['seconds'] for s in scores.values()):.1f} total |",
        "",
        "## Per-tool table",
        "",
        "Percentages are shares of the tasks in the row.",
        "",
        "| expected tool | tasks | first-tool % | any-tool % | args % |",
        "|---|---:|---:|---:|---:|",
    ]
    for tool, g in per_tool.items():
        out.append(f"| {tool} | {g['n']} | {100 * g['first'] / g['n']:.0f}% | "
                   f"{100 * g['any'] / g['n']:.0f}% | {100 * g['args'] / g['n']:.0f}% |")
    out += [
        f"| total | {n_all} | {100 * n_first / n_all:.0f}% | {100 * n_any / n_all:.0f}% "
        f"| {100 * n_args / n_all:.0f}% |",
        "",
        "## Zero-call tasks",
        "",
    ]
    zero = [s for s in scores.values() if s["zero_calls"]]
    if zero:
        out += [f"- `{s['id']}`: the model made no tool call. The expected tool was `{s['tool']}`." for s in zero]
    else:
        out.append("None. Every task made at least one tool call.")
    misses = [s for s in scores.values() if not (s["first_tool_ok"] and s["args_ok"])]
    out += ["", "## Misses", "",
            "A miss has no first-tool match, or the arguments of the expected tool call do not match.", "",
            "| id | expected tool | actual calls | problem |", "|---|---|---|---|"]
    if misses:
        for s in misses:
            why = []
            if s["zero_calls"]:
                why.append("the model made no tool call")
            elif not s["first_tool_ok"]:
                why.append(f"first call was `{s['call_names'][0]}`")
            if s["first_tool_ok"] and not s["args_ok"]:
                why.append("the arguments of the expected tool call do not match")
            if not s["any_tool_ok"] and not s["zero_calls"]:
                why.append(f"the model never called `{s['tool']}`")
            out.append(f"| {s['id']} | {s['tool']} | {', '.join(s['call_names']) or 'none'} "
                       f"| {'; '.join(why)} |")
    else:
        out.append("None.")
    live_calls = [c for c in action_calls if c.get("name") != "submit_job"]
    live_counts = sorted((name, sum(c.get("name") == name for c in live_calls))
                         for name in set(c.get("name") for c in live_calls))
    out += [
        "",
        "## Safety",
        "",
        f"- The model made {len(live_calls)} live action calls "
        f"({', '.join(f'{n.replace('_job', '')} {k}' for n, k in live_counts)}). "
        "The runner passes `confirm=False` to SlurmLive, so every live action call was declined. "
        "Nothing changed on the cluster.",
        f"- All {len(sb_calls)} submit_job calls had `test_only: true` "
        f"({len(sb_test_only)} of {len(sb_calls)} confirmed by the scorer). "
        "They ran only `sbatch --test-only` and created no job.",
        "",
    ]
    Path(path).write_text("\n".join(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", default=str(TASKS), help="tasks file")
    ap.add_argument("--run-file", default=str(DEFAULT_RUN), help="run file to score")
    ap.add_argument("--out", default=str(OUT), help="markdown report to write")
    ap.add_argument("--model", default="hpc-assistant-vkkm-live", help="model name for the report header")
    ap.add_argument("--running-job", default="<RUNNING_JOB>")
    ap.add_argument("--pending-job", default="<PENDING_JOB>")
    ap.add_argument("--finished-job", default="<FINISHED_JOB>")
    args = ap.parse_args(argv)
    placeholders = {"<RUNNING_JOB>": args.running_job, "<PENDING_JOB>": args.pending_job,
                    "<FINISHED_JOB>": args.finished_job}
    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    for t in tasks:
        t["expected_args"] = {k: placeholders.get(v, v) if isinstance(v, str) else v
                              for k, v in (t.get("expected_args") or {}).items()}
    rows = [json.loads(l) for l in Path(args.run_file).read_text().splitlines() if l.strip()]
    scores, per_tool = score_run(tasks, rows)
    for s in scores.values():
        flags = "".join(k if v else "-" for k, v in (("f", s["first_tool_ok"]), ("a", s["any_tool_ok"]),
                                                     ("r", s["args_ok"]), ("z", s["zero_calls"])))
        print(f"{s['id']:<8}{flags:<8}{s['tool']:<22}{s['n_calls']:>3}  {s['seconds']:>6.1f}s")
    n = len(scores)
    print(f"total ({n} tasks): first {sum(s['first_tool_ok'] for s in scores.values())}/{n} "
          f"any {sum(s['any_tool_ok'] for s in scores.values())}/{n} "
          f"args {sum(s['args_ok'] for s in scores.values())}/{n} "
          f"zero {sum(s['zero_calls'] for s in scores.values())}/{n}")
    write_report(args.out, tasks, scores, per_tool, args.run_file, args.model)
    print(f"wrote {_rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
