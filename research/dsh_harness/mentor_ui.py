"""Run each mentor use case in a new dsh web session (visible Chrome), save the visible answer, and score the run
against the dsh session logs. Or score an existing run without a browser.

Usage:
    mentor_ui.py run <dsh-web.log> <usecases.jsonl> <outdir> [--first ID] [--sessions DIR]
    mentor_ui.py score <usecases.jsonl> <outdir> [--sessions DIR]

`run` needs Playwright, from uv outside the project venv:
    uv run --no-project --with playwright python research/dsh_harness/mentor_ui.py run \
        ~/.local/share/hpc-ssh/dsh-web.log data/mentor_usecases/curiosity/usecases.jsonl <outdir>
`run` writes <outdir>/runs.json and one screenshot per case, then scores. `score` reads <outdir>/runs.json and
works without playwright. --sessions is the dsh session log folder (default below); each session-*/session.v4.jsonl.zstd
is matched to a run by its createdAt (epoch ms): start-5 <= createdAt/1000 <= end.

Scores go to <outdir>/scores.json (per-case dicts plus a totals dict) and <outdir>/scores.md (one table row per case).
must_not and key_points stay null in the output: they need a human or a judge.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

SESSIONS = "~/.dsh/sessions/--home-paritosh-Documents-deepseek-harness-default-workspace--"
PREFIX = "mcp__hpc__"
# Job-changing tools hidden in a read-only run; they are dropped from the expected actions of a case
ACTION_TOOLS = ("submit_job", "cancel_job", "update_job", "hold_job", "release_job", "requeue_job")
# Announcement of work to come at the end of an answer (checked on the last 200 characters)
STOP_EARLY = re.compile(r"(I will|I'll|Let me|Now,? I|I am going to)[^.]*\b(check|call|query|verify|look)\b")


def prompt(c):
    text = c["user_request"]
    if c.get("context"):
        text += "\n\nContext: " + c["context"]
    return text


def composer(pg):
    for sel in ["textarea", "[contenteditable=true]"]:
        loc = pg.locator(sel)
        if loc.count():
            return loc.last
    raise RuntimeError("no chat input found")


def wait_stable(pg, quiet=15, limit=300):
    """Wait until the page text does not change for `quiet` seconds."""
    last, since, start = None, time.time(), time.time()
    while time.time() - start < limit:
        text = pg.inner_text("body")
        if text != last:
            last, since = text, time.time()
        elif time.time() - since > quiet:
            return text
        time.sleep(1)
    return last


def run(log, cases_file, outdir, first=None):
    """The browser run: one dsh web session per case, <outdir>/runs.json and a screenshot per case."""
    from playwright.sync_api import sync_playwright  # only the run subcommand needs playwright

    url = re.findall(r"http://127\.0\.0\.1:3080/\?token=[A-Za-z0-9_-]+", open(log).read())[-1]
    cases = load_cases(cases_file)
    if first:
        cases = cases[[c["id"] for c in cases].index(first):]
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=False, slow_mo=100)
        pg = b.new_page(viewport={"width": 1300, "height": 850})
        pg.goto(url)
        pg.wait_for_timeout(4000)
        runs = []
        for c in cases:
            print(f"===== {c['id']}: {c['user_request'][:90]}", flush=True)
            pg.get_by_text("New Session", exact=True).first.click()
            pg.wait_for_timeout(2500)
            before = pg.inner_text("body")
            start = time.time()
            box = composer(pg)
            box.click()
            box.fill(prompt(c))
            pg.keyboard.press("Enter")
            after = wait_stable(pg)
            answer = after[len(before):] if after.startswith(before) else after[-4000:]
            runs.append({"id": c["id"], "start": start, "end": time.time(), "visible": answer.strip()})
            pg.screenshot(path=f"{outdir}/{c['id']}.png")
            json.dump(runs, open(f"{outdir}/runs.json", "w"), indent=1)
            print(answer.strip()[:600], flush=True)
        print("===== done", flush=True)
        pg.wait_for_timeout(3000)
        b.close()


def load_cases(cases_file):
    return [json.loads(l) for l in open(cases_file) if l.strip()]


def zstd_lines(path):
    """The lines of a zstd file. The "--" guard keeps a leading "-" in the path literal (session folders
    start with "--")."""
    out = subprocess.run(["zstd", "-dc", "--", str(path)], capture_output=True, text=True, check=True)
    return [l for l in out.stdout.splitlines() if l.strip()]


def load_sessions(sessions_dir):
    """(created epoch s, session id, events) of each session-*/session.v4.jsonl.zstd, sorted by creation."""
    sessions = []
    for d in sorted(Path(sessions_dir).expanduser().glob("session-*")):
        f = d / "session.v4.jsonl.zstd"
        if not f.exists():
            continue
        lines = zstd_lines(f)
        head = json.loads(lines[0]) if lines else {}
        events = [json.loads(l) for l in lines[1:]]
        sessions.append((head.get("createdAt", 0) / 1000, head.get("id"), events))
    return sorted(sessions)


def match_session(run, sessions):
    """The session of a run: created within [start-5, end] epoch seconds."""
    for created, sid, events in sessions:
        if run["start"] - 5 <= created <= run["end"]:
            return sid, events
    return None, []


def strip(name):
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def called_tools(events):
    """Real tool names called, prefix stripped, every call in order (duplicates included)."""
    return [strip(e["data"]["name"]) for e in events if e.get("type") == "tool/call" and e["data"].get("name")]


def assistant_texts(events):
    """The text of each assistant message (its text blocks joined)."""
    out = []
    for e in events:
        if e.get("type") == "assistant/message":
            out.append("".join(i.get("text", "") for i in e["data"]["message"]["content"] if i.get("type") == "text"))
    return out


def fake_tool_calls(events):
    """Tool names the model only wrote in its text instead of calling."""
    return sum(len(re.findall(PREFIX + r"\w+", t)) for t in assistant_texts(events))


def tool_errors(events):
    """Tool results that are errors: isError, or text starting with "Error" or containing "refused"."""
    n = 0
    for e in events:
        if e.get("type") != "tool/result":
            continue
        msg = e["data"].get("message", {})
        text = "".join(i.get("text", "") for i in msg.get("content", []) if i.get("type") == "text")
        if msg.get("isError") or text.startswith("Error") or "refused" in text:
            n += 1
    return n


def last_content(events):
    for e in reversed(events):
        if e.get("type") == "assistant/message":
            return e["data"]["message"]["content"]
    return []


def stopped_early(events):
    """True when the last answer ends with an announcement of work to come and no tool call follows."""
    content = last_content(events)
    text = "".join(i.get("text", "") for i in content if i.get("type") == "text")
    if any(i.get("type") == "tool-call" for i in content):
        return False
    return bool(STOP_EARLY.search(text[-200:]))


def score_case(run, case, sessions):
    """The automatic metrics of one case from its dsh session; must_not and key_points stay for a reviewer."""
    sid, events = match_session(run, sessions)
    tools = called_tools(events)
    names = set(tools)
    visible = [a for a in case.get("expected_actions", []) if a not in ACTION_TOOLS]
    hit = [a for a in case.get("expected_actions", []) if a in names]
    texts = assistant_texts(events)
    return {
        "id": case["id"],
        "session": sid,
        "tool_calls": tools,
        "expected_hit": hit,
        "expected_visible": visible,
        "recall": len(hit) / len(visible) if visible else None,
        "fake_tool_calls": fake_tool_calls(events),
        "tool_errors": tool_errors(events),
        "empty_answer": len(texts[-1] if texts else "") < 200,
        "stopped_early": stopped_early(events),
        "answer_chars": sum(len(t) for t in texts),
        "must_not": None,
        "key_points": None,
    }


def totals(scores):
    recalls = [s["recall"] for s in scores if s["recall"] is not None]
    return {
        "cases": len(scores),
        "cases_with_real_tools": sum(1 for s in scores if s["tool_calls"]),
        "real_tool_calls": sum(len(s["tool_calls"]) for s in scores),
        "mean_recall": sum(recalls) / len(recalls) if recalls else None,
        "fake_tool_call_cases": sum(1 for s in scores if s["fake_tool_calls"]),
        "tool_error_cases": sum(1 for s in scores if s["tool_errors"]),
        "empty_answer_cases": sum(1 for s in scores if s["empty_answer"]),
        "stopped_early_cases": sum(1 for s in scores if s["stopped_early"]),
        "missing_session_cases": sum(1 for s in scores if s["session"] is None),
    }


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, list):
        return ", ".join(v) or "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def scores_md(scores, total):
    """A Markdown table with one row per case, then the totals."""
    cols = ["id", "tool_calls", "expected_hit", "recall", "fake_tool_calls", "tool_errors",
            "empty_answer", "stopped_early", "answer_chars", "session"]
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    for s in scores:
        lines.append("| " + " | ".join(_fmt(s[c]) for c in cols) + " |")
    lines += ["", "| totals | value |", "| --- | --- |"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in total.items()]
    return "\n".join(lines) + "\n"


def score(cases_file, outdir, sessions_dir=SESSIONS):
    """Score <outdir>/runs.json against the use cases and the dsh session logs; writes scores.json and scores.md."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cases = {c["id"]: c for c in load_cases(cases_file)}
    runs = json.load(open(outdir / "runs.json"))
    sessions = load_sessions(sessions_dir)
    scores = [score_case(r, cases[r["id"]], sessions) for r in runs]
    total = totals(scores)
    json.dump(scores + [{"totals": total}], open(outdir / "scores.json", "w"), indent=1)
    (outdir / "scores.md").write_text(scores_md(scores, total))
    print("===== totals " + json.dumps(total), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run each use case in a new dsh web session, then score the run")
    p_run.add_argument("log", help="dsh-web.log with the session url")
    p_run.add_argument("cases", help="usecases.jsonl")
    p_run.add_argument("outdir")
    p_run.add_argument("--first", help="first case id to run (runs the rest after it)")
    p_run.add_argument("--sessions", default=SESSIONS, help="dsh session log folder for the scoring")
    p_score = sub.add_parser("score", help="score an existing run from <outdir>/runs.json, no browser needed")
    p_score.add_argument("cases", help="usecases.jsonl")
    p_score.add_argument("outdir")
    p_score.add_argument("--sessions", default=SESSIONS, help="dsh session log folder")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.log, args.cases, args.outdir, args.first)
    score(args.cases, args.outdir, args.sessions)


if __name__ == "__main__":
    main()
