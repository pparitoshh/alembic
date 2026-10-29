"""Fix 4 step 4: verify multi-step tool trajectories before the data build.

A row keeps "pass": true only if all rules hold; otherwise "pass": false and "fail_reasons" lists the broken rules:
  schema      every tool call parses (JSON arguments) and passes the tool JSON schema
  leak        no tool name, <tool_call> text or JSON tool block in assistant text content
  grounding   job IDs, partitions and numbers with a unit in the final answer come from the user turn or a tool result
  final       at most 6 tool calls; the last message is a non-empty assistant answer, not a promise to act
  coverage    needs_tool -> >= 1 call; multi_step -> >= 2 distinct tools; expected_tools recall >= 0.5
  action      action tools only when the family or the request asks for that action
  claim       "submitted/cancelled/held/..." only after the matching action tool returned success

usage: verify_traj.py IN.jsonl OUT.jsonl
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from distillkit.slurm_live import openai_tools  # noqa: E402

SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in openai_tools()}
MAX_CALLS = 6
THINK = re.compile(r"<think>.*?</think>\s*", re.S)

# action tool -> (words in the family or the request that ask for it, words in the answer that claim it)
ACTIONS = {
    "submit_job": (r"submit|sbatch|launch", r"submitted"),
    "cancel_job": (r"cancel|kill|scancel|stop|abort", r"cancell?ed|killed"),
    "update_job": (r"update|change|extend|increase|decrease|modify|move|set|scontrol", r"updated|changed|extended"),
    "hold_job": (r"\bhold|pause", r"\bheld\b|put on hold"),
    "release_job": (r"release|unhold|resume", r"released"),
    "requeue_job": (r"requeue|resubmit|rerun|re-run|restart", r"requeued"),
}
PROMISE = re.compile(r"\b(i will|i'll|let me|i am going to|i'm going to|i need to)\s+(now\s+)?"
                     r"(check|look|query|run|fetch|see|find|get|submit|cancel|call|use|verify|inspect)\b", re.I)
JOB_ID = re.compile(r"(?<![\d.])\d{5,}(?![\d.])")
UNIT_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(TB|GB|MB|GiB|TiB|G|GPUs?|CPUs?|cores?|hours?|hrs?|h|nodes?)\b", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")
PART_NEAR = re.compile(r"\b(?:partition|-p|--partition[= ])\s*[`'\"]?([A-Za-z][\w-]*)|"
                       r"`?([A-Za-z][\w-]*)`?\s+partition\b", re.I)
PART_STOP = {"the", "a", "this", "that", "which", "your", "same", "gpu", "cpu", "any", "each", "default", "another",
             "different", "other", "one", "is", "on", "in", "name", "names", "info", "limits", "and", "or", "to"}


def check_schema(args, schema):
    """Small JSON schema checker for the flat tool schemas (required, types, enums, no extra keys)."""
    errs = []
    if not isinstance(args, dict):
        return ["arguments are not an object"]
    props = schema.get("properties", {})
    for r in schema.get("required", []):
        if r not in args:
            errs.append(f"missing required '{r}'")
    types = {"string": str, "integer": int, "boolean": bool, "number": (int, float), "object": dict, "array": list}
    for k, v in args.items():
        if k not in props:
            errs.append(f"unknown argument '{k}'")
            continue
        p = props[k]
        t = p.get("type")
        if t and not (isinstance(v, types[t]) and not (t in ("integer", "number") and isinstance(v, bool))):
            errs.append(f"'{k}' is not {t}")
        if "enum" in p and v not in p["enum"]:
            errs.append(f"'{k}' not in enum")
    return errs


def norm(text):
    """Lower case; '1,000' -> '1000'."""
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", (text or "").lower())


def numbers(text):
    return {float(n) for n in NUM.findall(norm(text))}


def world_partitions(world):
    out = set()

    def walk(o, key=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
        elif isinstance(o, str) and "partition" in key.lower():
            out.add(o.lower())
    walk(world or {})
    return out


def calls_and_results(messages):
    """[(name, args_raw, result_text or None)] in order; tool messages pair with calls in order."""
    calls, results = [], []
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                f = tc.get("function", {})
                calls.append((f.get("name"), f.get("arguments")))
        elif m.get("role") == "tool":
            results.append(m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content")))
    return [(n, a, results[i] if i < len(results) else None) for i, (n, a) in enumerate(calls)]


def succeeded(result):
    if result is None:
        return False
    try:
        r = json.loads(result)
    except (TypeError, ValueError):
        r = None
    if isinstance(r, dict):
        if r.get("error") or r.get("ok") is False or r.get("success") is False or r.get("test_only"):
            return False
        return True
    return not re.search(r"\b(error|failed|denied|invalid|not allowed|refused)\b", result or "", re.I)


def verify(row):
    msgs = row.get("messages") or []
    task = row.get("task") or {}
    reasons = []
    calls = calls_and_results(msgs)

    # 1 schema
    for name, raw, _ in calls:
        if name not in SCHEMAS:
            reasons.append(f"schema: unknown tool '{name}'")
            continue
        try:
            args = json.loads(raw or "{}") if isinstance(raw, str) else raw
        except ValueError:
            reasons.append(f"schema: {name} arguments do not parse")
            continue
        reasons += [f"schema: {name} {e}" for e in check_schema(args, SCHEMAS[name])]

    # 2 leak
    names = re.compile(r"\b(" + "|".join(map(re.escape, SCHEMAS)) + r")\b")
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        c = THINK.sub("", m.get("content") or "")
        if names.search(c):
            reasons.append(f"leak: tool name '{names.search(c).group(1)}' in text")
        if "<tool_call>" in c or re.search(r"\{\s*\"(name|arguments)\"\s*:", c):
            reasons.append("leak: tool call block in text")

    # 4 final
    last = msgs[-1] if msgs else {}
    answer = THINK.sub("", last.get("content") or "").strip() if last.get("role") == "assistant" else ""
    if len(calls) > MAX_CALLS:
        reasons.append(f"final: {len(calls)} tool calls > {MAX_CALLS}")
    if last.get("role") != "assistant" or last.get("tool_calls") or not answer:
        reasons.append("final: no assistant text answer at the end")
    else:
        tail = re.split(r"(?<=[.!?])\s+", answer)[-1]
        if PROMISE.search(tail) or (PROMISE.search(answer) and not NUM.search(answer) and len(answer) < 200):
            reasons.append("final: promise to act without a result")

    # 3 grounding (final answer only)
    if answer:
        src = " ".join([m.get("content") or "" for m in msgs if m.get("role") == "user"] +
                       [r for _, _, r in calls if r])
        nsrc, srcnums = norm(src), numbers(src)
        a = norm(answer)
        for j in sorted(set(JOB_ID.findall(a))):
            if j not in nsrc:
                reasons.append(f"grounding: job id {j}")
        parts = {p for g in PART_NEAR.findall(answer) for p in g if p and p.lower() not in PART_STOP}
        parts |= {p for p in world_partitions(row.get("world")) if re.search(r"\b" + re.escape(p) + r"\b", a)}
        for p in sorted(p.lower() for p in parts):
            if p not in nsrc:
                reasons.append(f"grounding: partition {p}")
        for n, unit in UNIT_NUM.findall(a):
            if float(n) not in srcnums:
                reasons.append(f"grounding: number {n} {unit}")

    # 5 coverage
    used = [n for n, _, _ in calls]
    if task.get("needs_tool") and not used:
        reasons.append("coverage: needs_tool but no tool call")
    if task.get("multi_step") and len(set(used)) < 2:
        reasons.append("coverage: multi_step but < 2 distinct tools")
    exp = set(task.get("expected_tools") or [])
    if exp and len(exp & set(used)) / len(exp) < 0.5:
        reasons.append("coverage: expected_tools recall < 0.5")

    # 6 action
    ask = f"{row.get('family', '')} {task.get('family', '')} {task.get('user_request') or row.get('question') or ''}"
    for n in set(used) & set(ACTIONS):
        if not re.search(ACTIONS[n][0], ask, re.I):
            reasons.append(f"action: {n} not asked for")

    # 7 claim
    for n, (_, claim) in ACTIONS.items():
        for mt in re.finditer(claim, answer, re.I):
            before = answer[max(0, mt.start() - 20):mt.start()].lower()
            if re.search(r"\b(not|n't|never|no longer|be|been|get)\s+(\w+\s+)?$", before):
                continue  # negation or a future / conditional form
            if not any(cn == n and succeeded(r) for cn, _, r in calls):
                reasons.append(f"claim: '{mt.group(0)}' without a successful {n}")
                break

    reasons = list(dict.fromkeys(reasons))
    out = dict(row, **{"pass": not reasons})
    out["fail_reasons"] = reasons
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.exit(__doc__)
    fam, why = defaultdict(lambda: [0, 0]), Counter()
    with open(argv[0], encoding="utf-8") as fi, open(argv[1], "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            r = verify(json.loads(line))
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            f = fam[r.get("family") or "?"]
            f[0] += 1
            f[1] += r["pass"]
            why.update({x.split(":")[0] for x in r["fail_reasons"]})
    tot = [sum(v[0] for v in fam.values()), sum(v[1] for v in fam.values())]
    print(f"{'family':<28}{'n':>6}{'pass':>6}{'rate':>8}")
    for k, (n, p) in sorted(fam.items()) + [("ALL", tot)]:
        print(f"{k:<28}{n:>6}{p:>6}{(p / n if n else 0):>8.1%}")
    print(f"\n{'fail reason':<28}{'rows':>6}")
    for k, c in why.most_common():
        print(f"{k:<28}{c:>6}")
    return 0


if __name__ == "__main__":
    main()
