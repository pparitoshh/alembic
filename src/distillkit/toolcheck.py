"""Tool-call checks, APIGen-style (format -> execution; the judge does semantics) plus eval scoring.

- `check_trace`: verify a generated training transcript: every call valid and executable, every
  job id grounded (taken from the question or an earlier tool result, never invented), and the
  calling decision fits the question's mode.
- `score_tool_item`: score a student's raw output on a tool eval item: when-to-call decision,
  schema validity, grounding, BFCL-style AST match against the expected call, and execution.
"""

import json
import re

from .records import messages
from .tools import ToolError, execute, parse_arguments, validate_call

# what the assistant must do for each question mode. "ask" has no fixed decision: asking back and
# looking the job up (list_queue, then calls on the ids it returned) are both fine; grounding
# rejects the bad case, a call on an id the user never gave.
EXPECTED_DECISION = {"call": "call", "none": "no_call"}

_HERMES = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_hermes(text: str) -> tuple[str, list[dict]]:
    """Split raw model output into (text outside tool calls, calls). An unparseable call body
    is kept as {"name": None, "raw": ...} so it counts as an invalid call, not as no call."""
    calls = []
    for body in _HERMES.findall(text):
        try:
            obj = json.loads(body)
            calls.append({"name": obj.get("name"), "arguments": obj.get("arguments", {})} if isinstance(obj, dict) else {"name": None, "raw": body})
        except json.JSONDecodeError:
            calls.append({"name": None, "raw": body})
    return _HERMES.sub("", text).strip(), calls


def message_calls(msg: dict) -> list[dict]:
    """[{"name", "arguments"}] from an assistant message in OpenAI/record format."""
    return [{"name": c["function"]["name"], "arguments": c["function"].get("arguments", {})} for c in msg.get("tool_calls") or []]


def call_error(call: dict) -> str | None:
    """None if the call is well-formed, valid against its schema and runs on the mock cluster."""
    if call.get("name") is None:
        return "unparseable tool call"
    try:
        result = execute(call["name"], call["arguments"])
    except ToolError as e:
        return str(e)
    if call["name"] == "submit_job" and "error" in result:
        return f"submit_job: {result['error']}"  # the assistant wrote a script sbatch would reject
    return None


def ungrounded_ids(call: dict, context: str) -> list[str]:
    """job_id arguments that appear nowhere in `context` (the question plus earlier tool results).
    An array task id like 4242_7 is grounded if 4242_7 or its job id 4242 appears."""
    try:
        job_id = parse_arguments(call.get("arguments", {})).get("job_id")
    except ToolError:
        return []  # call_error reports it
    if job_id is None:
        return []
    found = lambda x: re.search(rf"(?<!\d){re.escape(str(x))}(?!\d)", context) is not None
    return [] if found(job_id) or found(str(job_id).split("_")[0]) else [str(job_id)]


def check_trace(row: dict) -> dict:
    """Verify a transcript. `row["mode"]` (call/ask/none) says whether a call was required."""
    context, calls, ungrounded = "", [], []
    for m in messages(row):
        if m["role"] in ("user", "tool"):
            context += "\n" + (m.get("content") or "")
        elif m["role"] == "assistant":
            for c in message_calls(m):
                calls.append(c)
                ungrounded += ungrounded_ids(c, context)
    errors = [e for c in calls if (e := call_error(c))]
    decision = "call" if calls else "no_call"
    ok = decision == EXPECTED_DECISION.get(row.get("mode", ""), decision)
    return {
        "n_calls": len(calls),
        "call_errors": errors,
        "ungrounded_ids": ungrounded,
        "decision": decision,
        "decision_ok": ok,
        "passed": not errors and not ungrounded and ok,
    }


def _norm(v):
    return v.strip().lower() if isinstance(v, str) else v


def ast_match(call: dict, expected: dict | list[dict]) -> bool:
    """BFCL-style match. `expected` = {"name": ..., "arguments": {arg: [allowed values]}}, or a list of
    such alternatives. Among the allowed values, "" means the argument may be omitted and "*" means
    any value. Extra args fail."""
    if isinstance(expected, list):
        return any(ast_match(call, e) for e in expected)
    if call.get("name") != expected["name"]:
        return False
    try:
        args = parse_arguments(call.get("arguments", {}))
    except ToolError:
        return False
    allowed = expected.get("arguments", {})
    if any(a not in allowed for a in args):
        return False
    for a, values in allowed.items():
        if a in args:
            if "*" not in values and _norm(args[a]) not in [_norm(v) for v in values]:
                return False
        elif "" not in values:
            return False
    return True


def score_tool_item(output: str, item: dict) -> dict:
    """Score one eval item {"expect": "call"|"no_call", "expected_call"?: {...}} against raw output.

    Only the first call is scored: eval items ask for one call."""
    _, calls = parse_hermes(output)
    decision = "call" if calls else "no_call"
    grounded = not any(ungrounded_ids(c, item["question"]) for c in calls if c.get("name"))
    # "allow_lookup" items (a detail is missing) also accept a grounded lookup instead of asking back
    lookup_ok = decision == "call" and item.get("allow_lookup", False) and grounded
    res = {"decision": decision, "decision_ok": decision == item["expect"] or lookup_ok}
    if calls:
        res["grounded"] = grounded
        first = calls[0]
        res["valid"] = first.get("name") is not None and _valid(first)
        res["exec_ok"] = call_error(first) is None
        if item.get("expected_call"):
            res["ast_ok"] = ast_match(first, item["expected_call"])
    return res


def _valid(call: dict) -> bool:
    try:
        validate_call(call["name"], parse_arguments(call["arguments"]))
        return True
    except ToolError:
        return False
