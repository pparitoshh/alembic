"""Tool-call checks, APIGen-style (format -> execution; the judge does semantics) plus eval scoring.

- `check_trace`: verify a generated training transcript: every call valid and executable, every
  job id grounded (taken from the question or an earlier tool result, never invented), and the
  calling decision fits the question's mode.
- `score_tool_item`: score a student's raw output on a tool eval item: when-to-call decision,
  schema validity, grounding, BFCL-style AST match against the expected call, and execution.
"""

import json
import re

from .records import final_answer, messages
from .job_status_guard import (WORKFLOW_VERSION, JobStatusPolicy, blocked_batch,
                               clarification_response, guard_result, response_issues, target_error)
from .tools import CATALOG_VERSION, MOCK_VERSION, ToolError, execute, mock_session, parse_arguments, validate_call

# what the assistant must do for each question mode. "ask" has no fixed decision: asking back and
# looking the job up (list_queue, then calls on the ids it returned) are both fine; grounding
# rejects the bad case, a call on an id the user never gave.
TRAINING_TRACE_POLICY_VERSION = "explicit-job-targets-v3"
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
    return _run_call(call)[1]


def _run_call(call):
    """Execute exactly once; action results are also the evidence used for replay."""
    if call.get("name") is None:
        return None, "unparseable tool call"
    try:
        result = execute(call["name"], call["arguments"])
    except ToolError as e:
        return None, str(e)
    if call["name"] == "submit_job" and "error" in result:
        return None, f"submit_job: {result['error']}"
    return result, None


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


def ungrounded_partition(call: dict, context: str) -> list[str]:
    """A supplied partition filter needs user/result evidence; an omitted optional filter is safe."""
    try:
        partition = parse_arguments(call.get("arguments", {})).get("partition")
    except ToolError:
        return []
    if not isinstance(partition, str):
        return []
    return [] if re.search(rf"(?<![\w-]){re.escape(partition)}(?![\w-])", context) else [partition]


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate result field: {key}")
        result[key] = value
    return result


def check_trace(row: dict, valid_flags=None) -> dict:
    catalog = row.get("mock_job_ids")
    try:
        if ("mock_job_ids" in row or "mock_catalog_version" in row) and (
                catalog is None or row.get("mock_catalog_version") != CATALOG_VERSION):
            raise ValueError("closed mock catalog requires its explicit supported version and job_ids")
        with mock_session(valid_flags, job_ids=catalog):
            return _check_trace(row)
    except (ValueError, TypeError) as exc:
        # Malformed experiment metadata is a per-record hard failure, not an aborted run.
        return {"n_calls":0,"call_errors":[],"trace_errors":[str(exc)],"result_errors":[],
                "ungrounded_ids":[],"ungrounded_partitions":[],"decision":"invalid",
                "decision_ok":False,"passed":False}


def _check_trace(row: dict) -> dict:
    """Check complete ordered traces against this version of the deterministic mocks.

    Stored results are evidence only after matching replay. This does not judge whether the
    final answer interprets those results correctly, or validate results from a real cluster.
    """
    context, calls, ungrounded, partitions = "", [], [], []
    question, observed_results = "", []
    errors, trace_errors, result_errors, pending = [], [], [], []
    policy_errors, expected_events = [], []
    guard_triggered = False
    policy = JobStatusPolicy(**row['tool_policy']) if row.get('tool_policy') is not None else None
    if row.get('workflow_version') == WORKFLOW_VERSION and policy is None:
        trace_errors.append('current job-status workflow requires its deployment-visible tool_policy')
    if row.get("mock_version", MOCK_VERSION) != MOCK_VERSION:
        trace_errors.append(f"mock version {row['mock_version']!r} requires its archived implementation; current: {MOCK_VERSION}")
    transcript = messages(row)
    if not isinstance(transcript, list):
        trace_errors.append("messages must be a list")
        transcript = []
    expected_role = "user"
    for i, m in enumerate(transcript):
        if not isinstance(m, dict) or m.get("role") != expected_role:
            trace_errors.append(f"turn {i}: expected {expected_role}")
            continue
        role = m["role"]
        if role == "user":
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                trace_errors.append(f"turn {i}: empty or invalid question")
            context = m.get("content") if isinstance(m.get("content"), str) else ""
            question = context
            expected_role = "assistant"
        elif role == "assistant":
            if guard_triggered:
                if (m.get('content') != clarification_response(question) or
                        m.get('origin') != 'runtime_guard' or m.get('tool_calls') or
                        i != len(transcript) - 1):
                    trace_errors.append('guard intervention must terminate with the exact application clarification and origin')
                expected_role = 'end of transcript'
                continue  # never replay additional calls after the application terminated the conversation
            raw_calls = m.get("tool_calls")
            if raw_calls is None:
                raw_calls = []
            if not isinstance(raw_calls, list):
                trace_errors.append(f"turn {i}: tool_calls must be a list")
                continue
            clean_calls = [raw.get('function', {}) if isinstance(raw, dict) else {} for raw in raw_calls]
            denied = blocked_batch(clean_calls, question, observed_results, policy) if policy and raw_calls else []
            guard_triggered = bool(denied)
            for call_index, raw in enumerate(raw_calls):
                f = raw.get("function") if isinstance(raw, dict) else None
                if not isinstance(f, dict) or not isinstance(f.get("name"), str):
                    errors.append(f"turn {i}: malformed tool call")
                    pending.append(None)
                    continue
                c = {"name": f["name"], "arguments": f.get("arguments", {})}
                calls.append(c)
                if denied:
                    reason = denied[call_index]
                    policy_errors.append(reason)
                    pending.append(guard_result(reason, policy))
                    expected_events.append({'turn':i,'attempted_call':c,'executed':False,'blocked':True,'reason':reason})
                    continue
                ungrounded += ungrounded_ids(c, context)
                partitions += ungrounded_partition(c, context)
                target_issue = target_error(c, question, observed_results)
                if target_issue:
                    job_id = str(parse_arguments(c["arguments"])["job_id"])
                    if job_id not in ungrounded:
                        ungrounded.append(job_id)
                    pending.append(None)
                    continue
                result, error = _run_call(c)
                if error:
                    errors.append(error)
                    pending.append(None)
                else:
                    pending.append(result)
                if policy:
                    event = {'turn':i,'attempted_call':c,'executed':not bool(error),'blocked':False}
                    if error:
                        event['error'] = error
                    expected_events.append(event)
            expected_role = "tool" if raw_calls else "end of transcript"
        else:  # results follow calls in order in the record schema (no call IDs)
            expected = pending.pop(0)
            try:
                actual = json.loads(m["content"], object_pairs_hook=_unique_fields)
                # JSON comparison retains boolean/number distinctions; formatting/order do not matter.
                matches = json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
            except (KeyError, TypeError, ValueError):
                matches = False
            if expected is not None and not matches:
                result_errors.append(f"turn {i}: stored result differs from deterministic mock replay")
            elif expected is not None:
                context += "\n" + json.dumps(expected)
                observed_results.append(expected)
            expected_role = "tool" if pending else "assistant"
    if pending:
        trace_errors.append(f"missing {len(pending)} tool result(s)")
    if not final_answer(row).strip():
        trace_errors.append("missing terminal assistant answer")
    workflow_errors = response_issues(row, final_answer(row))
    if any(m.get('origin') == 'runtime_guard' for m in transcript if isinstance(m, dict)) and not policy_errors:
        trace_errors.append('runtime guard response without a replayed guard intervention')
    if policy:
        audit = row.get('runtime_audit', {})
        origin = 'runtime_guard' if guard_triggered else 'teacher'
        delivered = clarification_response(question) if guard_triggered else final_answer(row)
        expected_audit = {'version':'tool-runtime-audit-v1','policy':policy.as_dict(),'events':expected_events,
                          'response_origin':origin,'user_response':delivered,
                          'model_final_answer':None if guard_triggered else final_answer(row)}
        if audit != expected_audit:
            trace_errors.append('runtime audit does not match replayed attempts, execution and delivered response')
    decision = "call" if calls else "no_call"
    ok = decision == EXPECTED_DECISION.get(row.get("mode", ""), decision)
    return {
        "workflow_errors": workflow_errors,
        "application_policy_errors": policy_errors,
        "model_policy_compliant": not policy_errors,
        "guard_blocked_calls": sum(e['blocked'] for e in expected_events),
        "executed_calls": sum(e['executed'] for e in expected_events) if policy else None,
        "runtime_enforcement_consistent": not errors and not trace_errors and not result_errors,
        "policy_version": TRAINING_TRACE_POLICY_VERSION,
        "n_calls": len(calls),
        "call_errors": errors,
        "trace_errors": trace_errors,
        "result_errors": result_errors,
        "ungrounded_ids": ungrounded,
        "ungrounded_partitions": partitions,
        "decision": decision,
        "decision_ok": ok,
        "passed": not errors and not trace_errors and not result_errors and not ungrounded and not partitions and not workflow_errors and not policy_errors and ok,
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
