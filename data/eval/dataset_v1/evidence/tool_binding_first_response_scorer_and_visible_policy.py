"""Versioned first-response tool evaluation, not multi-turn task completion."""
import hashlib
import json

from .tools import mock_session, parse_arguments, ToolError

PROTOCOL = "tool-first-response-v2"
POLICY = "hpc-tools-clarify-first-v1"
DISCOVERY_POLICY = "hpc-tools-readonly-discovery-v1"
POLICY_TEXT = """Tool application policy (hpc-tools-clarify-first-v1):
Use the available tools for explicit live-information or action requests. Before a
job-specific lookup or action, require an explicitly identified job in the visible
conversation. If its identity or selection is missing or ambiguous, ask for it
before using tools; do not guess from resource numbers, names or example IDs.
Explicit queue and resource queries do not require a job ID. A queue query supplies
candidates, not permission to act on an arbitrary candidate. Use returned IDs only
after their results are available, never from another simultaneous call.
Perform cancellation/submission only when explicitly requested for the identified
target. When asked only to validate a supplied script, preserve its text and use
submit_job with test_only=true. General explanations and script-writing requests
need no live tool call. Do not claim an action or observation without its result.
This evaluation records your first response only. Make at most ONE tool call in that
response; wait for its result before proposing another. Tool execution uses isolated mocks."""
DISCOVERY_TEXT = POLICY_TEXT.replace(POLICY, DISCOVERY_POLICY).replace(
    'If its identity or selection is missing or ambiguous, ask for it\nbefore using tools; do not guess from resource numbers, names or example IDs.',
    'If its identity or selection is missing or ambiguous, either ask for it or make one\n'
    'read-only list_queue call with no arguments to discover candidates. Do not guess a\n'
    'target, filters, resource numbers, names or example IDs; a returned candidate still\n'
    'requires an explicit user selection rule before job-specific lookup or action.')


def system_prompt(cfg, has_tools):
    text = cfg.student.system_prompt
    if has_tools and cfg.eval.tool_protocol == PROTOCOL:
        policy = cfg.eval.tool_application_policy
        return text + "\n\n" + (POLICY_TEXT if policy == POLICY else DISCOVERY_TEXT)
    return text


def answer_binding(cfg, questions, tools, *, backend='hf'):
    """Answer-side inputs only: reference/scorer changes do not invalidate answers."""
    value = {"model": cfg.student.model, "max_new_tokens": cfg.eval.max_new_tokens,
             "protocol": cfg.eval.tool_protocol, "questions": questions, "tools": tools,
             "systems": [system_prompt(cfg, bool(t)) for t in tools], "backend":backend,
             "inference": ({"dtype":cfg.train.dtype} if backend == 'hf' else
                 {"num_ctx":cfg.export.num_ctx, "ngl":cfg.eval.gguf_ngl,
                  "threads":cfg.export.bench_threads, "llama_cpp":str(cfg.export.llama_cpp)})}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _script(value):
    # Only transport newlines are insignificant. Case, spaces and commands matter.
    return value.replace("\r\n", "\n").removesuffix("\n") if isinstance(value, str) else value


def _argument_object(value):
    # Do not let legacy falsy-value coercion turn [], null, false or "" into {}.
    if isinstance(value, dict):
        return True
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return isinstance(json.loads(value), dict)
    except ValueError:
        return False


def _match(call, expected):
    if isinstance(expected, list):
        return any(_match(call, e) for e in expected)
    if call.get("name") != expected["name"]:
        return False
    try:
        args = parse_arguments(call.get("arguments", {}))
    except ToolError:
        return False
    allowed = expected.get("arguments", {})
    if set(args) - set(allowed):
        return False
    for name, values in allowed.items():
        if name not in args:
            if "" not in values:
                return False
        elif name == "script":
            if "*" in values or _script(args[name]) not in [_script(v) for v in values]:
                return False
        elif not any(type(args[name]) is type(v) and args[name] == v for v in values):
            return False
    return True


def score(output, item):
    from .toolcheck import parse_hermes, _valid, ungrounded_ids, ungrounded_partition, call_error
    _, calls = parse_hermes(output)
    # A dangling delimiter cannot silently become a successful no-call response.
    malformed = output.count("<tool_call>") != len(calls) or output.count("</tool_call>") != len(calls)
    decision = "call" if calls or malformed else "no_call"
    policy = item.get("application_policy")
    if policy not in {POLICY, DISCOVERY_POLICY}:
        raise ValueError("v2 tool item requires a declared application policy")
    valid = (len(calls) == 1 and not malformed and
             isinstance(calls[0].get('name'), str) and
             _argument_object(calls[0].get('arguments', {})) and _valid(calls[0]))
    grounded = all(not ungrounded_ids(c, item["question"]) and
                   not ungrounded_partition(c, item["question"]) for c in calls)
    # The optional permissive policy has exactly one ID-free read-only discovery
    # alternative. Hidden labels never change a deployed assistant's prompt/policy.
    lookup_ok = (policy == DISCOVERY_POLICY and valid and grounded and
                 bool(item.get("allow_lookup")) and _match(calls[0], {"name":"list_queue", "arguments":{}}))
    decision_ok = bool(output.strip()) and (decision == item["expect"] or lookup_ok) and not malformed
    if calls and len(calls) != 1:
        decision_ok = False
    result = {"protocol":PROTOCOL, "scope":"first-response decision/arguments/mock execution only",
              "decision":decision, "decision_ok":decision_ok, "n_calls":len(calls),
              "allowed_discovery":lookup_ok, "malformed_call":malformed,
              "response_quality_assessed":False, "task_completion_assessed":False}
    if calls or malformed:
        result.update(valid=valid, grounded=grounded, exec_ok=False)
        if valid:
            with mock_session():
                result["exec_ok"] = call_error(calls[0]) is None
        if item.get("expected_call"):
            result["ast_ok"] = valid and _match(calls[0], item["expected_call"])
    result["first_response_pass"] = decision_ok and (decision == "no_call" or
        (valid and grounded and result["exec_ok"] and (lookup_ok or result.get("ast_ok", False))))
    return result
