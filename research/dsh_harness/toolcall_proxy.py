"""Repair proxy between dsh and Ollama: turns fake tool calls written as text back into structured
tool_calls, so dsh runs the tool instead of showing the text to the user.

Problem: the HPC model sometimes writes a tool call as text in the reply content, for example a JSON
block {"name": "mcp__hpc__gpu_availability", "arguments": {"partition": "x"}} or a bash block with
`mcp__hpc__partition_info`. dsh then runs nothing. The proxy sits between dsh and Ollama, reads the
full reply of every /v1/chat/completions call and repairs it (see `repair`): it finds the first fake
call for a tool that dsh actually offered (never a job-changing action), checks the arguments
against the tool schema and rewrites the reply into one structured tool call. A bare tool name
counts only inside a code fence or inline backticks, so a prose mention stays an answer, and a
candidate that repeats an earlier assistant tool_calls entry of the same request is skipped, so a
finished loop does not start again.

Standard library only (http.server, urllib, json, re). Logs go to stderr only. The proxy listens on
127.0.0.1 only.

Run: PYTHONPATH=src python research/dsh_harness/toolcall_proxy.py [--port 11435] [--upstream http://127.0.0.1:11434]
Then point dsh at http://127.0.0.1:<port>/v1.
"""
import argparse
import json
import re
import secrets
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Job-changing actions are never repaired: a fake action call in the text must not become a real one
ACTION_SUFFIXES = ("submit_job", "cancel_job", "update_job", "hold_job", "release_job", "requeue_job")
REPAIR_HEADER = "X-Toolcall-Repair"
UPSTREAM_TIMEOUT = 600  # seconds; a slow local model must not time out mid-generation


def log(*parts):
    print("[toolcall-proxy]", *parts, file=sys.stderr, flush=True)


def allowed_tools(tools):
    """The tool names the repair may use: the function names of the request tool list minus the
    job-changing actions (any name ending in one of ACTION_SUFFIXES)."""
    names = []
    for tool in tools or []:
        name = (tool or {}).get("function", {}).get("name")
        if isinstance(name, str) and name and not any(name.endswith(s) for s in ACTION_SUFFIXES):
            names.append(name)
    return names


def _schemas(tools):
    """Map of tool name -> its JSON schema (the function parameters of the request tool list)."""
    out = {}
    for tool in tools or []:
        function = (tool or {}).get("function") or {}
        if isinstance(function.get("name"), str):
            out[function["name"]] = function.get("parameters") or {}
    return out


def _json_objects(text):
    """Every JSON object in the text, in order, as (start, value) pairs. Positions that do not start
    a complete object are skipped, so the scan works inside ```json fences too."""
    decoder = json.JSONDecoder()
    out = []
    for pos, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text, pos)
        except ValueError:
            continue
        if isinstance(value, dict):
            out.append((pos, value))
    return out


def _json_call(obj, allowed):
    """A JSON object that names an allowed tool with arguments: (canonical name, arguments) or None.
    The model may write the MCP-prefixed name (mcp__hpc__gpu_availability), so the last __ segment
    counts too. The arguments may be an object, a list with one object or a JSON string."""
    name = obj.get("name")
    if not isinstance(name, str):
        return None
    exact = [a for a in allowed if name == a]
    prefixed = [a for a in allowed if name.rsplit("__", 1)[-1] == a]
    canonical = (exact or prefixed or [None])[0]
    if canonical is None:
        return None
    raw = obj.get("arguments")
    if isinstance(raw, list):
        raw = raw[0] if len(raw) == 1 and isinstance(raw[0], dict) else None
    elif isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    return (canonical, raw) if isinstance(raw, dict) else None


def _clean_args(name, args, schemas):
    """The arguments checked against the tool schema: only known properties stay, and a missing
    required property rejects the candidate (None)."""
    schema = schemas.get(name) or {}
    properties = schema.get("properties")
    if isinstance(properties, dict):
        args = {key: value for key, value in args.items() if key in properties}
    if any(key not in args for key in schema.get("required") or []):
        return None
    return args


def _bare_name_pattern(name):
    """Pattern for a bare tool name with its optional MCP prefix (mcp__hpc__partition_info), then
    optionally key=value pairs on the same line. The lookbehind allows an underscore, so the
    prefixed form matches too."""
    return re.compile(rf"(?<![A-Za-z0-9])(?:[A-Za-z0-9_-]+__)*{re.escape(name)}"
                      rf"(?:[ \t]+[^\s=]+=[^\s]+)*(?![A-Za-z0-9])")


def _with_fence(text, start):
    """The start moved back over an opening code fence (```json, ```bash, ```), so the fence does not
    stay behind as content of the repaired reply."""
    fence = re.search(r"```[A-Za-z0-9_-]*[ \t]*\n$", text[:start])
    return fence.start() if fence else start


def _key(args):
    """The arguments as sorted JSON, so two argument dicts compare equal by content."""
    return json.dumps(args, sort_keys=True)


def _prior_calls(messages):
    """The (tool name, arguments key) of the assistant tool_calls in the request messages, for the
    duplicate check of the repair. Unparsable arguments keep their raw text as the key."""
    out = set()
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            raw = function.get("arguments")
            if isinstance(raw, dict):
                canonical = _key(raw)
            elif isinstance(raw, str):
                try:
                    canonical = _key(json.loads(raw))
                except ValueError:
                    canonical = repr(raw)
            else:
                canonical = repr(raw)
            out.add((function.get("name"), canonical))
    return out


def _code_spans(text):
    """The character spans of the fenced code blocks (a missing closing fence runs to the end) and
    the inline backtick code in the text. A bare tool name counts as a fake call only inside them."""
    spans = [(m.start(), m.end())
             for m in re.finditer(r"```[A-Za-z0-9_-]*[^\n]*\n.*?(?:```|\Z)", text, re.S)]
    spans += [(m.start(), m.end()) for m in re.finditer(r"`[^`\n]+`", text)]
    return spans


def _enclosing(pos, spans):
    """The code span that contains the position, or None."""
    for span in spans:
        if span[0] <= pos < span[1]:
            return span
    return None


def find_fake_call(text, allowed, schemas, prior=()):
    """The first valid fake tool call in the text: (name, arguments, start) or None. A JSON call wins
    over a bare name; within one form the first position with valid arguments wins. A candidate whose
    tool with the same arguments (parsed JSON) is in prior (the earlier assistant tool_calls of the
    request) is skipped, so the next valid candidate is used. A bare name counts only inside a code
    fence or inline backticks; plain prose never repairs. The start is the position of the first
    candidate of either form, so the kept content ends before the first fake call even when the
    repaired call is a later one."""
    first = None
    chosen = None
    for pos, obj in _json_objects(text):
        found = _json_call(obj, allowed)
        if found is None:
            continue
        start = _with_fence(text, pos)
        first = start if first is None else min(first, start)
        if chosen is None:
            args = _clean_args(found[0], found[1], schemas)
            if args is not None and (found[0], _key(args)) not in prior:
                chosen = (found[0], args, start)
    if chosen is None:
        spans = _code_spans(text)
        bare = []
        for name in allowed:
            for match in _bare_name_pattern(name).finditer(text):
                span = _enclosing(match.start(), spans)
                if span is None:
                    continue
                inline = text[span[0]:span[0] + 3] != "```" and span[0] + 1 == match.start()
                start = span[0] if inline else _with_fence(text, match.start())
                bare.append((start, name))
        for start, name in sorted(bare, key=lambda c: c[0]):
            first = start if first is None else min(first, start)
            if chosen is None:
                args = _clean_args(name, {}, schemas)
                if args is not None and (name, _key(args)) not in prior:
                    chosen = (name, args, start)
    return (chosen[0], chosen[1], first) if chosen else None


def repair(reply, tools, messages=None):
    """Repair one chat completion reply in place and return it. Only a reply whose first choice has
    no tool_calls and text content changes: the first fake call in the content becomes one structured
    tool call (id call_repair_<8 hex>), the content becomes the text before the first fake-call
    candidate (stripped, null if empty) and the finish_reason becomes "tool_calls". A candidate whose tool with the same
    arguments is already in an earlier assistant tool_calls of the request messages is skipped. A
    reply that already has tool_calls, an empty content or no valid fake call comes back unchanged."""
    try:
        message = reply["choices"][0]["message"]
    except (KeyError, TypeError, IndexError, AttributeError):
        return reply
    if message.get("tool_calls") or not isinstance(message.get("content"), str) or not message["content"]:
        return reply
    content = message["content"]
    allowed = allowed_tools(tools)
    prior = _prior_calls(messages) if messages else set()
    found = find_fake_call(content, allowed, _schemas(tools), prior) if allowed else None
    if found is None:
        return reply
    name, args, start = found
    message["tool_calls"] = [{"id": "call_repair_" + secrets.token_hex(4), "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args)}}]
    message["content"] = content[:start].strip() or None
    reply["choices"][0]["finish_reason"] = "tool_calls"
    log(f"repaired {name} {json.dumps(args)}")
    return reply


def _first_message(reply):
    """The message of the first choice, or an empty dict for a malformed reply."""
    try:
        return reply["choices"][0].get("message") or {}
    except (KeyError, TypeError, IndexError, AttributeError):
        return {}


def _sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


def sse_body(reply, include_usage=False):
    """One non-streamed reply as an OpenAI SSE body: a chunk with the delta role and content, one
    chunk per tool call (index, id, type "function", function name and arguments as a JSON string),
    a final chunk with the finish_reason, an optional usage chunk and [DONE]."""
    choice = (reply.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    base = {"id": reply.get("id"), "object": "chat.completion.chunk",
            "created": reply.get("created"), "model": reply.get("model")}

    def chunk(delta, finish=None):
        return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    delta = {"role": "assistant"}
    if message.get("content") is not None:
        delta["content"] = message["content"]
    parts = [_sse(chunk(delta))]
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {})
        parts.append(_sse(chunk({"tool_calls": [{"index": index, "id": call.get("id"),
                                                 "type": call.get("type") or "function",
                                                 "function": {"name": function.get("name"),
                                                              "arguments": arguments}}]})))
    parts.append(_sse(chunk({}, choice.get("finish_reason"))))
    if include_usage:
        parts.append(_sse({**base, "choices": [], "usage": reply.get("usage")}))
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = "http://127.0.0.1:11434"

    def log_message(self, fmt, *args):
        log(f"{self.address_string()} {fmt % args}")

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else None

    def _forward(self, body):
        """The request at the upstream. Returns (status, content type, payload); an unreachable
        upstream becomes a 502 with a JSON error body."""
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")} \
            if body is not None else {}
        if self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        request = urllib.request.Request(self.upstream + self.path, data=body, headers=headers,
                                         method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT) as reply:
                return reply.status, reply.headers.get("Content-Type", "application/json"), reply.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", "application/json"), e.read()
        except OSError as e:
            payload = json.dumps({"error": {"message": f"upstream {self.upstream} unreachable: {e}"}}).encode()
            return 502, "application/json", payload

    def _answer(self, status, content_type, payload, repaired=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if repaired:
            self.send_header(REPAIR_HEADER, "1")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _passthrough(self, body):
        status, content_type, payload = self._forward(body)
        self._answer(status, content_type, payload)

    def do_GET(self):
        self._passthrough(None)

    def do_POST(self):
        body = self._read_body()
        if self.path.split("?", 1)[0] != "/v1/chat/completions":
            self._passthrough(body)
            return
        try:
            request = json.loads(body)
        except (ValueError, TypeError):
            request = None
        if not isinstance(request, dict):
            self._passthrough(body)
            return
        stream = bool(request.get("stream"))
        include_usage = bool((request.get("stream_options") or {}).get("include_usage"))
        status, content_type, payload = self._forward(json.dumps({**request, "stream": False}).encode())
        if status != 200 or content_type.find("json") < 0:
            self._answer(status, content_type, payload)
            return
        try:
            reply = json.loads(payload)
        except ValueError:
            self._answer(status, content_type, payload)
            return
        had_calls = bool(_first_message(reply).get("tool_calls"))
        tools = request.get("tools") if isinstance(request.get("tools"), list) else []
        messages = request.get("messages") if isinstance(request.get("messages"), list) else None
        repair(reply, tools, messages)
        repaired = not had_calls and bool(_first_message(reply).get("tool_calls"))
        if stream:
            self._answer(200, "text/event-stream", sse_body(reply, include_usage), repaired)
        else:
            self._answer(200, "application/json", json.dumps(reply).encode(), repaired)


def serve(port, upstream):
    """The proxy server for one port (0 picks a free port) and one upstream base URL, not yet serving."""
    Proxy.upstream = upstream.rstrip("/")
    server = ThreadingHTTPServer(("127.0.0.1", port), Proxy)
    log(f"ready on http://127.0.0.1:{server.server_address[1]} -> {Proxy.upstream}")
    return server


def main():
    parser = argparse.ArgumentParser(description="Repair proxy between dsh and Ollama: fake tool "
                                                 "calls written as text become structured tool_calls.")
    parser.add_argument("--port", type=int, default=11435, help="local port to listen on (default 11435)")
    parser.add_argument("--upstream", default="http://127.0.0.1:11434", help="base URL of the model server")
    args = parser.parse_args()
    server = serve(args.port, args.upstream)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
