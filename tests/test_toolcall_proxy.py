"""Tool-call repair proxy: the repair of fake tool calls written as text, and the proxy end to end
against a fake upstream (no Ollama, no network outside 127.0.0.1)."""
import importlib.util
import json
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dsh_toolcall_proxy", ROOT / "research" / "dsh_harness" / "toolcall_proxy.py")
proxy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proxy)

# the conftest hermetic fixture blocks every socket.connect; this module only talks to the two
# 127.0.0.1 servers it starts itself, so it restores the real connect for its own tests
_REAL_CONNECT = socket.socket.connect
_REAL_CREATE_CONNECTION = socket.create_connection


@pytest.fixture(autouse=True)
def loopback(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _REAL_CONNECT)
    monkeypatch.setattr(socket, "create_connection", _REAL_CREATE_CONNECTION)


def tool(name, properties=None, required=()):
    return {"type": "function", "function": {"name": name,
            "parameters": {"type": "object", "properties": properties or {}, "required": list(required)}}}


TOOLS = [tool("gpu_availability", {"partition": {"type": "string"}}),
         tool("job_status", {"job_id": {"type": "string"}}, required=["job_id"]),
         tool("partition_info"),
         tool("submit_job", {"script": {"type": "string"}}, required=["script"])]


def reply(content, tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}]}


def assert_repaired(out, name, arguments, content):
    call = out["choices"][0]["message"]["tool_calls"][0]
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert call["type"] == "function" and call["function"]["name"] == name
    assert json.loads(call["function"]["arguments"]) == arguments
    assert call["id"].startswith("call_repair_") and len(call["id"]) == len("call_repair_") + 8
    assert out["choices"][0]["message"]["content"] == content


def test_json_block_repaired():
    content = 'Let me check.\n```json\n{"name": "mcp__hpc__gpu_availability", ' \
              '"arguments": {"partition": "x", "bogus": 1}}\n```'
    assert_repaired(proxy.repair(reply(content), TOOLS), "gpu_availability", {"partition": "x"}, "Let me check.")


def test_json_list_and_string_arguments():
    list_reply = proxy.repair(reply('{"name": "job_status", "arguments": [{"job_id": "42"}]}'), TOOLS)
    assert_repaired(list_reply, "job_status", {"job_id": "42"}, None)
    string_reply = proxy.repair(reply('{"name": "job_status", "arguments": "{\\"job_id\\": \\"42\\"}"}'), TOOLS)
    assert_repaired(string_reply, "job_status", {"job_id": "42"}, None)


def test_bare_name_in_bash_block():
    content = "Sure.\n```bash\nmcp__hpc__partition_info\n```"
    assert_repaired(proxy.repair(reply(content), TOOLS), "partition_info", {}, "Sure.")
    pairs = proxy.repair(reply("```bash\nmcp__hpc__partition_info partition=x days=3\n```"), TOOLS)
    assert_repaired(pairs, "partition_info", {}, None)


def test_action_tool_ignored():
    content = '{"name": "submit_job", "arguments": {"script": "#!/bin/bash"}}'
    out = proxy.repair(reply(content), TOOLS)
    assert "tool_calls" not in out["choices"][0]["message"]
    assert out["choices"][0]["message"]["content"] == content
    assert out["choices"][0]["finish_reason"] == "stop"


def test_missing_required_argument_ignored():
    content = '{"name": "job_status", "arguments": {}}'
    out = proxy.repair(reply(content), TOOLS)
    assert "tool_calls" not in out["choices"][0]["message"] and out["choices"][0]["message"]["content"] == content


def test_unknown_property_dropped():
    content = '{"name": "job_status", "arguments": {"job_id": "42", "verbose": true}}'
    assert_repaired(proxy.repair(reply(content), TOOLS), "job_status", {"job_id": "42"}, None)


def test_reply_with_real_tool_calls_unchanged():
    calls = [{"id": "call_1", "type": "function",
              "function": {"name": "job_status", "arguments": '{"job_id": "42"}'}}]
    original = reply(None, tool_calls=calls, finish="tool_calls")
    out = proxy.repair(original, TOOLS)
    assert out is original and out["choices"][0]["message"]["tool_calls"] == calls
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_text_before_the_call_is_kept():
    content = "I check the job now.\n\n{\"name\": \"job_status\", \"arguments\": {\"job_id\": \"7\"}}\nDone."
    assert_repaired(proxy.repair(reply(content), TOOLS), "job_status", {"job_id": "7"}, "I check the job now.")


def test_prose_mention_not_repaired():
    content = "gpu_availability shows 86 free GPUs right now, so I would submit there."
    out = proxy.repair(reply(content), TOOLS)
    assert "tool_calls" not in out["choices"][0]["message"] and out["choices"][0]["message"]["content"] == content


def test_inline_backtick_name_repaired():
    content = "Sure: `mcp__hpc__partition_info` tells you the partitions."
    assert_repaired(proxy.repair(reply(content), TOOLS), "partition_info", {}, "Sure:")


def test_duplicate_of_an_earlier_call_not_repaired():
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "gpu_availability", "arguments": '{"partition": "x"}'}}]}]
    content = '{"name": "gpu_availability", "arguments": {"partition": "x"}}'
    out = proxy.repair(reply(content), TOOLS, messages)
    assert "tool_calls" not in out["choices"][0]["message"] and out["choices"][0]["message"]["content"] == content


def test_next_valid_candidate_used_after_a_duplicate():
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "gpu_availability", "arguments": '{"partition": "x"}'}}]}]
    content = '{"name": "gpu_availability", "arguments": {"partition": "x"}}\n' \
              '{"name": "job_status", "arguments": {"job_id": "42"}}'
    assert_repaired(proxy.repair(reply(content), TOOLS, messages), "job_status", {"job_id": "42"}, None)


class FakeUpstream(BaseHTTPRequestHandler):
    reply = {}
    bodies = []

    def do_POST(self):
        self.bodies.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0))))
        payload = json.dumps(self.reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        payload = json.dumps({"object": "list", "data": [{"id": "m"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def servers():
    FakeUpstream.bodies = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    proxied = proxy.serve(0, f"http://127.0.0.1:{upstream.server_address[1]}")
    threading.Thread(target=proxied.serve_forever, daemon=True).start()
    yield proxied
    proxied.shutdown()
    upstream.shutdown()


def post(port, body):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.headers, response.read()


def test_end_to_end_stream_false(servers):
    FakeUpstream.reply = reply('{"name": "mcp__hpc__gpu_availability", "arguments": {"partition": "x"}}')
    status, headers, raw = post(servers.server_address[1], {"model": "m", "stream": False, "tools": TOOLS,
                                                           "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and headers.get("Content-Type") == "application/json"
    assert headers.get(proxy.REPAIR_HEADER) == "1"
    assert_repaired(json.loads(raw), "gpu_availability", {"partition": "x"}, None)
    assert FakeUpstream.bodies[-1]["stream"] is False and FakeUpstream.bodies[-1]["tools"] == TOOLS


def test_end_to_end_stream_true(servers):
    FakeUpstream.reply = reply("Checking now.\n```bash\nmcp__hpc__partition_info\n```", finish="stop")
    body = {"model": "m", "stream": True, "stream_options": {"include_usage": True}, "tools": TOOLS,
            "messages": [{"role": "user", "content": "hi"}]}
    status, headers, raw = post(servers.server_address[1], body)
    assert status == 200 and headers.get("Content-Type") == "text/event-stream"
    assert headers.get(proxy.REPAIR_HEADER) == "1"
    parts = [part for part in raw.decode().split("\n\n") if part]
    assert parts[-1] == "data: [DONE]"
    chunks = [json.loads(part[len("data: "):]) for part in parts[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "Checking now."}
    call = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0 and call["type"] == "function"
    assert call["function"]["name"] == "partition_info" and json.loads(call["function"]["arguments"]) == {}
    assert chunks[2]["choices"][0]["finish_reason"] == "tool_calls" and chunks[2]["choices"][0]["delta"] == {}
    assert chunks[3]["choices"] == [] and chunks[3]["usage"] == FakeUpstream.reply.get("usage")
    assert FakeUpstream.bodies[-1]["stream"] is False


def test_passthrough_forwards_other_paths(servers):
    request = urllib.request.Request(f"http://127.0.0.1:{servers.server_address[1]}/v1/models")
    with urllib.request.urlopen(request, timeout=10) as response:
        assert json.load(response) == {"object": "list", "data": [{"id": "m"}]}
