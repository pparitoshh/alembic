"""Tool-call validity (GOAL.md G2): does the model call a tool when it should, pick the right one, and give valid,
correct arguments? Sends the held-out tool workflows (first user turn) and data/eval/eval_tools.jsonl to an
OpenAI-compatible server started with tool parsing (vLLM --enable-auto-tool-choice --tool-call-parser hermes),
greedy, with the 6 Slurm tool schemas, and scores the first reply.

usage: tool_eval.py ENDPOINT SERVED TOOL_TEST.jsonl EVAL_TOOLS.jsonl OUT.json
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

endpoint, served, test_path, eval_path, out = sys.argv[1:6]
SYSTEM = ("You are an HPC assistant. Use supplied source excerpts for site-specific facts and tool results "
          "for live state. Name the applicable cluster and ask for missing required details. Give a complete, "
          "clear answer without inventing commands, flags or cluster defaults. Put scripts in fenced code blocks.")
test = [json.loads(l) for l in open(test_path)]
tools = test[0]['tools']
schema = {t['function']['name']: t['function']['parameters'] for t in tools}
items = []
for r in test:  # held-out factory workflows: expected = the reference's first assistant turn
    first = next(m for m in r['messages'] if m['role'] == 'assistant')
    tc = (first.get('tool_calls') or [None])[0]
    exp = None
    if tc:
        a = tc['function']['arguments']
        exp = {'name': tc['function']['name'], 'arguments': {k: [v] for k, v in (json.loads(a) if isinstance(a, str) else a).items()}}
    items.append({'set': 'heldout', 'id': r['id'], 'question': r['messages'][0]['content'], 'expect': 'call' if tc else 'no_call', 'expected_call': exp})
for r in map(json.loads, open(eval_path)):  # hand-written tool questions (draft, not human-reviewed)
    items.append({'set': 'eval_tools', 'id': r['id'], 'question': r['question'],
                  'expect': 'call' if r.get('expect') == 'call' else 'no_call', 'expected_call': r.get('expected_call')})


def ask(it):
    body = {'model': served, 'temperature': 0, 'max_tokens': 512, 'tools': tools,
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': it['question']}]}
    try:
        msg = requests.post(endpoint.rstrip('/') + '/chat/completions', json=body, timeout=300).json()['choices'][0]['message']
    except Exception as e:  # a failed request scores as a wrong reply
        return dict(it, error=repr(e), calls=[])
    calls = []
    for tc in msg.get('tool_calls') or []:
        try:
            args, ok = json.loads(tc['function']['arguments'] or '{}'), True
        except json.JSONDecodeError:
            args, ok = {}, False
        calls.append({'name': tc['function']['name'], 'arguments': args, 'json_ok': ok})
    return dict(it, calls=calls, content=(msg.get('content') or '')[:400])


with ThreadPoolExecutor(16) as pool:
    res = list(pool.map(ask, items))
json.dump({'served': served, 'results': res}, open(out + '.raw', 'w'), indent=1)  # kept even if scoring fails


def score(rs):
    n = len(rs)
    call_rs = [r for r in rs if r['expect'] == 'call']
    decision = sum((r['expect'] == 'call') == bool(r['calls']) for r in rs) / max(1, n)
    name_ok = valid = exact = 0
    for r in call_rs:
        if not r['calls']:
            continue
        c = r['calls'][0]
        exp = r['expected_call'] or {}
        exps = exp if isinstance(exp, list) else [exp]  # some eval_tools items accept several calls
        if any(c['name'] == e.get('name') for e in exps):
            name_ok += 1
        req = (schema.get(c['name']) or {}).get('required', [])
        props = (schema.get(c['name']) or {}).get('properties', {})
        if c['json_ok'] and c['name'] in schema and all(k in c['arguments'] for k in req) and all(k in props for k in c['arguments']):
            valid += 1
        if any(c['name'] == e.get('name') and all(str(c['arguments'].get(k, '')) in [str(x) for x in v]
                                                  for k, v in (e.get('arguments') or {}).items() if v and v != [''])
               for e in exps):
            exact += 1
    m = max(1, len(call_rs))
    return {'items': n, 'expect_call': len(call_rs), 'decision_accuracy': round(decision, 3),
            'right_tool': round(name_ok / m, 3), 'valid_call': round(valid / m, 3), 'right_tool_and_args': round(exact / m, 3)}


summary = {s: score([r for r in res if r['set'] == s]) for s in ('heldout', 'eval_tools')}
summary['all'] = score(res)
json.dump({'served': served, 'summary': summary, 'results': res}, open(out, 'w'), indent=1)
print(json.dumps({'served': served, **summary}))
