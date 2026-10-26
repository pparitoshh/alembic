"""Answer the execution-eval tasks with one served model: the evaluation system prompt, no documentation excerpts
(as in the Docker sandbox eval), greedy, 1,024 tokens, thinking off.

usage: answer_tasks.py TASKS.jsonl ENDPOINT SERVED OUT.jsonl
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

tasks, endpoint, served, out = sys.argv[1:5]
SYSTEM = ("You are an HPC assistant. Use supplied source excerpts for site-specific facts and tool results for live state. "
          "Name the applicable cluster and ask for missing required details. Give a complete, clear answer without "
          "inventing commands, flags or cluster defaults. Put scripts in fenced code blocks.")


def ask(t):
    body = {'model': served, 'temperature': 0, 'max_tokens': 1024, 'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': t['prompt']}]}
    req = Request(endpoint.rstrip('/') + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urlopen(req, timeout=600) as r:
        msg = json.loads(r.read())['choices'][0]['message']
    return {'id': t['id'], 'model': served, 'answer': msg.get('content') or ''}


rows = [json.loads(l) for l in open(tasks)]
with ThreadPoolExecutor(16) as ex, open(out, 'w') as f:
    for r in ex.map(ask, rows):
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('answered', len(rows), served)
