"""Answer the three test sets (v2, v3, d1) with one served model, the app's prompt (RAG_VARIANT, default e) and
ANSWER_MAX_TOKENS (default 1,536), greedy, thinking off: research/rag65/answer.py's rag arm, for any endpoint.

usage: answer.py ENDPOINT SERVED OUT_DIR ARM    writes OUT_DIR/{v2,v3,d1}__ARM.jsonl (resumable)
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

os.environ.setdefault('RAG_VARIANT', 'e')
from distillkit.config import load_config  # noqa: E402
from retrieve import retrieve, with_context  # noqa: E402

endpoint, served, out, arm = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
SYSTEM = load_config('configs/rag65_seed42.json').student.system_prompt
MAXTOK = int(os.environ.get('ANSWER_MAX_TOKENS') or 1536)
P, F = 'runs/rag65/indexes/documents.sqlite', 'runs/rag65/indexes/factory_docs.sqlite'


def ask(q):
    body = {'model': served, 'temperature': 0, 'max_tokens': MAXTOK, 'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}]}
    for _ in range(3):
        try:
            req = Request(endpoint.rstrip('/') + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with urlopen(req, timeout=900) as r:
                c = json.loads(r.read())['choices'][0]
                return c['message']['content'], c.get('finish_reason')
        except Exception:
            pass
    return None, None


for label, path in [('v2', 'data/eval/eval_all_v2.jsonl'), ('v3', 'data/eval/eval_v3.jsonl'), ('d1', 'data/eval/dataset_v1/development_normal.jsonl')]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
    dest = out / f'{label}__{arm}.jsonl'
    done = {json.loads(x)['id'] for x in dest.read_text().splitlines()} if dest.exists() else set()
    todo = [r for r in rows if r['id'] not in done]

    def one(r):
        docs = retrieve(r['question'], P, F)
        a, fin = ask(with_context(r['question'], docs))
        return {'id': r['id'], 'answer': a, 'finish_reason': fin, 'retrieval': [d['url'] for d in docs]} if a else None
    with ThreadPoolExecutor(64) as ex, dest.open('a') as f:
        for res in ex.map(one, todo):
            if res:
                f.write(json.dumps(res) + '\n')
    print(label, arm, sum(1 for _ in dest.open()), 'of', len(rows), flush=True)
