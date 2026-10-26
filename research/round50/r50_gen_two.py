"""Round 50 step 2: round 27 answers every question twice, with the retrieved excerpts (the app's prompt) and closed-book,
greedy, 1,536 tokens, same system prompt as training and evaluation. Writes the judge layout: EVAL/d1.jsonl (questions
with references), ANSWERS/d1__rag.jsonl and ANSWERS/d1__base_closed.jsonl.

usage: r50_gen_two.py RD_DIR ENDPOINT[,ENDPOINT...] SERVED   (requests are spread over the endpoints)
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

rd, ENDPOINTS, SERVED = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
SYSTEM = ("You are an HPC assistant. Use supplied source excerpts for site-specific facts and tool results for live state. "
          "Name the applicable cluster and ask for missing required details. Give a complete, clear answer without "
          "inventing commands, flags or cluster defaults. Put scripts in fenced code blocks.")
rows = [json.loads(l) for l in open(f'{rd}/questions.jsonl')]
os.makedirs(f'{rd}/eval', exist_ok=True)
os.makedirs(f'{rd}/answers', exist_ok=True)


def ask(prompt, k=0):
    ENDPOINT = ENDPOINTS[k % len(ENDPOINTS)]
    body = {'model': SERVED, 'temperature': 0, 'max_tokens': 1536,
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]}
    req = Request(ENDPOINT.rstrip('/') + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    for _ in range(3):
        try:
            with urlopen(req, timeout=600) as r:
                return json.loads(r.read())['choices'][0]['message']['content']
        except Exception:
            pass
    return None


def one(ir):
    i, r = ir
    return r, ask(r['rag_prompt'], i), ask(r['question'], i + 1)


with ThreadPoolExecutor(128) as ex:
    res = [x for x in ex.map(one, enumerate(rows)) if x[1] and x[2]]
with open(f'{rd}/eval/d1.jsonl', 'w') as fe, open(f'{rd}/answers/d1__rag.jsonl', 'w') as fr, \
        open(f'{rd}/answers/d1__base_closed.jsonl', 'w') as fc:
    for r, a_rag, a_closed in res:
        fe.write(json.dumps({'id': r['id'], 'question': r['question'], 'reference': r['reference']}, ensure_ascii=False) + '\n')
        fr.write(json.dumps({'id': r['id'], 'answer': a_rag}, ensure_ascii=False) + '\n')
        fc.write(json.dumps({'id': r['id'], 'answer': a_closed}, ensure_ascii=False) + '\n')
for lab in ('v2', 'v3'):
    open(f'{rd}/eval/{lab}.jsonl', 'w').close()
print(f'GEN READY: {len(res)} of {len(rows)} questions answered both ways')
