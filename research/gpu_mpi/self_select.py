"""Self-selection test (GPU/MPI push, Oct 25): the student itself picks, per question, between its answer WITH retrieved
excerpts and its closed-book answer. It sees only the question and the two answers (no reference answer, which would
not exist at inference), in both orders; it keeps the with-excerpts answer unless it prefers the closed-book one in
both orders. The chosen answers' judged scores (3-judge majority, already computed) give the system's score per set.

usage: self_select.py ENDPOINT SERVED OUT.json
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

ENDPOINT, SERVED, OUT = sys.argv[1:4]
C = '/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/code'


def answers(path):
    return {json.loads(l)['id']: json.loads(l)['answer'] for l in open(path)}


PROMPT = ("Two answers to the same HPC question follow. Pick the one that is more correct and more useful for the user: "
          "accurate commands and flags, no invented details, and actually answering what was asked. Reply with exactly "
          "one letter, A or B.\n\nQuestion:\n{q}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\nBetter answer (A or B):")


def ask(q, a, b):
    body = {'model': SERVED, 'temperature': 0, 'max_tokens': 4,
            'messages': [{'role': 'user', 'content': PROMPT.format(q=q, a=a[:6000], b=b[:6000])}]}
    req = Request(ENDPOINT.rstrip('/') + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urlopen(req, timeout=300) as r:
        t = json.loads(r.read())['choices'][0]['message']['content'].strip().upper()
    return 'A' if t.startswith('A') else 'B' if t.startswith('B') else '?'


jobs = []
for label, path in [('v2', 'data/eval/eval_all_v2.jsonl'), ('v3', 'data/eval/eval_v3.jsonl'),
                    ('d1', 'data/eval/dataset_v1/development_normal.jsonl')]:
    qs = {json.loads(l)['id']: json.loads(l)['question'] for l in open(os.path.join(C, path))}
    rag = answers(f'{C}/runs/r27e/answers/{label}__base_rag.jsonl')
    closed = answers(f'{C}/runs/gate27/raw/{label}__base_closed.jsonl')
    jobs += [(label, i, qs[i], rag[i], closed[i]) for i in qs if i in rag and i in closed]


def one(j):
    label, i, q, rag, closed = j
    first = ask(q, rag, closed)    # A = with excerpts
    second = ask(q, closed, rag)   # A = closed-book
    pick = 'closed' if (first == 'B' and second == 'A') else 'rag'
    return {'set': label, 'id': i, 'order1': first, 'order2': second, 'pick': pick}


with ThreadPoolExecutor(32) as ex:
    res = list(ex.map(one, jobs))
json.dump(res, open(OUT, 'w'), indent=1)
print('picked closed-book for', sum(r['pick'] == 'closed' for r in res), 'of', len(res))
