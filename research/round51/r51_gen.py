"""Round 51 step 2: round 27 answers every question with the retrieved excerpts (the app's prompt, variant e), greedy,
1,536 tokens, same system prompt as training and evaluation. Resumable (answers already written are kept). Writes the
judge layout: EVAL/d1.jsonl (questions with references), ANSWERS/d1__rag.jsonl (round 27) and ANSWERS/d1__base_closed.jsonl
(the verified reference as the comparison answer, so the judge says whether round 27 is as good as the reference).

usage: r51_gen.py RD_DIR ENDPOINT[,ENDPOINT...] SERVED   (requests are spread over the endpoints)
Multi-node (Oct 26): SHARD=k NSHARDS=n answers questions k, k+n, ... into ANSWERS/part{k}_*.jsonl and EVAL/part{k}_d1.jsonl;
the sbatch appends the parts to the main files after all nodes finish. Answers in the main files or any part are kept.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

rd, ENDPOINTS, SERVED = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
SYSTEM = ("You are an HPC assistant. Use supplied source excerpts for site-specific facts and tool results for live state. "
          "Name the applicable cluster and ask for missing required details. Give a complete, clear answer without "
          "inventing commands, flags or cluster defaults. Put scripts in fenced code blocks.")
rows = [json.loads(l) for l in open(f'{rd}/questions.jsonl')]
os.makedirs(f'{rd}/eval', exist_ok=True)
os.makedirs(f'{rd}/answers', exist_ok=True)
import glob  # noqa: E402
SHARD, NSHARDS = int(os.environ.get('SHARD', 0)), int(os.environ.get('NSHARDS', 1))
P = f'part{SHARD}_' if NSHARDS > 1 else ''
done = set()
for f in [f'{rd}/answers/d1__rag.jsonl'] + glob.glob(f'{rd}/answers/part*_d1__rag.jsonl'):
    if os.path.exists(f):
        done |= {json.loads(l)['id'] for l in open(f) if l.strip()}


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


todo = [(i, r) for i, r in enumerate(rows) if r['id'] not in done and i % NSHARDS == SHARD]
n = 0
with ThreadPoolExecutor(128) as ex, open(f'{rd}/eval/{P}d1.jsonl', 'a') as fe, open(f'{rd}/answers/{P}d1__rag.jsonl', 'a') as fr, \
        open(f'{rd}/answers/{P}d1__base_closed.jsonl', 'a') as fc:
    futs = {ex.submit(ask, r['rag_prompt'], i): r for i, r in todo}
    for f in as_completed(futs):
        r, a = futs[f], f.result()
        if not a:
            continue
        fe.write(json.dumps({'id': r['id'], 'question': r['question'], 'reference': r['reference']}, ensure_ascii=False) + '\n')
        fr.write(json.dumps({'id': r['id'], 'answer': a}, ensure_ascii=False) + '\n')
        fc.write(json.dumps({'id': r['id'], 'answer': r['reference']}, ensure_ascii=False) + '\n')
        for fh in (fe, fr, fc):
            fh.flush()
        n += 1
for lab in ('v2', 'v3'):
    open(f'{rd}/eval/{lab}.jsonl', 'w').close()
print(f'GEN READY (shard {SHARD}/{NSHARDS}): {n} new answers, {len(done)} earlier, {len(rows)} questions')
