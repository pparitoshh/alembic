"""Round 51b step 2: Qwen3-235B answers every round 51 question with the same prompt as round 27 (the app's excerpts,
variant e; same system prompt, greedy, 1,536 tokens). Resumable; node SHARD of NSHARDS answers questions SHARD,
SHARD+NSHARDS, ... into ANSWERS/part{SHARD}_teacher.jsonl; the sbatch appends the parts to ANSWERS/d1__base_closed.jsonl
(the comparison answer for the judge: round 27 = arm, Qwen3-235B = base, both judged with the verified reference).

usage: r51b_teacher.py RD_DIR ENDPOINT SERVED
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

rd, ENDPOINT, SERVED = sys.argv[1:4]
SHARD, NSHARDS = int(os.environ.get('SHARD', 0)), int(os.environ.get('NSHARDS', 1))
SYSTEM = ("You are an HPC assistant. Use supplied source excerpts for site-specific facts and tool results for live state. "
          "Name the applicable cluster and ask for missing required details. Give a complete, clear answer without "
          "inventing commands, flags or cluster defaults. Put scripts in fenced code blocks.")
rows = [json.loads(l) for l in open(f'{rd}/questions.jsonl')]
done = set()
for f in [f'{rd}/answers/d1__base_closed.jsonl'] + glob.glob(f'{rd}/answers/part*_teacher.jsonl'):
    if os.path.exists(f):
        done |= {json.loads(l)['id'] for l in open(f) if l.strip()}


def ask(prompt):
    body = {'model': SERVED, 'temperature': 0, 'max_tokens': 1536,
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]}
    req = Request(ENDPOINT.rstrip('/') + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    for _ in range(3):
        try:
            with urlopen(req, timeout=900) as r:
                return json.loads(r.read())['choices'][0]['message']['content']
        except Exception:
            pass
    return None


todo = [r for i, r in enumerate(rows) if r['id'] not in done and i % NSHARDS == SHARD]
n = 0
with ThreadPoolExecutor(96) as ex, open(f'{rd}/answers/part{SHARD}_teacher.jsonl', 'a') as fo:
    futs = {ex.submit(ask, r['rag_prompt']): r for r in todo}
    for f in as_completed(futs):
        a = f.result()
        if a:
            fo.write(json.dumps({'id': futs[f]['id'], 'answer': a}, ensure_ascii=False) + '\n')
            fo.flush()
            n += 1
print(f'TEACHER READY (shard {SHARD}/{NSHARDS}): {n} new answers, {len(done)} earlier, {len(rows)} questions')
