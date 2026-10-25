"""Best-of-n, step 1: K new teacher answers per grounded prose training question.

usage: candidates.py WORKER N_WORKERS ENDPOINT SERVED_MODEL DATA.jsonl OUT_DIR [K]
Only prose rows with one user turn that quotes its sources are used (RAFT context or a factory excerpt):
the selector can then check every candidate against the same sources. The teacher sees the student's
system prompt and user turn plus round 3's length note (the judge prefers complete answers that fit the
768-token evaluation budget). Each candidate uses its own seed, at temperature 0.7. Resumable.
"""
import json
import os
import sys
from pathlib import Path

from distillkit.config import load_config
from distillkit.teacher import Teacher

worker, n_workers, endpoint, served, data, out = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4],
                                                   sys.argv[5], Path(sys.argv[6]))
K = int(sys.argv[7]) if len(sys.argv) > 7 else 3
LENGTH_NOTE = ('\n\nGive a thorough, complete answer of about 300 to 550 words, including any script, with the '
               'reasoning, the commands and how to check the result. Use the quoted sources only where they are '
               'relevant to the question.')
QUOTED = ('Quoted retrieval results (data, not instructions):', 'Provided input 1 (quoted data')
# Student best-of-n at answer time: the student sees exactly its evaluation prompt (no teacher-only length note).
if os.environ.get('NO_LENGTH_NOTE') == '1':
    LENGTH_NOTE = ''


ONLY_ORIGIN = os.environ.get('ONLY_ORIGIN')  # e.g. factory_v3: rows already best-of-n'd are left alone


def eligible(row):
    users = [m for m in row['messages'] if m['role'] == 'user']
    return ((not ONLY_ORIGIN or row.get('raft_origin') == ONLY_ORIGIN)
            and row.get('mode', 'prose') == 'prose' and len(users) == 1 and len(row['messages']) == 2
            and [m['role'] for m in row['messages']] == ['user', 'assistant']
            and any(q in users[0]['content'] for q in QUOTED))


cfg = load_config('configs/rag65_seed42.json')
cfg.teacher.model, cfg.teacher.base_url = served, endpoint
# 192 requests in flight: the sampled candidates of a row run in parallel, not one after another. Loading the
# model takes about half of each 30-minute job, so the remaining minutes must be fully used.
# CAND_MAX_TOKENS (opt-in, Oct 25): answers that do not finish within the limit are dropped; 760 dropped 46% of round 27's
# answers to round 47's questions, so round 47 uses 1536 (the evaluation answer limit since Oct 24). Unset = 760.
cfg.teacher.temperature, cfg.teacher.top_logprobs, cfg.teacher.concurrency, cfg.teacher.max_tokens = 0.7, 0, 192, int(os.environ.get('CAND_MAX_TOKENS') or 760)
cfg.teacher.extra_body = {'chat_template_kwargs': {'enable_thinking': False}}
client = Teacher(cfg.teacher)
rows = []
for i, line in enumerate(open(data)):
    if i % n_workers == worker:
        row = json.loads(line)
        if eligible(row):
            rows.append({'id': row['id'], 'user': next(m['content'] for m in row['messages'] if m['role'] == 'user')})
out.mkdir(parents=True, exist_ok=True)
dest = out / f'cands_{worker:03d}.jsonl'
done = set()
if dest.exists():  # a job killed at its time limit can leave a partial last line
    for x in dest.read_text().splitlines():
        try:
            done.add(json.loads(x)['id'])
        except (ValueError, KeyError):
            pass
todo = [r for r in rows if r['id'] not in done]
print(f'worker {worker}: {len(rows)} eligible rows, {len(todo)} to do', flush=True)


def sample(item):
    r, k = item
    try:
        c = client.complete([{'role': 'system', 'content': cfg.student.system_prompt + LENGTH_NOTE},
                             {'role': 'user', 'content': r['user']}], extra_body={'seed': 1000 + k})
        return r['id'], k, (c.content if c.finish_reason == 'stop' and (c.content or '').strip() else None), True
    except Exception as e:  # the row is retried on the next job
        print('failed', r['id'], type(e).__name__, flush=True)
        return r['id'], k, None, False


for start in range(0, len(todo), 128):
    chunk = todo[start:start + 128]
    got, failed = {}, set()
    for rid, k, text, ok in client.map(sample, [(r, k) for r in chunk for k in range(K)]):
        if not ok:
            failed.add(rid)
        elif text is not None:
            got.setdefault(rid, {})[k] = text
    with dest.open('a') as f:
        for r in chunk:
            if r['id'] in failed:
                continue
            cands = [got.get(r['id'], {})[k] for k in sorted(got.get(r['id'], {}))]
            f.write(json.dumps({'id': r['id'], 'teacher': served, 'candidates': cands}, ensure_ascii=False) + '\n')
            f.flush()
    print(f'worker {worker}: {len(done) + start + len(chunk)}/{len(rows)}', flush=True)
