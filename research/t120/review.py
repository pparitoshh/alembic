"""Independent source-support review (source-support-v3, Gemma 4 26B) of the gpt-oss-120b training answers.

usage: review.py WORKER N_WORKERS ENDPOINT SERVED DATA.jsonl OUT_DIR
The evidence is the documentation excerpts quoted in each row's user turn, which is what the answer was asked to be
grounded in (the original rows were reviewed against their seed chunk; these answers were written from the prompt).
Output: OUT_DIR/reviews_<worker>.jsonl with the raw review report plus `status` (supported / unsupported / uncertain)
from the same check_support used by `distillkit verify`. Resumable across a changed node count.
"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from distillkit.config import load_config
from distillkit.support_v3 import check_support, request_review
from distillkit.teacher import Teacher

worker, n_workers, endpoint, served, data, out = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4],
                                                   sys.argv[5], Path(sys.argv[6]))
MARK = 'Quoted retrieval results'
cfg = load_config('configs/rag65_seed42.json')
cfg.judge.model, cfg.judge.base_url = served, endpoint
cfg.judge.temperature, cfg.judge.concurrency, cfg.judge.max_tokens = 0, 32, 20000  # long answers have many units; 8192 truncated ~89% of reviews (Oct 23). 32, not 128: with 128 long reviews in flight almost none finished before the 30 min debug limit killed the job (~25 saved per 8-node job)
reviewer = Teacher(cfg.judge, name='reviewer')


def evidence(row):
    user = next(m['content'] for m in row['messages'] if m['role'] == 'user')
    source = user[user.index(MARK):] if MARK in user else user
    return {**row, 'source_sha256': hashlib.sha256(source.encode()).hexdigest()}, source


out.mkdir(parents=True, exist_ok=True)
dest = out / f'reviews_{worker:03d}.jsonl'
done = set()
for f in out.glob('reviews_*.jsonl'):
    for x in f.read_text().splitlines():
        try:
            r = json.loads(x)
        except ValueError:
            continue
        if r.get('review') is not None:
            done.add(r['id'])
rows = [json.loads(l) for i, l in enumerate(open(data)) if i % n_workers == worker]
todo = [evidence(r) for r in rows if r['id'] not in done]
print(f'[review] worker {worker}: {len(rows)} rows, {len(todo)} to review', flush=True)


def one(item):
    row, source = item
    if MARK not in source:  # closed-book row: no quoted evidence to review against; recorded, not judged
        return {'id': row['id'], 'review': {}, 'status': 'no_excerpts'}
    report = request_review(reviewer, row, source)
    report['status'] = check_support(row, source, report)['status']
    return report


failed = 0
with ThreadPoolExecutor(cfg.judge.concurrency) as pool, dest.open('a') as f:
    for fut in as_completed([pool.submit(one, t) for t in todo]):
        try:
            f.write(json.dumps(fut.result(), ensure_ascii=False) + '\n'); f.flush()
        except Exception as exc:  # a failed request is retried by the next job
            failed += 1
            print(f'[review] request failed: {exc!r}', flush=True)
print(f'[review] worker {worker}: done ({failed} failed)', flush=True)
