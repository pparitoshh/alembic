"""gpt-oss-20b paired judging (both orders) of every arm against closed-book base answers.

usage: judge.py WORKER WORKERS ENDPOINT ANSWERS_DIR OUT_DIR
Work = (slice, arm, chunk of 34 questions), dealt round-robin to workers. Resumable per unit.
"""
import json
import sys
from pathlib import Path

from distillkit.config import load_config
from distillkit.evaluate import win_rate
from distillkit.teacher import Teacher

worker, workers, endpoint = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
answers, out = Path(sys.argv[4]), Path(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
cfg = load_config('configs/rag65_seed42.json')
cfg.judge.model, cfg.judge.base_url = 'openai/gpt-oss-20b', endpoint
cfg.judge.max_tokens, cfg.judge.extra_body, cfg.judge.concurrency = 8192, {'reasoning_effort': 'medium'}, 16
judge = Teacher(cfg.judge)
units = []
for label, path in [('v2', 'data/eval/eval_all_v2.jsonl'), ('v3', 'data/eval/eval_v3.jsonl'), ('d1', 'data/eval/dataset_v1/development_normal.jsonl')]:
    refs = [json.loads(x) for x in Path(path).read_text().splitlines()]
    arms = sorted(p.name[len(label) + 2:-6] for p in answers.glob(label + '__*.jsonl'))
    for arm in arms:
        if arm == 'base_closed':
            continue
        for start in range(0, len(refs), 34):
            units.append((label, arm, start, refs[start:start + 34]))
for n, (label, arm, start, refs) in enumerate(units):
    if n % workers != worker:
        continue
    dest = out / f'{label}__{arm}__{start:03d}.jsonl'
    if dest.exists():
        continue

    def load(a):
        return {json.loads(x)['id']: json.loads(x) for x in (answers / f'{label}__{a}.jsonl').read_text().splitlines()}
    cand, base = load(arm), load('base_closed')
    rows = [{'id': r['id'], 'question': r['question'], 'reference': r['reference'],
             'answer_arm': cand[r['id']]['answer'], 'answer_base': base[r['id']]['answer']} for r in refs]
    win_rate(judge, rows, 'arm', 'base')
    dest.with_suffix('.tmp').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    dest.with_suffix('.tmp').replace(dest)
    print('judged', dest.name, flush=True)
