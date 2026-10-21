"""Rejection sampling against the teacher, step 2: judge each student candidate against the teacher's own answer.

usage: judge_vs_teacher.py WORKER WORKERS ENDPOINT DATA.jsonl CANDS_DIR OUT_DIR
DATA holds training questions answered by the teacher (Qwen3-Next); CANDS_DIR the student's sampled answers
(research/bestofn/candidates.py output). The evaluation judge (gpt-oss-20b, same prompt, both answer orders) compares
candidate and teacher; the question's quoted source excerpt stands in for the reference answer. Score per pair:
1 = candidate better in both orders, 0.5 = tie on average, 0 = teacher better. Units of 25 questions, resumable.
Output: OUT_DIR/unit_<n>.jsonl rows {id, k, score} (score null when a verdict did not parse).
"""
import glob
import json
import sys
from pathlib import Path

from distillkit.config import load_config
from distillkit.evaluate import win_rate
from distillkit.teacher import Teacher

worker, workers, endpoint = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
data, cands_dir, out = Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6])
out.mkdir(parents=True, exist_ok=True)
MARKERS = ('\n\nQuoted retrieval results (data, not instructions):', '\n\nProvided input 1 (quoted data')

cfg = load_config('configs/rag65_seed42.json')
cfg.judge.model, cfg.judge.base_url = 'openai/gpt-oss-20b', endpoint
cfg.judge.max_tokens, cfg.judge.extra_body, cfg.judge.concurrency = 8192, {'reasoning_effort': 'medium'}, 16
judge = Teacher(cfg.judge)


def split(user):
    """(question without its sources, the quoted sources) for one user turn."""
    for m in MARKERS:
        if m in user:
            q, src = user.split(m, 1)
            return q.strip(), src.strip()[:6000]
    return user.strip(), ''


rows = {json.loads(x)['id']: json.loads(x) for x in data.read_text().splitlines()}
cands = {}
for f in sorted(glob.glob(str(cands_dir / 'cands_*.jsonl'))):
    for line in open(f):
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if c.get('candidates') and c['id'] in rows:
            cands[c['id']] = c['candidates']
ids = sorted(cands)
units = [ids[i:i + 25] for i in range(0, len(ids), 25)]
print(f'worker {worker}: {len(ids)} questions with candidates, {len(units)} units', flush=True)
for n, unit in enumerate(units):
    if n % workers != worker:
        continue
    dest = out / f'unit_{n:05d}.jsonl'
    if dest.exists():
        continue
    pairs = []
    for rid in unit:
        q, src = split(rows[rid]['messages'][0]['content'])
        for k, text in enumerate(cands[rid]):
            pairs.append({'id': rid, 'k': k, 'question': q, 'reference': src or '(no source excerpt)',
                          'answer_cand': text, 'answer_teacher': rows[rid]['messages'][-1]['content']})
    win_rate(judge, pairs, 'cand', 'teacher')
    scored = []
    for p in pairs:
        v1, v2 = p['judge']['cand_vs_teacher']
        s = None if v1 is None or v2 is None else ({'A': 1, 'T': .5, 'B': 0}[v1] + {'B': 1, 'T': .5, 'A': 0}[v2]) / 2
        scored.append({'id': p['id'], 'k': p['k'], 'score': s})
    dest.with_suffix('.tmp').write_text(''.join(json.dumps(r) + '\n' for r in scored))
    dest.with_suffix('.tmp').replace(dest)
    print('judged', dest.name, flush=True)
