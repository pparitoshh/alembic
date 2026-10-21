"""Rejection sampling against the teacher, step 3: keep the student answers that beat the teacher.

usage: build.py DATA.jsonl CANDS_DIR JUDGED_DIR OUT.jsonl [THRESHOLD]
For each question, the candidate with the highest score against the teacher is kept when its score reaches
THRESHOLD (default 0.75: better in one answer order and at least a tie in the other). The row keeps its question and
sources; the answer becomes the student's own winning answer. Prints counts at 0.5, 0.75 and 1.0.
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

data, cands_dir, judged, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.75
rows = {json.loads(x)['id']: json.loads(x) for x in data.read_text().splitlines()}
cands = {}
for f in glob.glob(str(cands_dir / 'cands_*.jsonl')):
    for line in open(f):
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if c.get('candidates'):
            cands[c['id']] = c['candidates']
scores = defaultdict(dict)
for f in glob.glob(str(judged / 'unit_*.jsonl')):
    for line in open(f):
        r = json.loads(line)
        if r['score'] is not None:
            scores[r['id']][r['k']] = r['score']
best = {rid: max(ks.items(), key=lambda kv: kv[1]) for rid, ks in scores.items() if ks}
for t in (0.5, 0.75, 1.0):
    print(f'questions whose best candidate scores >= {t}: {sum(s >= t for _, s in best.values())} of {len(best)}')
mean = sum(s for ks in scores.values() for s in ks.values()) / max(1, sum(len(ks) for ks in scores.values()))
print(f'mean candidate score against the teacher: {mean:.3f}')
n = 0
with out.open('w') as f:
    for rid, (k, s) in sorted(best.items()):
        if s >= threshold and rid in rows:
            r = dict(rows[rid])
            r['messages'] = [*r['messages'][:-1], {'role': 'assistant', 'content': cands[rid][k]}]
            r['rft'] = {'score_vs_teacher': s, 'candidate': k}
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            n += 1
print(n)
