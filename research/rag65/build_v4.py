"""Round-4 training set: the better of the round-2/3 sets plus the all-factory release.

usage: build_v4.py RESULT_A.json DATA_A.jsonl RESULT_B.json DATA_B.jsonl FACTORY.jsonl OUT_DIR
Picks the base set whose full-strength (x1.0) student with retrieval has the higher mean win rate over
v2, v3 and d1, then appends every factory row (licence-tagged). Answers are never edited.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

res_a, data_a, res_b, data_b, factory, out = sys.argv[1:7]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
if (out / 'verified.jsonl').exists():
    raise SystemExit('training set exists; use a new directory')


def score(path):
    means = json.loads(Path(path).read_text())['seed_means']
    return sum(means[f'{s}/student_rag']['mean'] for s in ('v2', 'v3', 'd1')) / 3


scores = {data_a: score(res_a), data_b: score(res_b)}
base = max(scores, key=scores.get)
rows = [json.loads(x) for x in open(base)]
stats = Counter(r.get('raft_origin', 'b7') for r in rows)
for line in open(factory):
    row = json.loads(line)
    row['raft_origin'] = 'factory_v1'
    rows.append(row)
    stats['factory_v1'] += 1
with open(out / 'verified.jsonl', 'w') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
summary = {'rows': len(rows), 'base_set': base, 'base_scores_x1_rag_mean': scores, 'by_origin': dict(stats),
           'by_licence': dict(Counter(r.get('source_licence', 'b7/ccby-existing') for r in rows)),
           'tool_fraction': round(sum(r.get('mode', 'prose') != 'prose' for r in rows) / len(rows), 3),
           'sha256': hashlib.sha256((out / 'verified.jsonl').read_bytes()).hexdigest()}
(out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
