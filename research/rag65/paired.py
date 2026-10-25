"""Paired bootstrap over questions between arms judged in the same judge run (mentor issue #37, point 5).

usage: paired.py JUDGED_DIR OUT.json ARM_A:ARM_B [ARM_A:ARM_B ...]
Each arm's per-question score is its both-orders result against closed-book base. The difference is
resampled by question, so judge drift between runs cannot enter the comparison.
"""
from collections import defaultdict
import json
import random
import sys
from pathlib import Path

judged, out, pairs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
S = {'A': 1., 'T': .5, 'B': 0.}
R = {'B': 1., 'T': .5, 'A': 0.}
scores = defaultdict(dict)
for p in judged.glob('*.jsonl'):
    label, arm, _ = p.stem.split('__')
    for r in map(json.loads, p.read_text().splitlines()):
        v = r['judge']['arm_vs_base']
        if None not in v:
            scores[(label, arm)][r['id']] = (S[v[0]] + R[v[1]]) / 2
report = {}
for pair in pairs:
    a, b = pair.split(':')
    for label in ('v2', 'v3', 'd1'):
        sa, sb = scores.get((label, a), {}), scores.get((label, b), {})
        ids = sorted(set(sa) & set(sb))
        if not ids:
            continue
        diff = [sa[i] - sb[i] for i in ids]
        rng = random.Random(42)
        boot = sorted(sum(rng.choices(diff, k=len(diff))) / len(diff) for _ in range(10000))
        report[f'{label}/{a}-vs-{b}'] = {
            'questions': len(ids), 'a': round(sum(sa[i] for i in ids) / len(ids), 4),
            'b': round(sum(sb[i] for i in ids) / len(ids), 4), 'diff': round(sum(diff) / len(diff), 4),
            'paired_95': [round(boot[250], 4), round(boot[9749], 4)],
            'p_a_not_better': round(sum(d <= 0 for d in boot) / len(boot), 4)}
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
