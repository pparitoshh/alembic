"""Majority vote of three judges (gpt-oss-20b, Gemma 4 26B, Qwen3-235B) on the same answers.

usage: majority.py RUN_DIR [OUT_SUBDIR]    reads RUN_DIR/judged, judged_gemma, judged_qwen; writes RUN_DIR/judged_majority
For every question and each answer order, the verdict at least two judges give (A, B or tie) is kept; when all three
disagree the pair counts as a tie. Unit files are combined only when all three judges finished them; the output has
the judges' format, so research/rag65/aggregate.py and paired.py run on it unchanged. A judge's unparsed verdict
leaves the other two, and a 1-1 split is a tie.
"""
import json
import sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
out = run / (sys.argv[2] if len(sys.argv) > 2 else 'judged_majority')
out.mkdir(parents=True, exist_ok=True)
judges = [run / 'judged', run / 'judged_gemma', run / 'judged_qwen']


def vote(vs):
    vs = [v for v in vs if v is not None]
    if not vs:
        return None
    top, n = Counter(vs).most_common(1)[0]
    return top if n >= 2 or len(vs) == 1 else 'T'


done = skipped = 0
for f in sorted(judges[0].glob('*__*__*.jsonl')):
    others = [j / f.name for j in judges[1:]]
    if not all(o.exists() for o in others):
        skipped += 1
        continue
    rows = [[json.loads(x) for x in p.read_text().splitlines()] for p in [f, *others]]
    by_id = [{r['id']: r for r in rs} for rs in rows]
    merged = []
    for r in rows[0]:
        key = next(iter(r.get('judge', {})), None)
        if key is None:
            continue
        verdicts = [(d.get(r['id'], {}).get('judge') or {}).get(key) or [None, None] for d in by_id]
        m = dict(r)
        m['judge'] = {key: [vote([v[0] for v in verdicts]), vote([v[1] for v in verdicts])]}
        m['judge_votes'] = {key: verdicts}
        merged.append(m)
    (out / f.name).write_text(''.join(json.dumps(x) + '\n' for x in merged))
    done += 1
print(f'majority: {done} units combined, {skipped} waiting for a judge')
