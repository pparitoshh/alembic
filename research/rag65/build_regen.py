"""Training set v2: RAFT rows with teacher-regenerated prose answers, plus CC-BY v1 and v2 rows.

usage: build_regen.py RAFT.jsonl REGEN_DIR CCBY_V2.jsonl OUT_DIR
A prose row keeps its regenerated answer only if the teacher finished (no truncation), the answer passes the
flag and bash checks and is not trivially short; otherwise the row is dropped, not reverted to the short answer.
Tool rows and CC-BY rows are unchanged.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from distillkit.checks import check_answer, load_flags

raft, regen_dir, ccby2, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
if (out / 'verified.jsonl').exists():
    raise SystemExit('training set exists; use a new directory')
regen = {}
for p in sorted(regen_dir.glob('regen_*.jsonl')):
    for r in map(json.loads, p.read_text().splitlines()):
        regen[r['id']] = r
flags, stats, lengths = load_flags(None), Counter(), []
rows = []
for row in map(json.loads, open(raft)):
    users = [m for m in row['messages'] if m['role'] == 'user']
    if row['raft_origin'] == 'b7' and row.get('mode', 'prose') == 'prose' and len(users) == 1:
        r = regen.get(row['id'])
        if r is None:
            stats['prose_missing'] += 1
            continue
        if r['finish_reason'] != 'stop' or len(r['answer']) < 200:
            stats['prose_truncated_or_short'] += 1
            continue
        checks = check_answer(r['answer'], flags)
        if not checks['passed']:
            stats['prose_failed_checks'] += 1
            continue
        assistant = next(m for m in row['messages'] if m['role'] == 'assistant')
        stats['old_chars'] += len(assistant['content'])
        assistant['content'] = r['answer']
        row['checks'], row['regenerated'] = checks, 'Qwen/Qwen3-32B-AWQ/regen-v1'
        lengths.append(len(r['answer']))
        stats['prose_regenerated'] += 1
    else:
        stats[f"kept_{row['raft_origin']}_{row.get('mode', 'prose')}"] += 1
    rows.append(row)
for row in map(json.loads, open(ccby2)):
    row['raft_origin'] = 'ccby_v2'
    rows.append(row)
    stats['ccby_v2_rows'] += 1
with open(out / 'verified.jsonl', 'w') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
lengths.sort()
old_mean = stats.pop('old_chars') / max(stats['prose_regenerated'], 1)
summary = {'rows': len(rows), **stats, 'regenerated_median_chars': lengths[len(lengths) // 2] if lengths else 0,
           'replaced_answers_mean_chars_before': round(old_mean),
           'sha256': hashlib.sha256((out / 'verified.jsonl').read_bytes()).hexdigest()}
(out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
