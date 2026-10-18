"""Round-5 training set: the round-4 set plus the top-up factory release (ai_factory_all_v2).

usage: build_v5.py ROUND4.jsonl FACTORY_V2.jsonl OUT_DIR
The top-up merge already dropped questions repeating B7, the CC-BY banks or batch 1. Answers are never edited.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

base, factory, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
if (out / 'verified.jsonl').exists():
    raise SystemExit('training set exists; use a new directory')
rows = [json.loads(x) for x in open(base)]
for line in open(factory):
    row = json.loads(line)
    row['raft_origin'] = 'factory_v2'
    rows.append(row)
ids = Counter(r['id'] for r in rows)
if max(ids.values()) > 1:
    raise SystemExit('duplicate row ids: ' + str([i for i, n in ids.items() if n > 1][:5]))
with open(out / 'verified.jsonl', 'w') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
summary = {'rows': len(rows), 'base_set': base, 'by_origin': dict(Counter(r.get('raft_origin', 'b7') for r in rows)),
           'by_licence': dict(Counter(r.get('source_licence', 'b7/ccby-existing') for r in rows)),
           'tool_fraction': round(sum(r.get('mode', 'prose') != 'prose' for r in rows) / len(rows), 3),
           'sha256': hashlib.sha256((out / 'verified.jsonl').read_bytes()).hexdigest()}
(out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
