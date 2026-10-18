"""RAFT training set: B7 prose questions with the evaluation retriever's context attached, plus CC-BY rows.

usage: build_raft.py B7.jsonl CCBY.jsonl PILOT_DB FACTORY_DB OUT_DIR
~80% of B7 prose rows get retrieved context (RAFT, Zhang et al. 2024); the rest stay closed-book so the
student still answers without documents. Tool traces stay unchanged: live state comes from tools.
CC-BY rows already carry their source excerpt. Answers are never edited.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import retrieve, with_context

b7, ccby, pilot_db, factory_db, out = sys.argv[1:6]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
if (out / 'verified.jsonl').exists():
    raise SystemExit('RAFT data exists; use a new directory')


def bucket(row_id):
    return int(hashlib.sha256(('raft-v1/' + row_id).encode()).hexdigest()[:8], 16) % 100


rows, stats = [], Counter()
for line in open(b7):
    row = json.loads(line)
    users = [m for m in row['messages'] if m['role'] == 'user']
    if row.get('mode', 'prose') == 'prose' and len(users) == 1 and bucket(row['id']) < 80:
        docs = retrieve(users[0]['content'], pilot_db, factory_db)
        if docs:
            row = copy.deepcopy(row)
            next(m for m in row['messages'] if m['role'] == 'user')['content'] = with_context(users[0]['content'], docs)
            row['raft_context'] = [{'url': d['url'], 'site': d['site']} for d in docs]
            stats['b7_with_context'] += 1
        else:
            stats['b7_no_retrieval_hit'] += 1
    else:
        stats['b7_closed_book_or_tool'] += 1
    row['raft_origin'] = 'b7'
    rows.append(row)
for line in open(ccby):
    row = json.loads(line)
    row['raft_origin'] = 'ccby_v1'
    rows.append(row)
    stats['ccby_rows'] += 1
with open(out / 'verified.jsonl', 'w') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
summary = {'rows': len(rows), **stats,
           'sha256': hashlib.sha256((out / 'verified.jsonl').read_bytes()).hexdigest(),
           'inputs': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [b7, ccby, pilot_db, factory_db]}}
(out / 'RAFT_SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
