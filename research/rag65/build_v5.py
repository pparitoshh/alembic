"""Round-5 training set: the round-4 set plus the later factory releases (top-up v2, coverage v3).

usage: build_v5.py ROUND4.jsonl OUT_DIR FACTORY_RELEASE.jsonl:ORIGIN [FACTORY_RELEASE.jsonl:ORIGIN ...]
Each later merge already dropped questions repeating B7, the CC-BY banks or the earlier batches. Answers are
never edited. For a fixed CLOSED_SHARE of factory prose rows the quoted source excerpt is removed from the user
turn, so the assistant also learns the facts without retrieval. Slurm setups differ between sites (partitions,
accounts, flags), so only rows whose question names a specific site or system go closed-book; a site-specific
fact without its site would teach the model to mix sites up. Rows whose question points at the excerpt keep it.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

CLOSED_SHARE = 35  # percent of factory rows trained closed-book
# Systems and centres named in the archive's questions (case-sensitive proper names).
NAMED = re.compile(r'\b(LUMI|Leonardo|CINECA|MareNostrum|Deucalion|JUWELS|JUPITER|JURECA|Jean Zay|IDRIS|Adastra|TGCC|'
                   r'Irene|Joliot-Curie|CINES|Vega|MeluXina|Karolina|Barbora|IT4I|Discoverer|Snellius|SURF|Alvis|Mimer|'
                   r'Berzelius|Dardel|Tetralith|Mahti|Puhti|Roihu|CSC|Daedalus|ARIS|GRNET|LRZ|SuperMUC(?:-NG)?|GWDG|'
                   r'HammerHAI|HLRS|Hawk|Hunter|BSC|JSC|EuroHPC|MareNostrum ?5|Lumi|Levante|Raven|Emmy|Grete|PDC|'
                   r'NSC|Sigma2|Betzy|Saga|Fram|Olivia|ICHEC|Kay|Meggie|Fritz|Alex|Noctua|ULHPC|Aion|Iris|'
                   r'IZUM|Arnes|Komondor|PLGrid|Athena|Helios|Ares|Prometheus|Cyfronet|Polaris|Perlmutter|Frontier)\b')
REFERS = re.compile(r'provided input|excerpt|above|attached|quoted|this (page|document|section)|'
                    r'the (input|snippet|text) (below|provided)', re.I)


def close_book(row):
    """Drop the quoted excerpt from a factory row's single user turn; False when that is not clean."""
    users = [m for m in row['messages'] if m['role'] == 'user']
    if len(users) != 1 or row.get('mode', 'prose') != 'prose' or not row.get('raft_origin', '').startswith('factory'):
        return False
    text = users[0]['content']
    cut = text.find('\n\nProvided input 1 (quoted data')
    if cut < 0 or REFERS.search(text[:cut]) or not NAMED.search(text[:cut]):
        return False
    if int(hashlib.sha256(('closed-v1/' + row['id']).encode()).hexdigest()[:8], 16) % 100 >= CLOSED_SHARE:
        return False
    users[0]['content'] = text[:cut].rstrip()
    row['raft_context_removed'] = True
    return True

base, out, releases = sys.argv[1], Path(sys.argv[2]), [a.rsplit(':', 1) for a in sys.argv[3:]]
out.mkdir(parents=True, exist_ok=True)
if (out / 'verified.jsonl').exists():
    raise SystemExit('training set exists; use a new directory')
rows = [json.loads(x) for x in open(base)]
for path, origin in releases:
    for line in open(path):
        row = json.loads(line)
        row['raft_origin'] = origin
        rows.append(row)
closed = Counter(r['raft_origin'] for r in rows if close_book(r))
ids = Counter(r['id'] for r in rows)
if max(ids.values()) > 1:
    raise SystemExit('duplicate row ids: ' + str([i for i, n in ids.items() if n > 1][:5]))
with open(out / 'verified.jsonl', 'w') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
summary = {'rows': len(rows), 'base_set': base, 'by_origin': dict(Counter(r.get('raft_origin', 'b7') for r in rows)),
           'by_licence': dict(Counter(r.get('source_licence', 'b7/ccby-existing') for r in rows)),
           'closed_book_factory_rows': dict(closed), 'closed_share_percent': CLOSED_SHARE,
           'tool_fraction': round(sum(r.get('mode', 'prose') != 'prose' for r in rows) / len(rows), 3),
           'sha256': hashlib.sha256((out / 'verified.jsonl').read_bytes()).hexdigest()}
(out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
