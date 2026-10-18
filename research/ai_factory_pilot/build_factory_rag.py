"""All-factory retrieval index (RAG only, never SFT training). CPU only.

usage: build_factory_rag.py ARCHIVE_DIR OUT_DIR
Indexes the archive's extracted technical text with SQLite FTS5 and writes a URL manifest,
so a laptop can rebuild the same index from the public pages instead of us redistributing text.
"""
from collections import Counter
import ast
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ARCHIVE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REPO = Path(__file__).resolve().parents[2]
OUT.mkdir(parents=True, exist_ok=True)
module = ast.parse((REPO / 'src/distillkit/seeds.py').read_text())
node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'chunk_text')
ns = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'chunk_text', 'exec'), ns)
chunk_text = ns['chunk_text']

KINDS = {'technical_documentation', 'operator_guides', 'operator_documentation', 'training_materials'}
SKIP_PATH = ('/news', '/changelog', '/legal', '/privacy', '/archive/', '/events', '/blog')
factories = {}
with open(ARCHIVE / 'FACTORY_COVERAGE.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        factories[row['factory_id']] = row['name']

tmp = OUT / 'factory_docs.sqlite.tmp'
tmp.unlink(missing_ok=True)
db = sqlite3.connect(tmp)
db.executescript('''CREATE TABLE chunks(id INTEGER PRIMARY KEY, url TEXT, source_id TEXT, factories TEXT,
 title TEXT, text TEXT, text_sha256 TEXT, retrieved_at TEXT);
 CREATE VIRTUAL TABLE search USING fts5(text, title, factories, content=chunks, content_rowid=id);''')
documents, skipped, seen = [], Counter(), set()
for line in open(ARCHIVE / 'manifest.jsonl'):
    r = json.loads(line)
    if r.get('status') != 'downloaded' or not r.get('text_path') or r.get('source_kind') not in KINDS:
        continue
    url = r.get('resolved_url') or r['url']
    if any(s in url.lower() for s in SKIP_PATH) or (r.get('language') or 'en') != 'en':
        skipped['out_of_scope_path_or_language'] += 1
        continue
    path = ARCHIVE / r['text_path']
    if not path.exists():
        skipped['missing_text'] += 1
        continue
    text = path.read_text(errors='replace')
    if hashlib.sha256(text.encode()).hexdigest() != r.get('text_sha256'):
        skipped['text_hash_mismatch'] += 1
        continue
    names = '; '.join(factories.get(f, f) for f in r.get('factory_ids') or [])
    count = 0
    for chunk in chunk_text(text, 2600):
        digest = hashlib.sha256(chunk.encode()).hexdigest()
        if not 200 <= len(chunk) <= 7000 or digest in seen:
            continue
        seen.add(digest)
        db.execute('INSERT INTO chunks(url,source_id,factories,title,text,text_sha256,retrieved_at) VALUES (?,?,?,?,?,?,?)',
                   (url, r['source_id'], names, r.get('title', ''), chunk, r['text_sha256'], r['retrieved_at']))
        count += 1
    if count:
        documents.append({'url': url, 'source_id': r['source_id'], 'factories': r.get('factory_ids'),
                          'kind': r['source_kind'], 'chunks': count, 'text_sha256': r['text_sha256'],
                          'retrieved_at': r['retrieved_at'], 'license_links': r.get('license_links') or []})
    else:
        skipped['no_indexable_chunk'] += 1
db.execute("INSERT INTO search(search) VALUES ('rebuild')")
db.commit()
assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
chunks = db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
db.close()
tmp.replace(OUT / 'factory_docs.sqlite')
(OUT / 'fetch_manifest.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in documents))
summary = {'purpose': 'retrieval_only_not_SFT_training', 'documents': len(documents), 'chunks': chunks,
           'by_factory': dict(Counter(f for d in documents for f in d['factories'] or ['none'])),
           'skipped': dict(skipped), 'evaluation_questions_or_answers_indexed': False,
           'distribution': 'ship fetch_manifest.jsonl + builder; users fetch public pages themselves',
           'sqlite_sha256': hashlib.sha256((OUT / 'factory_docs.sqlite').read_bytes()).hexdigest()}
(OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
