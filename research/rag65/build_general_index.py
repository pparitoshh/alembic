"""Retrieval index over the general HPC documentation crawls (RAG_VARIANT=c): every downloaded page of the given
archives, chunked like the pilot index (~2,200 characters on paragraph boundaries), in the same SQLite FTS5 schema
(chunks + search(text, title, url)). URLs on the evaluation-source exclusion list are left out, so no test material is
retrievable. usage: build_general_index.py OUT.sqlite EXCLUDE_URLS.txt ARCHIVE_DIR...
"""
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

out, exclude, *archives = sys.argv[1:]
skip = {u.strip().rstrip('/') for u in open(exclude) if u.strip()}
if os.path.exists(out):
    os.remove(out)
db = sqlite3.connect(out)
db.execute('CREATE TABLE chunks(id INTEGER PRIMARY KEY, url TEXT, site TEXT, title TEXT, text TEXT, original_sha256 TEXT, retrieved_at TEXT)')
db.execute('CREATE VIRTUAL TABLE search USING fts5(text, title, url, content=chunks, content_rowid=id)')


def chunks(text, size=2200):
    out, cur = [], ''
    for para in re.split(r'\n\s*\n', text):
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) + 2 > size and cur:
            out.append(cur)
            cur = ''
        while len(para) > size:
            out.append(para[:size])
            para = para[size:]
        cur = (cur + '\n\n' + para) if cur else para
    if cur:
        out.append(cur)
    return [c for c in out if len(c) >= 200]


n_doc = n_chunk = n_skip = 0
seen = set()
for arch in archives:
    for line in open(Path(arch) / 'manifest.jsonl'):
        r = json.loads(line)
        if r.get('status') != 'downloaded' or not r.get('text_path'):
            continue
        url = (r.get('resolved_url') or r['url']).rstrip('/')
        if url in skip or r['url'].rstrip('/') in skip:
            n_skip += 1
            continue
        if r.get('text_sha256') in seen:
            continue
        seen.add(r.get('text_sha256'))
        p = Path(arch) / r['text_path']
        if not p.exists():
            continue
        site = r.get('source_id', '').replace('_', ' ')
        for c in chunks(p.read_text(errors='replace')):
            cur = db.execute('INSERT INTO chunks(url, site, title, text, original_sha256, retrieved_at) VALUES (?,?,?,?,?,?)',
                             (r['url'], site, r.get('title') or '', c, r.get('sha256'), (r.get('retrieved_at') or '')[:10]))
            db.execute('INSERT INTO search(rowid, text, title, url) VALUES (?,?,?,?)', (cur.lastrowid, c, r.get('title') or '', r['url']))
            n_chunk += 1
        n_doc += 1
db.commit()
db.execute("INSERT INTO search(search) VALUES('optimize')")
db.commit()
print(f'INDEX READY: {n_doc} documents, {n_chunk} chunks, {n_skip} evaluation-source URLs excluded -> {out}')
