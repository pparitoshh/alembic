"""Retrieval index over the general HPC documentation crawls (RAG_VARIANT=c): every downloaded page of the given
archives, chunked like the pilot index (~2,200 characters on paragraph boundaries), in the same SQLite FTS5 schema
(chunks + search(text, title, url)). URLs on the evaluation-source exclusion list and every page of the held-out
evaluation source families are left out, so no test material is retrievable. usage: build_general_index.py OUT.sqlite EXCLUDE_URLS.txt ARCHIVE_DIR...
"""
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

out, exclude, *archives = sys.argv[1:]
# Loose URL match (scheme, www., query, fragment, trailing slash, /index.html, case): on Oct 24 an exact match missed 6
# evaluation-source pages written as .../index.html or with www.
def norm(u):
    return re.sub(r'^https?://(www\.)?', '', u.split('#')[0].split('?')[0]).rstrip('/').lower().replace('/index.html', '')


skip = {norm(u.strip()) for u in open(exclude) if u.strip()}
# Held-out source families of the evaluation sets (data/eval/dataset_v1/evidence/evaluation_family_assignments.json:
# Aalto Triton docs for the development split, Open MPI and NVIDIA CUDA Samples for the final split) are held out as
# whole sites, not page by page: any page of these sites, in any version, is left out.
HELD_OUT = ('scicomp.aalto.fi', 'aaltoscicomp.github.io', 'github.com/aaltoscicomp', 'githubusercontent.com/aaltoscicomp',
            'open-mpi.org', 'github.com/open-mpi/ompi', 'githubusercontent.com/open-mpi/ompi',
            'github.com/nvidia/cuda-samples', 'githubusercontent.com/nvidia/cuda-samples')


def held_out(u):
    return any(h in norm(u) for h in HELD_OUT)
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
        if norm(url) in skip or norm(r['url']) in skip or held_out(url) or held_out(r['url']):
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
print(f'INDEX READY: {n_doc} documents, {n_chunk} chunks, {n_skip} evaluation-source pages excluded -> {out}')
