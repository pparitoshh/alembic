"""Document-only retrieval baseline. Evaluation questions/answers are never indexed."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urljoin, urlsplit
from bs4 import BeautifulSoup
from markdownify import markdownify

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path('/Users/vaibhavmangroliya/Downloads/EU_AI_Factory_Documentation_2026-10-13')
OUT = ROOT / 'runs/ai_factory_pilot_20261013/retrieval'
OUT.mkdir(exist_ok=True)
module = ast.parse((ROOT / 'src/distillkit/seeds.py').read_text())
node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'chunk_text')
ns = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), 'chunk_text', 'exec'), ns)
chunk_text = ns['chunk_text']
source = sqlite3.connect('file:' + str(ARCHIVE / 'catalog.sqlite') + '?mode=ro', uri=True)
rows = [json.loads(r[0]) for r in source.execute("SELECT metadata FROM records WHERE status='downloaded' AND source IN ('slurm_upstream','lumi_docs','cineca_hpc') ORDER BY url")]
source.close()
tmp = OUT / 'documents.sqlite.tmp'
if tmp.exists(): tmp.unlink()
db = sqlite3.connect(tmp)
db.executescript('''CREATE TABLE chunks(id INTEGER PRIMARY KEY, url TEXT, site TEXT, title TEXT,
 text TEXT, original_sha256 TEXT, retrieved_at TEXT);
 CREATE VIRTUAL TABLE search USING fts5(text, title, url, content=chunks, content_rowid=id);''')
documents, skipped, seen = [], [], set()
for r in rows:
    if 'html' not in r.get('content_type', ''):
        continue
    url = r.get('resolved_url') or r['url']
    if any(x in urlsplit(url).path.lower() for x in ['/archive/', '/news', '/changelog', '/download', '/legal', '/privacy']):
        continue
    raw = (ARCHIVE / r['original_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != r['sha256']:
        raise RuntimeError('Original hash mismatch: ' + url)
    soup = BeautifulSoup(raw, 'lxml')
    selectors = {'slurm_upstream': 'div.content', 'lumi_docs': 'article.md-content__inner', 'cineca_hpc': '[role=main]'}
    main = soup.select_one(selectors[r['source_id']])
    if main is None:
        skipped.append({'url': url, 'reason': 'missing_expected_article_container'}); continue
    for t in main.select('script,style,nav,footer,#cse,.headerlink,.linenos,.md-content__button,.md-source-file'):
        t.decompose()
    blocks = []
    for pre in main.select('pre'):
        block = pre.get_text().strip()
        token = 'DKCODEBLOCK%05dTOKEN' % len(blocks)
        blocks.append(block)
        p = soup.new_tag('p'); p.string = token; pre.replace_with(p)
    for a in main.select('a[href]'): a['href'] = urljoin(url, a['href'])
    text = markdownify(str(main), heading_style='ATX', bullets='-', strip=['img'])
    for i, block in enumerate(blocks):
        fence = '`' * max(3, 1 + max([0] + [len(x.group()) for x in re.finditer(r'`+', block)]))
        token = 'DKCODEBLOCK%05dTOKEN' % i
        text = text.replace(token, '\n\n' + fence + '\n' + block + '\n' + fence + '\n\n')
        if block and block not in text:
            raise RuntimeError('Code extraction mismatch: ' + url)
    site = {'slurm_upstream':'Slurm', 'lumi_docs':'LUMI', 'cineca_hpc':'CINECA'}[r['source_id']]
    count = 0
    for chunk in chunk_text(text, 2600):
        digest = hashlib.sha256(chunk.encode()).hexdigest()
        if not 200 <= len(chunk) <= 7000 or digest in seen: continue
        seen.add(digest)
        db.execute('INSERT INTO chunks(url,site,title,text,original_sha256,retrieved_at) VALUES (?,?,?,?,?,?)',
                   (url,site,r.get('title',''),chunk,r['sha256'],r['retrieved_at']))
        count += 1
    if count: documents.append({'url':url,'site':site,'chunks':count,'sha256':r['sha256']})
db.execute("INSERT INTO search(search) VALUES ('rebuild')")
db.commit()
assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
count = db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
db.close(); tmp.replace(OUT / 'documents.sqlite')
manifest = {'purpose':'retrieval_only_not_SFT_training', 'source_documents':documents, 'chunks':count,
            'evaluation_questions_or_answers_indexed':False,
            'includes_public_docs_for_training_heldout_topics':True,
            'comparison_label':'document-assisted evaluation, separate from closed-book v2/v3',
            'limitations':['Lexical BM25 baseline, not an optimized retriever.',
                           'CINECA and LUMI are routed by named site; generic Slurm questions use upstream Slurm.',
                           'Snapshot facts can age; preserve retrieval date and URL.'], 'skipped':skipped}
(OUT / 'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'indexed_documents':len(documents),'chunks':count,'skipped':len(skipped)}))
