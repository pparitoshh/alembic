"""exp30 offline retrieval analysis (CPU only, no model). usage: retrieval_recall.py RETRIEVE_DIR VARIANT OUT_JSON

Gold signal per eval row:
  * key-term recall (all sets): key terms = reference-answer tokens (words, --long-options, %X patterns) that are not in
    the question, have >= 3 chars, occur in the pilot corpus, and occur in < 20% of its chunks (informative, not
    generic English). term_recall@k = share of a row's key terms found in the concatenated top-k excerpts;
    hit50@k = share of rows with term_recall@k >= 0.5. Rows without key terms are skipped.
  * gold-page hit@k (v3 only): the v3 manifest names the Slurm manual chunks each topic was written from
    (slurm_man_sbatch#17 -> https://slurm.schedmd.com/sbatch.html); hit = any of the top-k URLs is one of those pages.
"""
import json
import os
import re
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, sys.argv[1])
os.environ['RAG_VARIANT'] = '' if sys.argv[2] == 'cur' else sys.argv[2]
import retrieve as _R  # noqa: E402
from retrieve import retrieve  # noqa: E402
for _n in ('VARIANT_K', 'VARIANT_BUDGET', 'VARIANT_GENERIC'):  # grid search overrides (analysis only)
    if os.environ.get('EXP_' + _n):
        setattr(_R, _n, int(os.environ['EXP_' + _n]))

PILOT = 'runs/rag65/indexes/documents.sqlite'
FACTORY = 'runs/rag65/indexes/factory_docs.sqlite'
TOK = re.compile(r'--[a-z][a-z0-9-]*|%[a-z]\b|[a-z][a-z0-9_]*')


def toks(text):
    return set(TOK.findall(text.lower()))


db = sqlite3.connect('file:' + PILOT + '?mode=ro', uri=True)
texts = [t for (t,) in db.execute('SELECT text FROM chunks')]
df = Counter(t for x in texts for t in toks(x))
N = len(texts)
pages = {}
for d in json.load(open('data/eval/v3_sources/sources_manifest.json'))['documents']:
    urls = set()
    for s in d['source_chunks']:
        m = re.match(r'slurm_(?:man|guide)_([a-z_]+)#', s)
        if m:
            urls.add('https://slurm.schedmd.com/' + m.group(1) + '.html')
    pages[d['doc_id']] = urls

KS = [1, 2, 3, 5, 8]
res = {}
for label, path in [('v2', 'data/eval/eval_all_v2.jsonl'), ('v3', 'data/eval/eval_v3.jsonl'),
                    ('d1', 'data/eval/dataset_v1/development_normal.jsonl')]:
    rows = [json.loads(x) for x in open(path)]
    agg = {k: [0.0, 0, 0] for k in KS}
    page = {k: [0, 0] for k in KS}
    n_exc, chars = [], []
    for r in rows:
        docs = retrieve(r['question'], PILOT, FACTORY)
        n_exc.append(len(docs))
        chars.append(sum(len(d['text']) for d in docs))
        ref = re.sub(r'\n*Source: eval_v3_.*$', '', r['reference'], flags=re.S)
        key = {t for t in toks(ref) - toks(r['question']) if len(t) >= 3 and 0 < df[t] < 0.2 * N}
        for k in KS:
            got = set().union(*[toks(d['text']) for d in docs[:k]]) if docs else set()
            if key:
                rec = len(key & got) / len(key)
                agg[k][0] += rec
                agg[k][1] += rec >= 0.5
                agg[k][2] += 1
            if r['doc_id'] in pages:
                page[k][0] += any(d['url'] in pages[r['doc_id']] for d in docs[:k])
                page[k][1] += 1
    res[label] = {
        'rows': len(rows), 'mean_excerpts': sum(n_exc) / len(rows), 'mean_chars': sum(chars) / len(rows),
        'max_chars': max(chars),
        'term_recall': {k: round(a[0] / a[2], 4) for k, a in agg.items()},
        'hit50': {k: round(a[1] / a[2], 4) for k, a in agg.items()},
        'rows_with_key_terms': agg[1][2],
        'gold_page_hit': {k: round(p[0] / p[1], 4) for k, p in page.items()} if page[1][1] else None,
    }
    print(sys.argv[2], label, json.dumps(res[label]), flush=True)
json.dump(res, open(sys.argv[3], 'w'), indent=1)
