import glob, json, os, re, sys, statistics as st
C = '/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/code'
os.environ['RAG_VARIANT'] = 'b'
sys.path.insert(0, C + '/research/rag65')
import retrieve
def sc(v): return ((v[0] == 'A') + (v[0] == 'T') * .5 + (v[1] == 'B') + (v[1] == 'T') * .5) / 2
def load(d, arm):
    out = {}
    for f in glob.glob(f'{d}/*__{arm}__*.jsonl'):
        label = os.path.basename(f).split('__')[0]
        for l in open(f):
            r = json.loads(l); out[(label, r['id'])] = (sc(r['judge']['arm_vs_base']), r['question'])
    return out
rag = load(C + '/runs/vsb_r27e/judged_majority', 's42_rag')
closed = load(C + '/runs/gate27/judged_majority', 'r27_closed')
keys = sorted(set(rag) & set(closed))
P, F = C + '/runs/rag65/indexes/documents.sqlite', C + '/runs/rag65/indexes/factory_docs.sqlite'
STOP = retrieve.STOP | {'with', 'that', 'this', 'have', 'from', 'what', 'should', 'would', 'there', 'their', 'when', 'into', 'which', 'only', 'does', 'cluster', 'node', 'nodes', 'using', 'want', 'need', 'about', 'after', 'before'}
feat = {}
for k in keys:
    q = closed[k][1]
    words = {w for w in re.findall(r'[a-z0-9_]{4,}', q.lower()) if w not in STOP}
    docs = retrieve.retrieve(q, P, F)
    text = ' '.join(d['text'].lower() for d in docs[:3])
    feat[k] = len([w for w in words if w in text]) / max(1, len(words))
by = lambda lab: [k for k in keys if k[0] == lab]
print('per set: n, rag, closed, oracle')
for lab in ('v2', 'v3', 'd1'):
    ks = by(lab)
    r = st.mean(rag[k][0] for k in ks); c = st.mean(closed[k][0] for k in ks); o = st.mean(max(rag[k][0], closed[k][0]) for k in ks)
    print(f'{lab}: {len(ks)}  rag {r:.3f}  closed {c:.3f}  oracle {o:.3f}')
ks = [k for k in by('d1') if k[1].startswith('dev')]
print('d1 Triton part:', len(ks), 'rag %.3f closed %.3f' % (st.mean(rag[k][0] for k in ks), st.mean(closed[k][0] for k in ks)))
print('\nrule: closed-book when excerpt coverage of the question words < t')
for t in (0.3, 0.4, 0.5, 0.6, 0.7):
    row = []
    for lab in ('v2', 'v3', 'd1'):
        ks = by(lab)
        g = st.mean((closed[k][0] if feat[k] < t else rag[k][0]) for k in ks)
        n = sum(feat[k] < t for k in ks)
        row.append(f'{lab} {g:.3f} ({n} closed)')
    print(f't={t}: ' + ' | '.join(row))
