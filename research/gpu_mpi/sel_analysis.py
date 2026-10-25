import glob, json, os, statistics as st
C = '/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/code'
def sc(v): return ((v[0] == 'A') + (v[0] == 'T') * .5 + (v[1] == 'B') + (v[1] == 'T') * .5) / 2
def load(d, arm):
    out = {}
    for f in glob.glob(f'{d}/*__{arm}__*.jsonl'):
        label = os.path.basename(f).split('__')[0]
        for l in open(f):
            r = json.loads(l); out[(label, r['id'])] = sc(r['judge']['arm_vs_base'])
    return out
rag = load(C + '/runs/vsb_r27e/judged_majority', 's42_rag')
closed = load(C + '/runs/gate27/judged_majority', 'r27_closed')
sel = {(r['set'], r['id']): r for r in json.load(open(C + '/runs/gate27/self_select.json'))}
for lab in ('v2', 'v3', 'd1'):
    ks = [k for k in sel if k[0] == lab and k in rag and k in closed]
    r = st.mean(rag[k] for k in ks); s = st.mean((closed[k] if sel[k]['pick'] == 'closed' else rag[k]) for k in ks)
    o = st.mean(max(rag[k], closed[k]) for k in ks); n = sum(sel[k]['pick'] == 'closed' for k in ks)
    right = sum((sel[k]['pick'] == 'closed') == (closed[k] > rag[k]) for k in ks if closed[k] != rag[k])
    diff = sum(closed[k] != rag[k] for k in ks)
    print(f'{lab}: rag {r:.3f} -> self-select {s:.3f} ({s - r:+.3f}; {n} closed-book picks) | ceiling {o:.3f} | right choice on {right}/{diff} questions where it matters')
ks = [k for k in sel if k[0] == 'd1' and k[1].startswith('dev')]
print('d1 Triton: rag %.3f -> self-select %.3f' % (st.mean(rag[k] for k in ks), st.mean((closed[k] if sel[k]['pick'] == 'closed' else rag[k]) for k in ks)))
import collections; print('raw votes', collections.Counter((r['order1'], r['order2']) for r in sel.values()).most_common(5))
