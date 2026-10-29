"""Questions that name a cluster our docs do not cover (from the Oct 28 sweeps), answered by dev (~/cp/product) and by
the uncovered-site change (~/cp/product_us) on the same model server, each answer reviewed by Qwen3-235B.
usage: qa_uncovered.py OUT_DIR"""
import concurrent.futures as cf, importlib, io, json, os, sys
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
CAND = 'http://127.0.0.1:8100/v1'
sys.argv = [sys.argv[0], OUT]
sys.path.insert(0, os.path.expanduser('~/cp/bin'))
import qa_sweep  # REVIEW prompt and review()
def load(path):
    sys.path.insert(0, path + '/src')
    for m in [m for m in sys.modules if m.startswith('distillkit')]:
        del sys.modules[m]
    mod = importlib.import_module('distillkit.assistant')
    sys.path.pop(0)
    return mod
new = load(os.path.expanduser('~/cp/product_us'))
qs = []
for f in ('run2', 'eu'):
    for cat, q in json.load(open(os.path.expanduser(f'~/cp/qa_sweep/{f}/questions.json'))):
        if new.uncovered_site(q):
            qs.append((cat, q))
print(len(qs), 'questions name an uncovered cluster', flush=True)
rows = []
for label, path in (('dev', '~/cp/product'), ('new', '~/cp/product_us')):
    mod = load(os.path.expanduser(path))
    def answer(item, mod=mod, label=label):
        cat, q = item
        buf = io.StringIO()
        ans, _ = mod.Assistant(os.path.expanduser('~/cp/product_index'), model='cand', openai=CAND).ask(q, out=buf)
        return {'variant': label, 'category': cat, 'question': q, 'answer': ans, 'shown': buf.getvalue()[:300]}
    with cf.ThreadPoolExecutor(12) as ex:
        rows += list(ex.map(answer, qs))
with cf.ThreadPoolExecutor(16) as ex:
    rows = list(ex.map(qa_sweep.review, rows))
with open(OUT + '/results.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
from collections import Counter
for v in ('dev', 'new'):
    print(v, Counter(r['review']['verdict'] for r in rows if r['variant'] == v), flush=True)
