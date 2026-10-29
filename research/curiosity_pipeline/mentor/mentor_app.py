"""Issue #115 follow-up: the app's answers to the Leonardo mentor cases with three prompt orders, on curiosity.
A = the app as it is (dev); B = ClusterPilot notes first, then the other excerpts; C = B with the excerpts cut to
16,000 characters. ClusterPilot-4B 1.3 on vLLM :8100.  usage: mentor_app.py CASES OUT_DIR"""
import concurrent.futures as cf, io, json, os, sys, time
sys.path.insert(0, os.path.expanduser('~/cp/product/src'))
from distillkit.assistant import Assistant  # noqa: E402
INDEX, CAND = os.path.expanduser('~/cp/product_index'), 'http://127.0.0.1:8100/v1'
NOTE = '/index/notes/'


class NotesFirst(Assistant):
    budget = None

    def search(self, question, k=None):
        found = super().search(question, k)
        found = [d for d in found if NOTE in d['url']] + [d for d in found if NOTE not in d['url']]
        if self.budget:
            kept, used = [], 0
            for d in found:
                if used + len(d['text']) <= self.budget:
                    kept.append(d); used += len(d['text'])
            found = kept
        return found


class Short(NotesFirst):
    budget = 16000


def answer(args):
    variant, case = args
    bot = {'A': Assistant, 'B': NotesFirst, 'C': Short}[variant](INDEX, model='cand', openai=CAND)
    q = case['user_request'] + ('\n\nContext: ' + case['context'] if case.get('context') else '')
    t = time.time()
    ans, sources = bot.ask(q, out=io.StringIO())
    return variant, {'id': case['id'], 'answer': ans, 'sources': sources, 'seconds': round(time.time() - t, 1)}


cases = [json.loads(l) for l in open(sys.argv[1])]
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
res = {v: [] for v in 'ABC'}
with cf.ThreadPoolExecutor(12) as ex:
    for v, row in ex.map(answer, [(v, c) for v in 'ABC' for c in cases]):
        res[v].append(row)
for v, rows in res.items():
    with open(f'{out}/app_{v}.jsonl', 'w') as f:
        for r in sorted(rows, key=lambda r: r['id']):
            f.write(json.dumps(r) + '\n')
    print(v, 'notes first in prompt:', sum(NOTE in (r['sources'] or [''])[0] for r in rows), '/', len(rows), flush=True)
