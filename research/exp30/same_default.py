"""Check: new retrieve.py with RAG_VARIANT unset returns exactly what the original returns, for all eval rows."""
import importlib.machinery, importlib.util, json, os, sys
os.environ.pop('RAG_VARIANT', None)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path, loader=importlib.machinery.SourceFileLoader(name, path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old, new = load('old', sys.argv[1]), load('new', sys.argv[2])
P, F = 'runs/rag65/indexes/documents.sqlite', 'runs/rag65/indexes/factory_docs.sqlite'
n = bad = 0
qs = [json.loads(x)['question'] for p in ['data/eval/eval_all_v2.jsonl', 'data/eval/eval_v3.jsonl', 'data/eval/dataset_v1/development_normal.jsonl'] for x in open(p)]
qs += ['How do I submit a GPU job on MeluXina?', 'Leonardo booster partition QOS?', 'LUMI-G job script with 8 GPUs', 'Jean Zay A100 partition', '???']
for q in qs:
    a, b = old.retrieve(q, P, F), new.retrieve(q, P, F)
    n += 1; bad += a != b or old.with_context(q, a) != new.with_context(q, b)
print('default-identical checked', n, 'mismatches', bad)
