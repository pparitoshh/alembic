import ast, sys
p = sys.argv[1]; s = open(p).read()
a = "for label, path in [('v2', 'data/eval/eval_all_v2.jsonl'), ('v3', 'data/eval/eval_v3.jsonl'), ('d1', 'data/eval/dataset_v1/development_normal.jsonl')]:"
b = """# EVAL_DIR (opt-in, Oct 25): judge against reviewed sets EVAL_DIR/{v2,v3,d1}.jsonl (duplicates dropped, references fixed).
SETS = [('v2', 'data/eval/eval_all_v2.jsonl'), ('v3', 'data/eval/eval_v3.jsonl'), ('d1', 'data/eval/dataset_v1/development_normal.jsonl')]
if os.environ.get('EVAL_DIR'):
    SETS = [(label, os.path.join(os.environ['EVAL_DIR'], label + '.jsonl')) for label, _ in SETS]
for label, path in SETS:"""
assert s.count(a) == 1
s = s.replace(a, b); ast.parse(s); open(p + '.new', 'w').write(s); print('ok')
