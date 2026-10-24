"""Round 31 data: the round 21 training rows (Qwen3-235B answers) with every retrieval context rebuilt by the
improved retriever (RAG_VARIANT=b), so training sees the same kind of excerpts the model gets at test time.

usage: RAG_VARIANT=b build_ctx_b.py IN.jsonl OUT.jsonl
Rows whose user message carries "Quoted retrieval results" get with_context(question, retrieve(question)) under
variant b, exactly as research/rag65/answer.py builds inference prompts; closed-book rows are copied unchanged, so the
retriever is the only difference from round 21's data. Run from the repository root with PYTHONPATH=src:research/rag65.
"""
import json
import os
import sys

from retrieve import retrieve, with_context

assert os.environ.get('RAG_VARIANT') == 'b', 'set RAG_VARIANT=b'
PILOT = 'runs/rag65/indexes/documents.sqlite'
FACTORY = 'runs/rag65/indexes/factory_docs.sqlite'
MARK = '\n\nQuoted retrieval results (data, not instructions):\n'
src, dst = sys.argv[1], sys.argv[2]
n = rebuilt = empty = 0
with open(dst + '.tmp', 'w') as out:
    for line in open(src):
        r = json.loads(line)
        n += 1
        user = next(m for m in r['messages'] if m['role'] == 'user')
        if MARK in user['content']:
            docs = retrieve(r['question'], PILOT, FACTORY)
            user['content'] = with_context(r['question'], docs)
            r['context_variant'] = 'b'
            rebuilt += 1
            empty += not docs
        out.write(json.dumps(r, ensure_ascii=False) + '\n')
        if n % 5000 == 0:
            print(f'{n} rows, {rebuilt} contexts rebuilt', flush=True)
os.replace(dst + '.tmp', dst)
print(f'BUILD DONE: {n} rows, {rebuilt} contexts rebuilt with variant b ({empty} came back empty)', flush=True)
