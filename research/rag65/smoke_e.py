import os
import sys

sys.path.insert(0, 'research/rag65')
import retrieve as r  # noqa: E402

P, F = 'runs/rag65/indexes/documents.sqlite', 'runs/rag65/indexes/factory_docs.sqlite'
q = 'On Triton, how do I save my current module collection?'
os.environ['RAG_VARIANT'] = 'b'
db = r.retrieve(q, P, F)
cb = r.with_context(q, db)
os.environ['RAG_VARIANT'] = 'e'
de = r.retrieve(q, P, F)
ce = r.with_context(q, de)
print('same excerpts:', [d['url'] for d in db] == [d['url'] for d in de], len(de))
print('b header:', cb[len(q):len(q) + 60].strip())
print('e header:', ce[len(q):len(q) + 280].strip())
os.environ['RAG_VARIANT'] = ''
print('original unchanged:', r.with_context('x', [{'url': 'u', 'site': 's', 'retrieved_at': 't', 'text': 'y'}])[:50])
