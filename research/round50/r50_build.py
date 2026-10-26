"""Round 50 (GPU/MPI push, Oct 25) step 1: questions for "better of two" training. GPU/MPI/performance and software-
environment questions from the verified banks (factory v26+, held-out evaluation families and near-eval questions
excluded), each with the excerpts the app retrieves (variant e: retriever b + relevance header) and its verified
reference answer. Step 2 (gen_two.py) has round 27 answer each one with and without the excerpts.

usage: r50_build.py OUT_DIR [N]
"""
import glob
import json
import os
import random
import re
import sys

C = '/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/code'
S = '/leonardo_work/EUHPC_D30_031/shared_split/code'
os.environ['RAG_VARIANT'] = 'e'
sys.path.insert(0, C + '/research/rag65')
import retrieve  # noqa: E402

out, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000
os.makedirs(out, exist_ok=True)
rng = random.Random(50)
HELD = ('open-mpi', 'openmpi.org', 'aalto', 'cuda-samples', 'cuda_samples')
TOPIC = re.compile(r'\b(gpu|gpus|cuda|nvcc|nccl|rocm|hip|openacc|mpi|mpirun|mpiexec|srun|openmp|omp_|hybrid|rank|ranks|'
                   r'profil\w*|nsight|nsys|ncu|perf|vtune|scaling|speedup|module|modules|lmod|spack|easybuild|conda|'
                   r'venv|pip|container|apptainer|singularity|compil\w+|gcc|nvhpc|linker|ld_library_path|scratch|'
                   r'lustre|filesystem|quota|i/o|hdf5)\b', re.I)


def grams(q):
    w = re.findall(r'[a-z0-9]+', q.lower())
    return {' '.join(w[i:i + 3]) for i in range(len(w) - 2)}


evals = []
for p in ['data/eval/eval_all_v2.jsonl', 'data/eval/eval_v3.jsonl', 'data/eval/dataset_v1/development_normal.jsonl',
          'data/eval/dataset_v1/final_normal.jsonl']:
    evals += [grams(json.loads(line)['question']) for line in open(os.path.join(C, p))]


def near_eval(q):
    g = grams(q)
    return any(len(g & e) / max(1, len(g | e)) >= 0.5 for e in evals if e)


cands, seen = [], set()
for f in sorted(glob.glob(S + '/runs/ai_factory_all_v*/release/verified.jsonl')):
    v = int(re.search(r'_v(\d+)/', f).group(1))
    if v < 26:
        continue
    for line in open(f):
        r = json.loads(line)
        q = (r.get('question') or '').split('\n\nProvided input')[0].strip()
        ref = next((m['content'] for m in r.get('messages', []) if m['role'] == 'assistant'), '')
        if r.get('mode') != 'prose' or not q or not ref or len(q) > 700 or q in seen or not TOPIC.search(q):
            continue
        if any(h in (r.get('doc_id') or '').lower() or h in (r.get('source_url') or '').lower() for h in HELD):
            continue
        seen.add(q)
        cands.append((r['id'], q, ref))
rng.shuffle(cands)
P, F = C + '/runs/rag65/indexes/documents.sqlite', C + '/runs/rag65/indexes/factory_docs.sqlite'
n = 0
with open(f'{out}/questions.jsonl', 'w') as fo:
    for rid, q, ref in cands:
        if n >= N:
            break
        if near_eval(q):
            continue
        docs = retrieve.retrieve(q, P, F)
        fo.write(json.dumps({'id': 'r50-' + rid, 'question': q, 'reference': ref,
                             'rag_prompt': retrieve.with_context(q, docs)}, ensure_ascii=False) + '\n')
        n += 1
print(f'BUILD READY: {n} questions (from {len(cands)} GPU/MPI/environment candidates)')
