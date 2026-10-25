"""Round 47 data: questions paired with the excerpts the improved retriever (RAG_VARIANT=b) really returns for them,
for on-policy distillation. 6,000 non-Slurm HPC questions from the general factory batches (v5+), where those excerpts
are mostly irrelevant Slurm/LUMI/CINECA pages (the d1 failure mode: the model comments on the excerpts or follows them
instead of answering), plus 2,000 Slurm questions from round 31's data, whose excerpts are relevant. Rows from the
held-out evaluation source families (Aalto, Open MPI, CUDA Samples) and questions close to an evaluation question are
left out. The assistant turn is a placeholder: the student's own answer replaces it in the generation step.

usage: build_distractor.py OUT.jsonl
"""
import glob
import json
import os
import random
import re
import sys

C = '/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/code'
S = '/leonardo_work/EUHPC_D30_031/shared_split/code'
os.environ['RAG_VARIANT'] = 'b'
sys.path.insert(0, C + '/research/rag65')
import retrieve  # noqa: E402

out = sys.argv[1]
N_GEN, N_SLURM = int(os.environ.get('N_GEN') or 6000), int(os.environ.get('N_SLURM') or 2000)
PREFIX = os.environ.get('PREFIX') or 'r47'
rng = random.Random(int(os.environ.get('SEED') or 47))
# EXCLUDE_IDS (opt-in): source files of earlier rounds; their questions are not reused
used = set()
for x in filter(None, (os.environ.get('EXCLUDE_IDS') or '').split(',')):
    used |= {json.loads(l)['id'].split('-', 1)[1] for l in open(x)}
HELD = ('open-mpi', 'aalto', 'cuda-samples', 'cuda_samples')
COMMANDS = ['sbatch', 'srun', 'salloc', 'sacct', 'scontrol', 'squeue', 'scancel', 'sinfo', 'sstat', 'sprio',
            'sshare', 'sacctmgr', 'sreport', 'strigger', 'sattach', 'sbcast']


def slurm_explicit(q):
    low = q.lower()
    return 'slurm' in low or any(re.search(r'\b' + c + r'\b', low) for c in COMMANDS) or \
        any(s in low for s in ['lumi', 'leonardo', 'cineca', 'marconi'])


def grams(q):
    w = re.findall(r'[a-z0-9]+', q.lower())
    return {' '.join(w[i:i + 3]) for i in range(len(w) - 2)}


evals = []
for p in ['data/eval/eval_all_v2.jsonl', 'data/eval/eval_v3.jsonl', 'data/eval/dataset_v1/development_normal.jsonl',
          'data/eval/dataset_v1/final_normal.jsonl']:
    evals += [grams(json.loads(l)['question']) for l in open(os.path.join(C, p))]


def near_eval(q):
    g = grams(q)
    return any(len(g & e) / max(1, len(g | e)) >= 0.5 for e in evals if e)


general = []
for f in sorted(glob.glob(S + '/runs/ai_factory_all_v*/release/verified.jsonl')) + \
        sorted(glob.glob(C + '/runs/ai_factory_all_v[5-8]/release/verified.jsonl')):
    for line in open(f):
        r = json.loads(line)
        q = (r.get('question') or '').split('\n\nProvided input')[0].strip()
        if r.get('mode') != 'prose' or not q or len(q) > 600:
            continue
        if any(h in (r.get('doc_id') or '').lower() or h in (r.get('source_url') or '').lower() for h in HELD):
            continue
        if slurm_explicit(q) or re.search(r'provided (input|text|excerpt|output|log|script|file|snippet|code)|the input', q.lower()):
            continue
        general.append((r['id'], q))
rng.shuffle(general)
slurm = []
for line in open(S + '/runs/rb31/train/seed42/verified.jsonl'):
    r = json.loads(line)
    if any(h in json.dumps(r).lower()[:2000] for h in HELD):
        continue
    slurm.append(r)
rng.shuffle(slurm)

P, F = C + '/runs/rag65/indexes/documents.sqlite', C + '/runs/rag65/indexes/factory_docs.sqlite'
rows, seen = [], set()
for rid, q in general:
    if len(rows) >= N_GEN:
        break
    if q in seen or rid in used or near_eval(q):
        continue
    seen.add(q)
    docs = retrieve.retrieve(q, P, F)
    rows.append({'id': PREFIX + 'g-' + rid, 'messages': [{'role': 'user', 'content': retrieve.with_context(q, docs)},
                                                   {'role': 'assistant', 'content': '.'}], 'kind': 'general'})
n_g = len(rows)
for r in [r for r in slurm if r['id'] not in used][:N_SLURM]:
    rows.append({'id': PREFIX + 's-' + r['id'], 'messages': [r['messages'][0], {'role': 'assistant', 'content': '.'}],
                 'kind': 'slurm'})
rng.shuffle(rows)
with open(out, 'w') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'{len(rows)} rows: {n_g} general (from {len(general)} candidates), {len(rows) - n_g} Slurm')
