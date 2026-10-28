"""Training prompts for teacher distillation on Curiosity: questions from every verified factory bank (prose rows),
held-out evaluation families and near-eval questions (3-gram Jaccard >= 0.5 with any test question) excluded,
deduplicated, stratified by domain, each with the app's excerpts (retriever e) in the user turn.

usage: build_prompts.py OUT.jsonl N [SEED]   (run from the repo root; RAG_VARIANT defaults to e; PROMPT_MIX=slurm,gpu_mpi_env,other
shares, default 0.4,0.4,0.2; BANK_GLOB=<pattern> limits the banks read;
PROMPT_REGEX=<regex> keeps only matching questions, all domains, up to N)
"""
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

os.environ.setdefault('RAG_VARIANT', 'e')
sys.path.insert(0, 'research/rag65')
import retrieve  # noqa: E402

out, N = sys.argv[1], int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
BANKS = os.path.expanduser('~/cp/data/banks')
HELD = ('aalto', 'open-mpi', 'openmpi', 'cuda-samples', 'cuda_samples')
SLURM = re.compile(r'\b(slurm|sbatch|srun|salloc|squeue|scancel|scontrol|sacct|sinfo|sstat|sprio|sshare|partition|'
                   r'job array|array job|requeue\w*|preempt\w*|dependency|afterok|afterany|qos|fairshare|gres|'
                   r'#sbatch|time limit|walltime)\b', re.I)
GPUMPI = re.compile(r'\b(gpu|gpus|cuda|nvcc|nccl|rocm|hip|openacc|mpi|mpirun|mpiexec|openmp|omp_|hybrid|rank|ranks|'
                    r'profil\w*|nsight|nsys|ncu|vtune|scaling|speedup|module|modules|lmod|spack|easybuild|conda|'
                    r'venv|pip|container|apptainer|singularity|compil\w+|gcc|nvhpc|linker|ld_library_path|scratch|'
                    r'lustre|filesystem|quota|i/o|hdf5|interactive|jupyter)\b', re.I)
TOPIC = re.compile(os.environ['PROMPT_REGEX'], re.I) if os.environ.get('PROMPT_REGEX') else None  # topic round: matching questions only
MIX = dict(zip(('slurm', 'gpu_mpi_env', 'other'), map(float, os.environ.get('PROMPT_MIX', '0.4,0.4,0.2').split(','))))  # shares


def grams(q):
    w = re.findall(r'[a-z0-9]+', q.lower())
    return {' '.join(w[i:i + 3]) for i in range(len(w) - 2)}


evals = []
for p in ['data/eval/eval_all_v2.jsonl', 'data/eval/eval_v3.jsonl', 'data/eval/dataset_v1/development_normal.jsonl',
          'data/eval/dataset_v1/final_normal.jsonl', 'data/eval/clean/d1.jsonl', 'data/eval/clean/v2.jsonl',
          'data/eval/clean/v3.jsonl']:
    evals += [grams(json.loads(line)['question']) for line in open(p)]
index = defaultdict(set)
for i, g in enumerate(evals):
    for x in g:
        index[x].add(i)


def near_eval(q):
    g = grams(q)
    hits = Counter(i for x in g for i in index.get(x, ()))
    return any(len(g & evals[i]) / max(1, len(g | evals[i])) >= 0.5 for i, n in hits.items() if n >= 0.3 * len(g))


seen, pool = set(), defaultdict(list)
for f in sorted(glob.glob(BANKS + '/' + os.environ.get('BANK_GLOB', '*.jsonl'))):  # BANK_GLOB: one batch's bank only
    for line in open(f):
        r = json.loads(line)
        q = (r.get('question') or '').split('\n\nProvided input')[0].strip()
        if r.get('mode') != 'prose' or not q or len(q) > 700 or q in seen:
            continue
        where = ' '.join(str(r.get(k) or '') for k in ('doc_id', 'source_url', 'source_site', 'source_family')).lower()
        if any(h in where for h in HELD):
            continue
        if TOPIC and not TOPIC.search(q):
            continue
        seen.add(q)
        ref = next((m['content'] for m in r.get('messages', []) if m['role'] == 'assistant'), '')
        dom = 'slurm' if SLURM.search(q) else 'gpu_mpi_env' if GPUMPI.search(q) else 'other'
        pool[dom].append((r['id'], q, ref))
rng = random.Random(seed)
picked = []
if TOPIC:  # take every matching question, whatever its domain, then cap at N
    MIX = {d: 1.0 for d in MIX}
for dom, share in MIX.items():
    c = pool[dom]
    rng.shuffle(c)
    want, k = int(N * share), 0
    for rid, q, ref in c:
        if k >= want:
            break
        if near_eval(q):
            continue
        picked.append({'id': rid, 'domain': dom, 'question': q, 'reference': ref})
        k += 1
rng.shuffle(picked)
picked = picked[:N]
print('candidates', {d: len(v) for d, v in pool.items()}, 'picked', Counter(p['domain'] for p in picked), flush=True)
P, F = 'runs/rag65/indexes/documents.sqlite', 'runs/rag65/indexes/factory_docs.sqlite'


def ctx(p):
    return {**p, 'user': retrieve.with_context(p['question'], retrieve.retrieve(p['question'], P, F))}


with Pool(min(96, os.cpu_count() or 8)) as ex, open(out, 'w') as fo:
    for row in ex.imap(ctx, picked, chunksize=64):
        fo.write(json.dumps(row, ensure_ascii=False) + '\n')
print('PROMPTS READY', len(picked), out)
