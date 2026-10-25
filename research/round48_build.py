"""Round 48 data (d1 push, Oct 25): Slurm questions on requeue, signals, preemption, checkpointing and job arrays (d1's
requeue/signals part: round 27 0.785 vs Qwen3-235B 0.924; arrays 0.850 vs 0.933), taken from the verified Q&A banks
(no new generation), plus general Slurm questions from round 31's data so the Slurm set (v3) is protected. Each user
turn carries the excerpts retriever b returns (the format round 27 was trained and evaluated with). Held-out evaluation
families and questions close to an evaluation question are left out. The assistant turn is a placeholder: round 27's
own answer replaces it in the generation step (on-policy distillation, ctl/r47.sh with RD=r48).

usage: round48_build.py OUT.jsonl   (env N_TOPIC, default 5000; N_SLURM, default 2000)
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
N_TOPIC, N_SLURM = int(os.environ.get('N_TOPIC', '5000')), int(os.environ.get('N_SLURM', '2000'))
rng = random.Random(48)
HELD = ('open-mpi', 'openmpi.org', 'aalto', 'cuda-samples', 'cuda_samples')
TOPIC = re.compile(r'\b(requeue\w*|--requeue|--no-requeue|signal\w*|--signal|SIGTERM|SIGUSR[12]|SIGKILL|SIGINT|trap|'
                   r'preempt\w*|checkpoint\w*|restart\w*|resubmit\w*|grace ?time|KillWait|'
                   r'job arrays?|--array|SLURM_ARRAY_\w+|array tasks?|%A|%a)\b', re.I)
SLURMISH = re.compile(r'\b(slurm|sbatch|srun|scontrol|scancel|squeue|sacct|salloc|#SBATCH)\b', re.I)


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


topic, seen = [], set()
for f in sorted(glob.glob(S + '/runs/ai_factory_all_v*/release/verified.jsonl')) + \
        sorted(glob.glob(C + '/runs/ai_factory_all_v*/release/verified.jsonl')):
    for line in open(f):
        r = json.loads(line)
        q = (r.get('question') or '').split('\n\nProvided input')[0].strip()
        if r.get('mode') != 'prose' or not q or len(q) > 700 or q in seen:
            continue
        if any(h in (r.get('doc_id') or '').lower() or h in (r.get('source_url') or '').lower() for h in HELD):
            continue
        if not (TOPIC.search(q) and SLURMISH.search(q + ' ' + (r.get('source_url') or ''))):
            continue
        seen.add(q)
        topic.append((r['id'], q))
rng.shuffle(topic)
slurm = []
for line in open(S + '/runs/rb31/train/seed42/verified.jsonl'):
    r = json.loads(line)
    if any(h in json.dumps(r).lower()[:2000] for h in HELD):
        continue
    slurm.append(r)
rng.shuffle(slurm)

P, F = C + '/runs/rag65/indexes/documents.sqlite', C + '/runs/rag65/indexes/factory_docs.sqlite'
rows = []
for rid, q in topic:
    if len(rows) >= N_TOPIC:
        break
    if near_eval(q):
        continue
    rows.append({'id': 'r48t-' + rid, 'messages': [{'role': 'user', 'content': retrieve.with_context(q, retrieve.retrieve(q, P, F))},
                                                   {'role': 'assistant', 'content': '.'}], 'kind': 'topic'})
n_t = len(rows)
for r in slurm[:N_SLURM]:
    rows.append({'id': 'r48s-' + r['id'], 'messages': [r['messages'][0], {'role': 'assistant', 'content': '.'}], 'kind': 'slurm'})
rng.shuffle(rows)
with open(out, 'w') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'{len(rows)} rows: {n_t} requeue/signal/preemption/checkpoint/array (from {len(topic)} candidates), {len(rows) - n_t} Slurm')
