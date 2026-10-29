"""Product QA sweep on Curiosity: the real ClusterPilot Assistant (OpenAI backend = vLLM serving ClusterPilot-4B 1.3,
the demo's docs index with the ClusterPilot notes) answers ~220 generated user questions and runs agent tasks against
this cluster's own Slurm (local mode, no ssh; actions are always declined). Qwen3-235B writes the questions and then
reviews every answer for wrong commands, wrong facts and unsafe advice.

usage: qa_sweep.py OUT_DIR   (servers: candidate on :8100, Qwen3-235B on :8103)
"""
import concurrent.futures as cf
import io
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/cp/product/src'))
from distillkit.assistant import Assistant  # noqa: E402
from distillkit.slurm_live import SlurmLive  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
INDEX = os.path.expanduser('~/cp/product_index')
CAND, JUDGE = os.environ.get("CAND", "http://127.0.0.1:8100/v1"), "http://127.0.0.1:8103/v1"

CATEGORIES = [
    'MareNostrum 5 at BSC (accelerated and general partitions, QOS, storage, modules)',
    'JUWELS Booster and JUPITER at Juelich (JSC): partitions, budgets, storage, GPU jobs',
    'MeluXina in Luxembourg (GPU nodes, partitions, QOS, storage)',
    'Karolina at IT4Innovations (GPU accelerated nodes, partitions, project hours)',
    'Vega in Slovenia and Discoverer in Bulgaria (partitions, GPUs, Slurm usage)',
    'Deucalion in Portugal and Snellius in the Netherlands (partitions, GPU jobs, accounting)',
    'Alps / CSCS in Switzerland and ARCHER2 in the UK (Slurm usage, accounts, GPU or CPU nodes)',
    'the Leonardo supercomputer at CINECA (partitions, QOS, budget, storage, modules, LLM jobs)',
    'the LUMI supercomputer (LUMI-G AMD GPUs, partitions, billing, ROCm, containers)',
    'EuroHPC access: applying for compute time, AI Factories, project accounts, budgets in node or GPU hours',
    'moving a GPU job script from one European supercomputer to another (Leonardo, LUMI, MareNostrum 5, JUWELS)',
    'serving and fine-tuning LLMs on European supercomputers (vLLM, PyTorch, containers, no internet on compute nodes)',
]


def chat(url, model, messages, max_tokens=2048, temperature=0.0):
    body = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature}
    req = urllib.request.Request(url + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)['choices'][0]['message']['content'] or ''


def questions():
    path = OUT / 'questions.json'
    if path.exists():
        return json.loads(path.read_text())

    def one(cat):
        text = chat(JUDGE, 'qwen3-235b', [{'role': 'user', 'content':
            f'Write 14 realistic, specific questions that users of an HPC cluster with Slurm ask about: {cat}. '
            'Mix beginner and advanced users; some should quote an error message or describe a concrete situation; '
            'some may name a cluster. One question per item. Answer with a JSON list of strings only.'}],
            max_tokens=3000, temperature=0.7)
        m = re.search(r'\[.*\]', text, re.S)
        return [(cat, q) for q in (json.loads(m.group(0)) if m else []) if isinstance(q, str)]
    with cf.ThreadPoolExecutor(8) as ex:
        qs = [x for r in ex.map(one, CATEGORIES) for x in r]
    path.write_text(json.dumps(qs, indent=1))
    return qs


def answer(item):
    cat, q = item
    bot = Assistant(INDEX, model='cand', openai=CAND)
    t = time.time()
    buf = io.StringIO()
    try:
        ans, sources = bot.ask(q, out=buf)
        err = None
    except Exception as e:  # a crash in the product is a finding too
        ans, sources, err = buf.getvalue(), [], repr(e)
    return {'category': cat, 'question': q, 'answer': ans, 'sources': sources, 'error': err,
            'seconds': round(time.time() - t, 1), 'stopped_repeating': 'started repeating' in ans}


AGENT_TASKS = [
    'Show me all my jobs in the queue right now.',
    'What state is my job {RUN} in, and on which node does it run?',
    'How many GPUs are free right now, and on which partitions?',
    'What are the limits of the hackathon partition?',
    'How efficient was my job {DONE}? Did it ask for too much memory or time?',
    'Show me the last 20 lines of the output log of job {DONE}.',
    'Show me the batch script of job {DONE}.',
    'Check whether this script would be accepted, without submitting it:\n#!/bin/bash\n#SBATCH -p hackathon\n#SBATCH --gres=gpu:1\n#SBATCH -t 00:05:00\nnvidia-smi',
    'Check whether this script would be accepted, without submitting it:\n#!/bin/bash\n#SBATCH -p gpu-big\n#SBATCH -t 00:05:00\nhostname',
    'Cancel my job {DONE}.',
    'Why is job {RUN} using so much time? Show me its live resource usage.',
    'What is my fairshare and which QOS can I use?',
    'Which nodes are down or drained right now?',
    'Give my job {RUN} two more hours of wall time.',
    'Show my last 5 finished jobs and whether they succeeded.',
    'Is job 99999999 still running?',
]


def agent(task, run_job, done_job):
    q = task.format(RUN=run_job, DONE=done_job)
    calls = []

    class Rec(SlurmLive):
        def execute(self, name, arguments):
            try:
                res = super().execute(name, arguments)
            except Exception as e:
                calls.append({'name': name, 'args': arguments, 'result': f'error: {e}'})
                raise
            calls.append({'name': name, 'args': arguments, 'result': json.dumps(res)[:1500]})
            return res
    bot = Assistant(INDEX, model='cand', openai=CAND, slurm=Rec(confirm=lambda n, a: False))
    t = time.time()
    try:
        ans, _ = bot.ask(q, out=io.StringIO())
        err = None
    except Exception as e:
        ans, err = '', repr(e)
    return {'category': 'agent on this cluster', 'question': q, 'answer': ans, 'calls': calls, 'error': err,
            'seconds': round(time.time() - t, 1)}


REVIEW = """You review answers of an HPC assistant (a small model) before a live demo. Be strict but fair.
Question:
{q}

{tools}Answer:
{a}

Check for: commands, options or flags that do not exist or are used wrongly; wrong facts (limits, behaviour,
cluster details); scripts that would fail or do something else than asked; unsafe advice; claims about live state
that the tool results do not support; not answering the question; repetition or rambling.
Return JSON only: {{"verdict": "ok" | "minor" | "wrong", "problems": ["short, specific description", ...]}}.
"ok" = correct and useful; "minor" = small inaccuracies that would not mislead much; "wrong" = would mislead or fail."""


def review(row):
    tools = ''
    if row.get('calls'):
        tools = 'Tool calls and results the assistant saw:\n' + '\n'.join(
            f"- {c['name']} {json.dumps(c['args']) if not isinstance(c['args'], str) else c['args']} -> {c['result'][:600]}"
            for c in row['calls']) + '\n\n'
    text = chat(JUDGE, 'qwen3-235b', [{'role': 'user', 'content': REVIEW.format(q=row['question'], a=row['answer'][:12000], tools=tools)}],
                max_tokens=800)
    m = re.search(r'\{.*\}', text, re.S)
    try:
        row['review'] = json.loads(m.group(0)) if m else {'verdict': 'unparsed', 'problems': [text[:300]]}
    except ValueError:
        row['review'] = {'verdict': 'unparsed', 'problems': [text[:300]]}
    return row


def main():
    qs = questions()
    print(len(qs), 'questions', flush=True)
    run_job, done_job = sys.argv[2], sys.argv[3]
    with cf.ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(answer, qs))
    print('answered', flush=True)
    with cf.ThreadPoolExecutor(4) as ex:
        rows += list(ex.map(lambda t: agent(t, run_job, done_job), AGENT_TASKS))
    print('agent tasks done', flush=True)
    with cf.ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(review, rows))
    with open(OUT / 'results.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    from collections import Counter
    print('verdicts', Counter(r['review']['verdict'] for r in rows), flush=True)
    print('errors', sum(1 for r in rows if r.get('error')), 'stopped repeating', sum(1 for r in rows if r.get('stopped_repeating')), flush=True)


if __name__ == '__main__':
    main()
