"""Multi-turn agent check on curiosity's own Slurm (local mode; every action is declined): after earlier questions,
is the right tool still called? Variants of what the tool-first probe sees: a = full history (dev), b = no history,
c = history with earlier text answers cut to 300 characters.  usage: qa_multiturn.py OUT RUN_JOB DONE_JOB VARIANTS"""
import concurrent.futures as cf, io, json, os, sys
sys.path.insert(0, os.path.expanduser(os.environ.get('PRODUCT', '~/cp/product') + '/src'))
import distillkit.assistant as A
from distillkit.slurm_live import SlurmLive
INDEX, CAND = os.path.expanduser("~/cp/product_index"), os.environ.get("CAND", "http://127.0.0.1:8100/v1")
OUT, RUN, DONE, VARIANTS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SCRIPT = '#!/bin/bash\n#SBATCH -p hackathon\n#SBATCH --gres=gpu:1\n#SBATCH -t 00:05:00\nnvidia-smi'
S = [  # (earlier turns, last question, tool that must be called; None = must call no action tool)
    (['How do I see why my job is pending?'], 'Submit this job for me:\n' + SCRIPT, 'submit_job'),
    (['What is a QOS in Slurm?', 'How do I request GPUs?'], 'Submit this job for me:\n' + SCRIPT, 'submit_job'),
    (['Explain job arrays briefly.'], 'Check whether this script would be accepted, without submitting it:\n' + SCRIPT, 'submit_job'),
    ([f'What happened to my job {DONE}?'], 'Show me the end of its log.', 'read_job_log'),
    ([f'What happened to my job {DONE}?'], 'And its batch script?', 'job_script'),
    ([f'Is job {RUN} still running?'], 'How much memory is it using right now?', 'job_usage'),
    ([f'Is job {RUN} still running?'], 'Give it two more hours.', 'update_job'),
    (['How do I request GPUs in a Slurm job?'], 'How many GPUs are free right now?', 'gpu_availability'),
    (['What does sacct do?'], 'Show my last 5 jobs and whether they succeeded.', 'recent_jobs'),
    (['How do I write a job array?'], 'Show me all my jobs in the queue right now.', 'list_queue'),
    (['Show my jobs in the queue.'], f'Cancel job {RUN}.', 'cancel_job'),
    (['What is fairshare?'], 'What is my fairshare?', None),
    (['How many GPUs are free right now?'], 'What are the limits of the hackathon partition?', 'partition_info'),
    (['How do I checkpoint a PyTorch job?', 'And how do I requeue it?'], f'Show me the end of the log of job {DONE}.', 'read_job_log'),
    (['hello'], 'Show me my jobs in the queue.', 'list_queue'),
    ([f'Show me the batch script of job {DONE}.'], 'Now submit the same script again.', 'submit_job'),
]
orig = A.Assistant._agent_first
def probe_for(v):
    def probe(self, q):
        saved = self.history
        if v == 'b':
            self.history = saved[:1]
        elif v == 'c':
            self.history = [m if m['role'] != 'assistant' or m.get('tool_calls') else {**m, 'content': (m.get('content') or '')[:300]}
                            for m in saved]
        try:
            return orig(self, q)
        finally:
            self.history = saved
    return probe
def run(args):
    v, (before, q, want) = args
    calls = []
    class Rec(SlurmLive):
        def execute(self, name, a):
            calls.append(name)
            return super().execute(name, a)
    bot = A.Assistant(INDEX, model='cand', openai=CAND, slurm=Rec(confirm=lambda n, a: False))
    bot._agent_first = probe_for(v).__get__(bot)
    for b in before:
        bot.ask(b, out=io.StringIO())
    calls.clear()
    ans, _ = bot.ask(q, out=io.StringIO())
    ok = (want in calls) if want else not any(c in A.ACTIONS for c in calls) if hasattr(A, 'ACTIONS') else True
    return {'variant': v, 'before': before, 'question': q, 'want': want, 'calls': calls, 'ok': ok, 'answer': ans[:600]}
jobs = [(v, s) for v in VARIANTS for s in S for _ in range(2)]
with cf.ThreadPoolExecutor(12) as ex:
    rows = list(ex.map(run, jobs))
with open(OUT, 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
for v in VARIANTS:
    rs = [r for r in rows if r['variant'] == v]
    print(v, sum(r['ok'] for r in rs), '/', len(rs), [r['want'] for r in rs if not r['ok']], flush=True)
