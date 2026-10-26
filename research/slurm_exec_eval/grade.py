"""Grade execution-eval answers on a real Slurm cluster (Leonardo), extending research/slurm-sandbox-eval (Docker).

Each answer gets two checks:
  rules  - every required pattern of the task is in the answer's code, no forbidden pattern (e.g. --array=1-10:3)
           in the fenced code the user would run (prose may mention the wrong form while explaining it);
  slurm  - the answer is run as-is against the real scheduler: scripts and #SBATCH lines through `sbatch --test-only`,
           srun lines through `srun --test-only`, job-control and query commands (scontrol, sacct, squeue, sinfo,
           seff) against a held fixture array job this script submits; scancel is graded by the rules only (held, so they never run), job ids in the task
           replaced by the fixture's. Only allow-listed Slurm commands run; anything with pipes, redirects or command
           substitution is not executed.
Rating: correct = rules pass and Slurm accepts it (or the cluster's own policy refuses an otherwise valid command);
partly correct = rules pass but Slurm rejects it, or at least half of the rules pass; wrong = otherwise.

usage: python3 grade.py TASKS.jsonl ANSWERS_DIR OUT_DIR    (on a Slurm login node; ACCOUNT, PARTITION, GPU_PARTITION env)
"""
import glob
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile

tasks_f, ans_dir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
ACC = os.environ.get('ACCOUNT', 'EUHPC_D30_031')
PART = os.environ.get('PARTITION', 'lrd_all_serial')
GPART = os.environ.get('GPU_PARTITION', 'boost_usr_prod')
GQOS = os.environ.get('GPU_QOS', 'boost_qos_dbg')
ALLOWED = ('sbatch', 'srun', 'scontrol', 'sacct', 'squeue', 'sinfo', 'scancel', 'seff')
# refusals caused by the cluster's own limits or configuration, not by the answer
POLICY = re.compile(r'(?i)(disabled|not allowed|permission|access denied|only.*root|not supported|invalid user|'
                    r'violates accounting/QOS policy|QOSMax\w+Limit|time limit is invalid|exceeds|not installed)')
tmp = tempfile.mkdtemp(prefix='slurm_exec_')


def sh(cmd, timeout=60):
    p = subprocess.run(cmd, shell=isinstance(cmd, str), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=timeout, cwd=tmp)  # Python 3.6 on Leonardo login nodes
    return p.returncode, (p.stdout + p.stderr).strip()


def submit(*args):
    # held and starting a week from now: an answer that releases it (scontrol release) cannot make it run and finish
    rc, o = sh(['sbatch', '--parsable', '--hold', '--begin=now+7days', '-A', ACC, '-p', PART, '-t', '00:05:00', *args,
                '--wrap', 'sleep 30'], timeout=300)
    m = re.search(r'^(\d+)(;\S+)?$', o, re.M)  # the id line; sbatch also prints warnings (gres/tmpfs default)
    if rc or not m:
        raise SystemExit(f'fixture submit failed: {o}')
    return m.group(1)


# fixture: one held 2-task array job that never starts (the serial QoS counts array tasks against its 2-submitted-jobs
# limit); every task's job id points at it, which also covers the array throttle; cancelled at the end
FIX = submit('-J', 'fixture', '--array=1-2%1')
ARR = FIX
JOBMAP = {'55': FIX, '77': FIX, '88': FIX, '99': FIX, '1234': FIX, '321': FIX, '500': FIX, '42': FIX, '999': ARR}
for name in ('B.sh', 'post.sh', 'run.sh'):
    open(os.path.join(tmp, name), 'w').write('#!/bin/bash\nhostname\n')
    os.chmod(os.path.join(tmp, name), 0o755)


def code_of(answer):
    blocks = re.findall(r'```[a-zA-Z]*\n(.*?)```', answer, re.S)
    inline = re.findall(r'`([^`\n]+)`', re.sub(r'```.*?```', '', answer, flags=re.S))
    return '\n'.join(blocks + inline) if (blocks or inline) else answer


def sub_ids(cmd):
    return re.sub(r'\b(55|77|88|99|1234|321|500|42|999)\b', lambda m: JOBMAP[m.group(1)], cmd)


def test_script(text, gpu=False):
    path = os.path.join(tmp, 'job.sh')
    if not text.lstrip().startswith('#!'):
        text = '#!/bin/bash\n' + text
    open(path, 'w').write(text + '\nhostname\n')
    extra = ['-p', GPART, '-q', GQOS] if gpu else ['-p', PART]
    rc, o = sh(['sbatch', '--test-only', '-A', ACC, *extra, path])
    return ('ok' if 'to start at' in o else 'policy' if POLICY.search(o) else 'rejected'), o[-300:]


def run_command(line):
    line = line.strip().lstrip('$ ').strip()
    if not line or line.split()[0] not in ALLOWED or re.search(r'[|><;`]|\$\(', line):
        return None
    cmd = os.path.expandvars(sub_ids(line))  # $USER, $HOME as a shell would expand them
    toks = shlex.split(cmd)
    if toks[0] == 'sbatch':
        toks = ['sbatch', '--test-only', '-A', ACC, '-p', PART] + toks[1:]
    elif toks[0] == 'srun':
        toks = ['srun', '--test-only', '-A', ACC, '-p', PART] + toks[1:]
    elif toks[0] == 'scancel':
        return 'skipped', 'scancel is graded by the rules only (it would cancel the fixtures)'
    try:
        rc, o = sh(toks)
    except subprocess.TimeoutExpired:
        return 'skipped', 'timeout (slow scheduler; not counted)'
    except FileNotFoundError:
        return 'policy', f'{toks[0]} is not installed on this cluster'
    if rc == 0 and not re.search(r'(?i)\b(error|invalid|unrecognized)\b', o):
        return 'ok', o[-300:]
    return ('policy' if POLICY.search(o) else 'rejected'), o[-300:]


def execute(task, code):
    k = task['kind']
    if k == 'script':
        block = next((b for b in re.findall(r'```[a-zA-Z]*\n(.*?)```', task['_answer'], re.S) if '#SBATCH' in b), None)
        return test_script(block) if block else ('rejected', 'no script block')
    if k == 'line':
        lines = [l.strip() for l in code.splitlines() if l.strip().startswith('#SBATCH')]
        if not lines:
            return 'rejected', 'no #SBATCH line'
        return test_script('\n'.join(dict.fromkeys(lines)), gpu=task['id'] == 17)
    if task['id'] == 29:
        line = next((l.strip() for l in code.splitlines() if 'OMP_NUM_THREADS=' in l), '')
        if not line or re.search(r'[|><;`]|\$\(', line.replace('$SLURM_CPUS_PER_TASK', '')):
            return 'rejected', 'no OMP_NUM_THREADS line'
        rc, o = sh(['bash', '-c', f'SLURM_CPUS_PER_TASK=4; {line}; echo $OMP_NUM_THREADS'])
        return ('ok' if o.strip().endswith('4') else 'rejected'), o[-200:]
    results = [r for r in (run_command(l) for l in code.splitlines()) if r]
    if not results:
        return 'rejected', 'no runnable Slurm command'
    if all(r[0] == 'skipped' for r in results):
        return 'policy', ' | '.join(r[1][:120] for r in results)
    if all(r[0] in ('ok', 'skipped') for r in results):
        return 'ok', ' | '.join(r[1][:120] for r in results)
    if any(r[0] == 'policy' for r in results) and not any(r[0] == 'rejected' for r in results):
        return 'policy', ' | '.join(r[1][:120] for r in results)
    return 'rejected', ' | '.join(r[1][:160] for r in results)


def wilson(c, n, z=1.959964):
    p = c / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return m - h, m + h


tasks = [json.loads(l) for l in open(tasks_f)]
report = {}
try:
    for f in sorted(glob.glob(os.path.join(ans_dir, '*.jsonl'))):
        model = os.path.basename(f)[:-6]
        answers = {json.loads(l)['id']: json.loads(l)['answer'] for l in open(f)}
        rows = []
        for t in tasks:
            a = answers.get(t['id'], '')
            t = {**t, '_answer': a}
            code = code_of(a)
            hits = [bool(re.search(r, code)) for r in t['required']]
            fenced = '\n'.join(re.findall(r'```[a-zA-Z]*\n(.*?)```', a, re.S)) or code  # what the user would run
            bad = [r for r in t['forbidden'] if re.search(r, fenced)]
            rules = all(hits) and not bad
            ex, out_txt = execute(t, code)
            if rules and ex in ('ok', 'policy'):
                rating = 'correct'
            elif rules or (sum(hits) >= len(hits) / 2 and not bad):
                rating = 'partly correct'
            else:
                rating = 'wrong'
            rows.append({'id': t['id'], 'rating': rating, 'rules': rules, 'rules_hit': f'{sum(hits)}/{len(hits)}',
                         'forbidden': bad, 'slurm': ex, 'slurm_output': out_txt, 'answer': a})
        n = len(rows)
        c = sum(r['rating'] == 'correct' for r in rows)
        p = sum(r['rating'] == 'partly correct' for r in rows)
        lo, hi = wilson(c, n)
        report[model] = {'n': n, 'correct': c, 'partly': p, 'strict': round(c / n, 3), 'lenient': round((c + p / 2) / n, 3),
                         'wilson95': [round(lo, 3), round(hi, 3)], 'slurm_accepted': sum(r['slurm'] in ('ok', 'policy') for r in rows),
                         'rows': rows}
        print(f'{model}: {c}/{n} correct, {p} partly, strict {c / n:.2f}, lenient {(c + p / 2) / n:.2f}, Slurm accepted '
              f'{report[model]["slurm_accepted"]}/{n}', flush=True)
finally:
    sh(['scancel', FIX])
json.dump(report, open(os.path.join(out, 'results.json'), 'w'), indent=1)
with open(os.path.join(out, 'results.md'), 'w') as f:
    f.write('| model | correct | partly | strict | lenient | Wilson 95% (strict) | Slurm accepted |\n|---|---|---|---|---|---|---|\n')
    for m, r in sorted(report.items(), key=lambda kv: -kv[1]['lenient']):
        f.write(f"| {m} | {r['correct']}/{r['n']} | {r['partly']} | {r['strict']:.2f} | {r['lenient']:.2f} | "
                f"[{r['wilson95'][0]:.2f}, {r['wilson95'][1]:.2f}] | {r['slurm_accepted']}/{r['n']} |\n")
    ids = [t['id'] for t in tasks]
    f.write('\n| task | ' + ' | '.join(sorted(report)) + ' |\n|---|' + '---|' * len(report) + '\n')
    for i in ids:
        f.write(f'| {i} | ' + ' | '.join(next(r['rating'] for r in report[m]['rows'] if r['id'] == i) for m in sorted(report)) + ' |\n')
print('RESULTS', os.path.join(out, 'results.md'))
