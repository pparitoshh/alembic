"""Curiosity autopilot: decides the next 4B experiment from the results so far, queues it, and marks a winner for
release. Runs on the login node every 2 minutes (autopilot.sh). State: ~/cp/autopilot/state.json, log:
~/cp/logs/autopilot.log. The student is always the 4B (round 27 lineage or Qwen3-4B-Instruct-2507).

A result (runs/<name>/RESULT.json, written by eval.sh: candidate vs round 27 + e on v2/v3/d1, 3-judge majority)
WINS when its mean over the three sets beats the current best by >= 0.02 (round 27 re-answered on this cluster
scored +0.014 against itself: noise) and its v3 is at most 0.015 below round 27's. The best model is the student of the next experiments; a new best is written to ~/cp/release/READY for the
laptop's release watcher (GGUF + Hugging Face push; the token never reaches the cluster).
"""
import json
import os
import time
from pathlib import Path

CP = Path.home() / 'cp'
Q, ST, LOG, REL = CP / 'queue', CP / 'autopilot' / 'state.json', CP / 'logs' / 'autopilot.log', CP / 'release'
R27 = str(CP / 'models' / 'r27')
BASE = str(CP / 'models' / 'qwen3-4b')


def log(msg):
    with LOG.open('a') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ ', time.gmtime()) + msg + '\n')


def load():
    if ST.exists():
        return json.loads(ST.read_text())
    return {'best': {'name': 'r27', 'model': R27, 'mean': None}, 'results': {}, 'queued': ['kd1'], 'seed': 1,
            'tried': [{'name': 'kd1', 'n': 60000, 'student': R27, 'env': {}}]}  # kd1 is queued by hand (020_kd1.sh)


def save(s):
    ST.parent.mkdir(parents=True, exist_ok=True)
    tmp = ST.with_suffix('.tmp')
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(ST)


def busy():
    return any(Q.glob('*.sh')) or any((Q / 'running').glob('*.sh'))


def queue(s, name, n, student, **env):
    e = ' '.join(f'{k}={v}' for k, v in env.items())
    (Q / f'5{len(s["tried"]):02d}_{name}.sh').write_text(f'#!/bin/bash\n{e} exec bash $HOME/cp/bin/exp_kd.sh {name} {n} {student}\n')
    s['tried'].append({'name': name, 'n': n, 'student': student, 'env': env})
    s['queued'].append(name)
    log(f'queued {name}: N={n} student={student} {e}')


def next_experiment(s):
    """Adaptive plan: keep distilling from the best model with fresh teacher data while it improves; otherwise vary
    the recipe (learning rate, pure KD, more data), then a fresh distillation of the base 4B."""
    best = s['best']
    names = [t['name'] for t in s['tried']]
    k = len(names)
    last = s['results'].get(names[-1]) if names else None
    won_last = bool(last and last.get('won'))
    if won_last:  # iterate: new prompts, teacher answers for the new best
        s['seed'] += 1
        return queue(s, f'kd{k + 1}', 60000, best['model'], PROMPT_SEED=s['seed'])
    variants = [
        ('lr1e-5', dict(LR='1e-5', TEACHER_FROM='kd1')),
        ('a1', dict(ALPHA='1.0', TEACHER_FROM='kd1')),
        ('lr2e-6', dict(LR='2e-6', TEACHER_FROM='kd1')),
        ('big', dict(PROMPT_SEED=7, EPOCHS=1)),
        ('base2ep', dict(EPOCHS=2, TEACHER_FROM='kd1')),
    ]
    done = {t['name'] for t in s['tried']}
    for tag, env in variants:
        name = f'kd1_{tag}'
        if name in done:
            continue
        student = BASE if tag == 'base2ep' else best['model']
        n = 150000 if tag == 'big' else 60000
        return queue(s, name, n, student, **env)
    log('plan exhausted; waiting for new plan entries')


def main():
    s = load()
    # collect finished results
    for name in list(s['queued']):
        rf = CP / 'runs' / name / 'RESULT.json'
        failed = any((Q / 'failed').glob(f'*_{name}.sh'))
        if rf.exists():
            r = json.loads(rf.read_text())
            maj = r['majority']
            if s['best']['mean'] is None:
                s['best']['mean'] = r['mean_r27e']  # round 27 + e, judged in this run
            v3 = maj.get('v3', {}).get('diff', -1)
            won = r['mean_cand'] >= s['best']['mean'] + 0.02 and v3 >= -0.015
            s['results'][name] = {**r, 'won': won}
            s['queued'].remove(name)
            log(f'RESULT {name}: mean {r["mean_cand"]} vs best {s["best"]["name"]} {s["best"]["mean"]}; '
                f'majority diffs vs r27e: ' + ', '.join(f'{l} {v["diff"]:+.3f}' for l, v in maj.items()) + f' -> {"WIN" if won else "no"}')
            if won:
                model = str(CP / 'exp' / name / 'model')
                s['best'] = {'name': name, 'model': model, 'mean': r['mean_cand'], 'result': r}
                REL.mkdir(exist_ok=True)
                (REL / 'READY').write_text(json.dumps({'name': name, 'model': model, 'result': r}, indent=1))
                log(f'NEW BEST {name}: release requested')
        elif failed:
            s['queued'].remove(name)
            s['results'][name] = {'failed': True, 'won': False}
            log(f'FAILED {name} (logs/{name}*.log)')
    if not busy() and not (Q / 'STOP').exists():
        next_experiment(s)
    save(s)


if __name__ == '__main__':
    main()
