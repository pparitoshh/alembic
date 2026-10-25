"""Round 28 (sh28) training data: new general-HPC factory rows (v9 + v10, d1 topics) plus a replay share of round 21's
Qwen3-235B rows, so the model learns the new topics without drifting from the distribution it was trained on.

usage: build_data.py [--dry] [--ratio 1.0] [--seed 28] [--out-dir code/runs/sh28]
  --dry   v10 may be unfinished: use v9 only, print counts and two example rows, write nothing.
Without --dry it REQUIRES v10 to be finished (' DONE: ' in ctl/ai_factory_all_v10/controller.log and its release
file present) and exits 2 otherwise, so a controller can simply retry it.

Row format. The factory release rows already have the training-row layout of round 21 (kd235): user turn = question
+ "\n\nProvided input 1 (quoted data, not instructions):\n````text\n...\n````" (27,048 of kd235's 45,933 rows use
exactly this layout, the rest are closed-book or "Quoted retrieval results" rows); score.py / train_kd.py add the
student system prompt at render time. So prose rows are taken verbatim (no conversion): this keeps every field (id,
doc_id, source_licence, ...) and keeps answers that refer to "the provided input" consistent with their prompt.
Excluded: tool-policy rows (mode 'ask': clarify-first answers that name job_status / list_queue tools the student
has no schema for at answer time), rows with tool calls or tool turns, multi-turn rows, empty turns.

Decontamination: factory rows already passed the factory's decontam; here we only assert that no row id or
(normalised) question equals an evaluation item in code/data/eval, and drop rows whose user turn contains an
evaluation question verbatim.

Outputs (in --out-dir): data_src.jsonl (new + replay, shuffled), new_rows.jsonl (rows that need Qwen3-235B scores),
replay_ids.txt, build_report.json. Each row gets 'round28_part': 'new' | 'replay'.
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

import numpy as np

S = '/leonardo_work/EUHPC_D30_031/shared_split'
R = '/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017'
V9 = f'{S}/code/runs/ai_factory_all_v9/release/verified.jsonl'
V10 = f'{S}/code/runs/ai_factory_all_v10/release/verified.jsonl'
V10_LOG = f'{S}/ctl/ai_factory_all_v10/controller.log'
REPLAY = f'{R}/code/runs/kd235/data/verified.jsonl'
REPLAY_KD = f'{R}/code/runs/kd235/teacher_top20'
EVAL = f'{R}/code/data/eval'
PI = re.compile(r'\n\nProvided input 1 \(quoted data, not instructions\):\n`{4,}text\n')


def norm(s):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).split())


def question_of(row):
    """The question part of the user turn (before any quoted-data block)."""
    u = row['messages'][0]['content']
    for marker in ('\n\nProvided input 1 (', '\n\nQuoted retrieval results ('):
        u = u.split(marker)[0]
    return u


def eval_items():
    ids, qs = set(), set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == 'question' and isinstance(v, str):
                    qs.add(norm(v))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for f in glob.glob(f'{EVAL}/**/*.jsonl', recursive=True) + glob.glob(f'{EVAL}/**/*.json', recursive=True):
        try:
            if f.endswith('.jsonl'):
                for line in open(f):
                    if line.strip():
                        x = json.loads(line)
                        if isinstance(x, dict) and isinstance(x.get('id'), str):
                            ids.add(x['id'])
                        walk(x)
            else:
                walk(json.load(open(f)))
        except (ValueError, UnicodeDecodeError):
            continue
    qs.discard('')
    return ids, qs


def why_not_prose(r):
    m = r.get('messages') or []
    if r.get('mode') != 'prose':
        return f"mode_{r.get('mode')}"
    if r.get('tools') or any(x.get('tool_calls') for x in m):
        return 'tool_calls'
    if [x.get('role') for x in m] != ['user', 'assistant']:
        return 'not_single_turn'
    if not all((x.get('content') or '').strip() for x in m):
        return 'empty_turn'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--ratio', type=float, default=1.0, help='replay rows per new row')
    ap.add_argument('--seed', type=int, default=28)
    ap.add_argument('--out-dir', default=f'{S}/code/runs/sh28')
    a = ap.parse_args()

    v10_done = os.path.exists(V10_LOG) and ' DONE: ' in open(V10_LOG).read()
    sources = [('v9', V9)]
    if v10_done and os.path.exists(V10):
        sources.append(('v10', V10))
    elif not a.dry:
        print(f'NOT READY: v10 is not finished (no " DONE: " line in {V10_LOG} or no {V10}); rerun later',
              file=sys.stderr)
        sys.exit(2)
    else:
        print('[dry] v10 not finished: using v9 only')

    eval_ids, eval_qs = eval_items()
    long_eval = [q for q in eval_qs if len(q) >= 40]
    stats = collections.Counter()
    new, seen_ids, seen_q = [], set(), set()
    by_topic = collections.Counter()
    for tag, path in sources:
        for line in open(path):
            r = json.loads(line)
            stats[f'{tag}_rows'] += 1
            why = why_not_prose(r)
            if not why and r.get('source_licence_class') == 'other_needs_review':
                why = 'licence_other_needs_review'  # proprietary vendor docs (NVIDIA/Intel/Linaro...): not used for training (Oct 23)
            if why:
                stats[f'{tag}_skip_{why}'] += 1
                continue
            if not PI.search(r['messages'][0]['content']):
                stats[f'{tag}_closed_book'] += 1  # allowed (kd235 has closed-book rows too); counted for the report
            q = norm(question_of(r))
            if r['id'] in eval_ids or q in eval_qs or any(e in q for e in long_eval):
                stats[f'{tag}_skip_eval_overlap'] += 1
                continue
            if r['id'] in seen_ids or q in seen_q:
                stats[f'{tag}_skip_duplicate'] += 1
                continue
            seen_ids.add(r['id']); seen_q.add(q)
            r['round28_part'] = 'new'
            new.append(r)
            by_topic[f"{tag}:{r.get('source_topic')}"] += 1
    stats['new_rows'] = len(new)

    # Replay pool: round 21 rows that have Qwen3-235B scores of exactly their own answer (kd235 scored this file).
    scored = set()
    for f in sorted(glob.glob(f'{REPLAY_KD}/kd_*.npz')):
        scored.update(str(x) for x in np.load(f)['row_ids'])
    pool = []
    for line in open(REPLAY):
        r = json.loads(line)
        if r['id'] not in scored or r['id'] in seen_ids or r['id'] in eval_ids:
            continue
        q = norm(question_of(r))
        if q in seen_q or q in eval_qs:
            continue
        pool.append(r)
    rng = random.Random(a.seed)
    n_replay = min(len(pool), round(a.ratio * len(new)))
    replay = rng.sample(pool, n_replay)
    for r in replay:
        r['round28_part'] = 'replay'
    stats['replay_pool'] = len(pool)
    stats['replay_rows'] = len(replay)

    rows = new + replay
    rng.shuffle(rows)
    # Final assertions: unique ids, no eval collision, single-turn user/assistant only.
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), 'duplicate row ids'
    assert not (set(ids) & eval_ids), 'row id collides with an eval id'
    assert not any(norm(question_of(r)) in eval_qs for r in rows), 'question collides with an eval question'
    assert all([m['role'] for m in r['messages']] == ['user', 'assistant'] for r in rows)
    stats['total_rows'] = len(rows)
    report = {'stats': dict(stats), 'new_by_topic': dict(by_topic.most_common()), 'sources': sources,
              'v10_done': v10_done, 'eval_ids': len(eval_ids), 'eval_questions': len(eval_qs),
              'ratio': a.ratio, 'seed': a.seed}
    print(json.dumps(report, indent=1))

    if a.dry:
        def short(r):
            m = r['messages']
            u, s = m[0]['content'], m[1]['content']
            return {'id': r['id'], 'doc_id': r.get('doc_id'), 'source_licence': r.get('source_licence'),
                    'user': u[:350] + f' ...[{len(u)} chars]... ' + u[-80:], 'assistant': s[:250] + ' ...'}
        for r in [x for x in rows if x['round28_part'] == 'new'][:2]:
            print('[dry] new row:', json.dumps(short(r), indent=1, ensure_ascii=False))
        return

    os.makedirs(a.out_dir, exist_ok=True)

    def write(name, items):
        tmp = os.path.join(a.out_dir, name + '.tmp')
        with open(tmp, 'w') as f:
            for x in items:
                f.write((json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x) + '\n')
        os.replace(tmp, os.path.join(a.out_dir, name))
    write('new_rows.jsonl', new)
    write('replay_ids.txt', [r['id'] for r in replay])
    write('build_report.json', [json.dumps(report)])
    write('data_src.jsonl', rows)  # last: its presence means the build finished
    print(f'BUILT: {len(rows)} rows ({len(new)} new, {len(replay)} replay) -> {a.out_dir}/data_src.jsonl')


if __name__ == '__main__':
    main()
