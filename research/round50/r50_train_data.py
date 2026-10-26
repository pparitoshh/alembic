"""Round 50 step 4: training rows from the Qwen3-235B verdicts. For every question the target is round 27's BETTER answer
(with excerpts vs closed-book, judged with the verified reference, both orders; a tie keeps the with-excerpts answer),
and the prompt always carries the excerpts (the app's prompt), so the student learns to fall back on its own knowledge
when the excerpts do not help. Replay rows (round 38's non-tool rows, Qwen3-235B scores, KD_MIXED=1) protect the Slurm
sets.

usage: r50_train_data.py RD_DIR REPLAY_SRC OUT.jsonl [N_REPLAY]
"""
import glob
import json
import random
import sys

rd, replay, out = sys.argv[1:4]
n_replay = int(sys.argv[4]) if len(sys.argv) > 4 else 6000


def sc(v):
    return ((v[0] == 'A') + (v[0] == 'T') * .5 + (v[1] == 'B') + (v[1] == 'T') * .5) / 2


q = {json.loads(l)['id']: json.loads(l) for l in open(f'{rd}/questions.jsonl')}
rag = {json.loads(l)['id']: json.loads(l)['answer'] for l in open(f'{rd}/answers/d1__rag.jsonl')}
closed = {json.loads(l)['id']: json.loads(l)['answer'] for l in open(f'{rd}/answers/d1__base_closed.jsonl')}
rows, wins = [], {'rag': 0, 'closed': 0}
for f in glob.glob(f'{rd}/judged_qwen/d1__rag__*.jsonl'):
    for line in open(f):
        r = json.loads(line)
        s = sc(r['judge']['arm_vs_base'])  # rag (arm) vs closed-book (base)
        pick = 'closed' if s < 0.5 else 'rag'
        wins[pick] += 1
        target = closed[r['id']] if pick == 'closed' else rag[r['id']]
        rows.append({'id': r['id'], 'pick': pick, 'messages': [{'role': 'user', 'content': q[r['id']]['rag_prompt']},
                                                               {'role': 'assistant', 'content': target}]})
rng = random.Random(50)
rep = [r for r in map(json.loads, open(replay)) if not r.get('tools')]
rng.shuffle(rep)
data = rows + rep[:n_replay]
rng.shuffle(data)
with open(out, 'w') as f:
    for r in data:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'INPUT READY: {len(rows)} judged questions (better answer: {wins}), {min(n_replay, len(rep))} replay rows, {len(data)} rows')
