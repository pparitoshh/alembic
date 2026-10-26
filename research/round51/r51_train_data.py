"""Round 51 step 4: training rows from the Qwen3-235B verdicts (round 27's answer vs the verified reference, judged with
the reference, both orders). Where round 27 is worse than the reference (score < 0.5) the target is the verified
reference; for as many questions where round 27 is as good, the target is round 27's own answer (keeps its style).
The prompt always carries the excerpts (the app's prompt). Replay rows (round 38's non-tool rows, Qwen3-235B scores,
KD_MIXED=1) protect the Slurm sets.

usage: r51_train_data.py RD_DIR REPLAY_SRC OUT.jsonl [N_REPLAY]
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
own = {json.loads(l)['id']: json.loads(l)['answer'] for l in open(f'{rd}/answers/d1__rag.jsonl')}
wrong, right = [], []
for f in glob.glob(f'{rd}/judged_qwen/d1__rag__*.jsonl'):
    for line in open(f):
        r = json.loads(line)
        (wrong if sc(r['judge']['arm_vs_base']) < 0.5 else right).append(r['id'])
rng = random.Random(51)
rng.shuffle(right)
rows = [{'id': i, 'pick': 'reference', 'messages': [{'role': 'user', 'content': q[i]['rag_prompt']},
                                                    {'role': 'assistant', 'content': q[i]['reference']}]} for i in wrong]
rows += [{'id': i, 'pick': 'own', 'messages': [{'role': 'user', 'content': q[i]['rag_prompt']},
                                               {'role': 'assistant', 'content': own[i]}]} for i in right[:len(wrong)]]
rep = [r for r in map(json.loads, open(replay)) if not r.get('tools')]
rng.shuffle(rep)
data = rows + rep[:n_replay]
rng.shuffle(data)
with open(out, 'w') as f:
    for r in data:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'INPUT READY: {len(wrong) + len(right)} judged questions, round 27 worse than the reference on {len(wrong)}; '
      f'{len(rows)} rows ({len(wrong)} reference, {len(rows) - len(wrong)} own), {min(n_replay, len(rep))} replay rows, {len(data)} rows')
