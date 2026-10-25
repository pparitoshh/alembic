"""Regenerate the B7 prose answers with the teacher, given exactly the user turn the student trains on.

usage: regen.py WORKER N_WORKERS ENDPOINT RAFT.jsonl OUT_DIR
The old B7 answers are short; the judge prefers complete answers. The teacher sees the same system prompt and
the same (RAFT) user turn, plus a teacher-only length note so answers fit the 768-token evaluation budget.
Resumable in chunks; one output file per worker.
"""
import json
import os
import sys
from pathlib import Path

from distillkit.config import load_config
from distillkit.teacher import Teacher

worker, n_workers, endpoint, raft, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
# v1 asked for at most ~450 words and the student came out half as long as base, which the judge penalised.
LENGTH_NOTES = {
    'v1': ('\n\nAnswer completely but concisely: at most about 450 words, including any script. '
           'Use the quoted sources only where they are relevant to the question.'),
    'v2': ('\n\nGive a thorough, complete answer of about 300 to 550 words, including any script, with the '
           'reasoning, the commands and how to check the result. Use the quoted sources only where they are '
           'relevant to the question.'),
}
LENGTH_NOTE = LENGTH_NOTES[os.environ.get('REGEN_PROMPT', 'v1')]
cfg = load_config('configs/rag65_seed42.json')
cfg.teacher.model, cfg.teacher.base_url = 'Qwen/Qwen3-32B-AWQ', endpoint
cfg.teacher.temperature, cfg.teacher.top_logprobs, cfg.teacher.concurrency = 0.3, 0, 64
# v2 must fit the 768-token evaluation budget; longer answers stop at the limit and are dropped later.
cfg.teacher.max_tokens = 1100 if os.environ.get('REGEN_PROMPT', 'v1') == 'v1' else 760
cfg.teacher.extra_body = {'chat_template_kwargs': {'enable_thinking': False}}
client = Teacher(cfg.teacher)

rows = []
for i, line in enumerate(open(raft)):
    if i % n_workers != worker:
        continue
    row = json.loads(line)
    users = [m for m in row['messages'] if m['role'] == 'user']
    if row['raft_origin'] == 'b7' and row.get('mode', 'prose') == 'prose' and len(users) == 1:
        rows.append({'id': row['id'], 'user': users[0]['content']})
out.mkdir(parents=True, exist_ok=True)
dest = out / f'regen_{worker:02d}.jsonl'
done = {json.loads(x)['id'] for x in dest.read_text().splitlines()} if dest.exists() else set()
todo = [r for r in rows if r['id'] not in done]
print(f'worker {worker}: {len(rows)} prose rows, {len(todo)} to do', flush=True)


def one(r):
    try:
        return generate(r)
    except Exception as e:  # a failed request is retried on the next round
        print('failed', r['id'], type(e).__name__, flush=True)


def generate(r):
    c = client.complete([{'role': 'system', 'content': cfg.student.system_prompt + LENGTH_NOTE},
                         {'role': 'user', 'content': r['user']}], temperature=0.3)
    return {'id': r['id'], 'answer': c.content, 'finish_reason': c.finish_reason}


for start in range(0, len(todo), 256):
    results = client.map(one, todo[start:start + 256])
    with dest.open('a') as f:
        for result in results:
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f'worker {worker}: {len(done) + start + len(results)}/{len(rows)}', flush=True)
