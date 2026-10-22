"""Teacher upgrade: gpt-oss-120b answers the student's training questions (same system prompt and user turn).

usage: regen.py WORKER N_WORKERS ENDPOINT SERVED DATA.jsonl OUT_DIR
On our leaderboard gpt-oss-120b (0.85 mean win rate) is clearly stronger than the Qwen3-Next teacher (0.78) the
student already matches, so its answers raise the ceiling. It sees exactly the user turn the student trains on (question
plus quoted sources) and the student's system prompt, plus a teacher-only note asking for a complete answer that fits the
768-token evaluation budget. Reasoning effort medium; only the final answer is kept. A new answer is kept only if it
finished, has at most 550 words and passes the deterministic checks (shell syntax, known Slurm flags).
Output: OUT_DIR/answers_<worker>.jsonl with {id, answer (null when rejected), words, reason}. Resumable.
"""
import json
import sys
from pathlib import Path

from distillkit.checks import check_answer, load_flags
from distillkit.config import load_config
from distillkit.teacher import Teacher

worker, n_workers, endpoint, served, data, out = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4],
                                                   sys.argv[5], Path(sys.argv[6]))
NOTE = ('\n\nGive a thorough, complete answer of about 300 to 500 words, including any script, with the reasoning, the '
        'commands and how to check the result. Use the quoted sources only where they are relevant to the question.')
cfg = load_config('configs/rag65_seed42.json')
SYSTEM = cfg.student.system_prompt
cfg.teacher.model, cfg.teacher.base_url = served, endpoint
cfg.teacher.temperature, cfg.teacher.top_logprobs, cfg.teacher.concurrency, cfg.teacher.max_tokens = 0, 0, 192, 4096
cfg.teacher.extra_body = {'reasoning_effort': 'medium'}
client = Teacher(cfg.teacher)
flags = load_flags()
rows = [json.loads(line) for i, line in enumerate(open(data)) if i % n_workers == worker]
out.mkdir(parents=True, exist_ok=True)
dest = out / f'answers_{worker:03d}.jsonl'
done = set()
if dest.exists():
    for x in dest.read_text().splitlines():
        try:
            done.add(json.loads(x)['id'])
        except (ValueError, KeyError):
            pass
todo = [r for r in rows if r['id'] not in done]
print(f'worker {worker}: {len(rows)} rows, {len(todo)} to do', flush=True)


def one(row):
    user = row['messages'][0]['content']
    try:
        c = client.complete([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user + NOTE}])
    except Exception as e:  # retried on the next job
        print('failed', row['id'], type(e).__name__, flush=True)
        return None
    text = (c.content or '').strip()
    words = len(text.split())
    if c.finish_reason != 'stop':
        reason = 'unfinished'
    elif not 0 < words <= 550:
        reason = 'length'
    elif not check_answer(text, flags)['passed']:
        reason = 'checks'
    else:
        reason = 'ok'
    return {'id': row['id'], 'answer': text if reason == 'ok' else None, 'words': words, 'reason': reason}


for start in range(0, len(todo), 384):
    results = client.map(one, todo[start:start + 384])
    with dest.open('a') as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'worker {worker}: {len(done) + start + len(results)}/{len(rows)}', flush=True)
