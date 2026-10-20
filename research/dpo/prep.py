"""DPO pairs in TRL's conversational format: the student's system prompt + user turn, chosen vs rejected answer.

usage: prep.py PAIRS.jsonl OUT.jsonl N [SEED]
Pairs come from best-of-n: chosen = the selector's best answer (kept only if supported and check-clean),
rejected = its worst. A fixed random sample of N pairs keeps training within a few hours.
"""
import json
import random
import sys

from distillkit.config import load_config

pairs, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
system = load_config('configs/rag65_seed42.json').student.system_prompt
rows = [json.loads(x) for x in open(pairs)]
rows = [r for r in rows if r['chosen'].strip() and r['rejected'].strip() and r['chosen'] != r['rejected']]
random.Random(seed).shuffle(rows)
with open(out, 'w') as f:
    for r in rows[:n]:
        f.write(json.dumps({'prompt': [{'role': 'system', 'content': system}, *r['prompt']],
                            'chosen': [{'role': 'assistant', 'content': r['chosen']}],
                            'rejected': [{'role': 'assistant', 'content': r['rejected']}]}, ensure_ascii=False) + '\n')
print('wrote', min(n, len(rows)), 'of', len(rows), 'pairs')
