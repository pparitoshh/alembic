"""Calibration text for llama-imatrix: training conversations in the model's chat format (GOAL.md: imatrix from
the run's own transcripts). usage: calibration.py VERIFIED.jsonl SYSTEM_PROMPT_CONFIG OUT.txt N"""
import json
import random
import sys

src, config, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
system = json.load(open(config))['student']['system_prompt']
rows = [json.loads(x) for x in open(src)]
random.Random(42).shuffle(rows)
with open(out, 'w') as f:
    for r in rows[:n]:
        f.write(f'<|im_start|>system\n{system}<|im_end|>\n')
        for m in r['messages']:
            if m['role'] in ('user', 'assistant') and m.get('content'):
                f.write(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n')
print('wrote', n, 'conversations to', out)
