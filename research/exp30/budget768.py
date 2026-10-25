"""Equal answer budget for the leaderboard: every arm whose answers exceed 768 tokens (Qwen3-4B tokenizer, the budget
our student is generated with) gets a copy '<arm>_b768' with each answer cut at 768 tokens. Arms already within the
budget are unchanged and need no copy."""
import json, glob, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
A = 'runs/leaderboard/answers'
arms = sorted({os.path.basename(p).split('__', 1)[1][:-6] for p in glob.glob(f'{A}/v3__*.jsonl')})
for arm in arms:
    if arm.endswith('_b768') or arm == 'base_closed':
        continue
    over, rows_by = 0, {}
    for l in ('v2', 'v3', 'd1'):
        rows = [json.loads(x) for x in open(f'{A}/{l}__{arm}.jsonl')]
        for r in rows:
            ids = tok(r.get('answer') or '', add_special_tokens=False)['input_ids']
            if len(ids) > 768:
                over += 1; r['answer'] = tok.decode(ids[:768]); r['truncated_to'] = 768
        rows_by[l] = rows
    if over:
        for l, rows in rows_by.items():
            with open(f'{A}/{l}__{arm}_b768.jsonl', 'w') as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{arm}: {over} answers over 768 tokens' + (' -> wrote _b768 copy' if over else ''))
