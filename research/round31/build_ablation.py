"""Rounds 39-41 data (GOAL.md G5 and G14), all from factory batch v11 (Qwen3-235B teacher, same sources):
  ab39f  7,000 verified rows (passed checks + source-support review), random order       = control
  ab40u  7,000 raw generated rows (before any review), random order                       = G5: filtering off
  ab41c  the same 7,000 verified rows as ab39f, sorted by answer length (short -> long)    = G14: curriculum
Raw rows whose question matches an evaluation question (word 3-gram Jaccard >= 0.8) or that lack a final answer are
dropped, so only the review step differs. Trained on base Qwen3-4B so the comparison is not affected by earlier rounds.

usage: build_ablation.py V11_BATCH_DIR EVAL_JSONL... OUT_ROOT
"""
import glob
import json
import os
import random
import re
import sys

batch, *evals, out_root = sys.argv[1:]
N = 7000
rng = random.Random(39)


def grams(q):
    w = re.findall(r'\w+', q.lower())
    return {' '.join(w[i:i + 3]) for i in range(max(1, len(w) - 2))}


eval_g = [grams(json.loads(l)['question']) for p in evals for l in open(p)]


def leaks(q):
    g = grams(q)
    return any(len(g & e) / max(1, len(g | e)) >= 0.8 for e in eval_g)


def ok(r):
    m = r.get('messages') or []
    return len(m) >= 2 and m[-1]['role'] == 'assistant' and (m[-1].get('content') or '').strip()


verified = [json.loads(l) for l in open(f'{batch}/release/verified.jsonl')]
raw = [json.loads(l) for p in sorted(glob.glob(f'{batch}/shard_*/generated.jsonl')) for l in open(p)]
raw = [r for r in raw if ok(r) and not leaks(r['question'])]
rng.shuffle(verified)
rng.shuffle(raw)
sets = {'ab39f': verified[:N], 'ab40u': raw[:N]}
sets['ab41c'] = sorted(sets['ab39f'], key=lambda r: len(r['messages'][-1]['content']))
for name, rows in sets.items():
    os.makedirs(f'{out_root}/{name}', exist_ok=True)
    with open(f'{out_root}/{name}/data_src.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'BUILD READY: verified {len(verified)} rows, raw usable {len(raw)}; wrote ' +
      ', '.join(f'{k} {len(v)}' for k, v in sets.items()))
