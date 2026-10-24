"""Logit distillation, step 1: the teacher's top-20 next-token distribution at every answer token.

usage: score.py WORKER N_WORKERS ENDPOINT SERVED_MODEL DATA.jsonl OUT_DIR
Each row is rendered exactly as the student trains on it (student system prompt, chat template, thinking off).
Qwen3-Next and Qwen3-4B share the vocabulary and the chat template (checked on 200 rows, Oct 19), so the teacher
scores the student's own token sequence: one prefill per row (max_tokens=1, prompt_logprobs=20), no generation.
Output: OUT_DIR/kd_<worker>_<chunk>.npz with, per row, the answer tokens' top-20 ids (int32) and log-probs
(float16), concatenated, plus offsets and row ids. Resumable per chunk.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer

from distillkit.config import load_config

worker, n_workers, endpoint, served, data, out = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4],
                                                   sys.argv[5], Path(sys.argv[6]))
K, CHUNK = 20, 256
MAX_LEN = int(os.environ.get("SCORE_MAX_LEN") or 4096)  # rows longer than this are skipped; round 32 (8 excerpts per prompt) uses 16384
system = load_config('configs/rag65_seed42.json').student.system_prompt
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
out.mkdir(parents=True, exist_ok=True)


def render(row):
    """Token ids of the conversation and where the answer starts. Prompt and answer are tokenized separately, as at
    generation time (the prompt ends with 'assistant\n'); one joint string can merge that newline into the answer's
    first token (e.g. '\n```')."""
    msgs = [{'role': 'system', 'content': system}, *row['messages']]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt = tok(prompt, add_special_tokens=False)['input_ids']
    answer = tok(msgs[-1]['content'] + '<|im_end|>', add_special_tokens=False)['input_ids']
    return prompt + answer, len(prompt)


def score(row):
    full, start = render(row)
    if len(full) > MAX_LEN:
        return row['id'], None
    r = requests.post(f'{endpoint}/completions', timeout=600, json={
        'model': served, 'prompt': full, 'max_tokens': 1, 'temperature': 0, 'prompt_logprobs': K})
    r.raise_for_status()
    plp = r.json()['choices'][0]['prompt_logprobs']  # one dict per prompt position (None at position 0)
    ids = np.zeros((len(full) - start, K), np.int32)
    lps = np.full((len(full) - start, K), -1e4, np.float16)
    for j, pos in enumerate(range(start, len(full))):
        top = sorted(((int(t), v['logprob'], v['rank']) for t, v in plp[pos].items()), key=lambda x: x[2])[:K]
        ids[j, :len(top)] = [t for t, _, _ in top]
        lps[j, :len(top)] = [lp for _, lp, _ in top]
    return row['id'], (ids, lps)


rows = [json.loads(line) for i, line in enumerate(open(data)) if i % n_workers == worker]
chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
print(f'worker {worker}: {len(rows)} rows in {len(chunks)} chunks', flush=True)
with ThreadPoolExecutor(64) as pool:
    for c, chunk in enumerate(chunks):
        dest = out / f'kd_{worker:03d}_{c:04d}.npz'
        if dest.exists():
            continue
        got = [(rid, res) for rid, res in pool.map(score, chunk) if res is not None]
        ids = np.concatenate([res[0] for _, res in got])
        lps = np.concatenate([res[1] for _, res in got])
        offsets = np.cumsum([0] + [len(res[0]) for _, res in got])
        tmp = dest.with_suffix('.tmp.npz')
        np.savez(tmp, ids=ids, lps=lps, offsets=offsets, row_ids=np.array([rid for rid, _ in got]))
        tmp.rename(dest)  # complete chunks only
        print(f'worker {worker}: chunk {c + 1}/{len(chunks)} ({len(got)} rows, {len(ids)} answer tokens)', flush=True)
