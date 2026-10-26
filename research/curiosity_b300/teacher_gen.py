"""Teacher answers with top-20 log-probabilities for distillation (offline vLLM, one process per GPU group).

usage: teacher_gen.py PROMPTS.jsonl OUT_DIR PART NPARTS TEACHER_DIR STUDENT_DIR
Prompt token ids come from the student's chat template (system prompt of the evaluation + the user turn with excerpts),
so training sees exactly the tokens the app sends. Writes OUT_DIR/part{PART}.jsonl (id, text, n_prompt, n_out) and
OUT_DIR/part{PART}.npz (prompt ids, answer ids, top-20 ids and log-probs per answer token, offsets). Truncated answers
(finish_reason=length) are dropped. Resumable per part (skips when the npz exists).
"""
import json
import os
import sys

import numpy as np



def main():
    src, out, part, nparts, teacher, student = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f'{out}/part{part}.npz'):
        sys.exit(0)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, os.path.expanduser('~/cp/alembic/src'))
    from distillkit.config import load_config

    SYSTEM = load_config(os.path.expanduser('~/cp/alembic/configs/rag65_seed42.json')).student.system_prompt
    K = 20
    rows = [json.loads(l) for i, l in enumerate(open(src)) if i % nparts == part]
    tok = AutoTokenizer.from_pretrained(student)
    prompts = []
    for r in rows:
        ids = tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': r['user']}],
                                      add_generation_prompt=True, tokenize=True)
        prompts.append({'prompt_token_ids': list(ids['input_ids'] if hasattr(ids, 'keys') else ids)})
    llm = LLM(teacher, tensor_parallel_size=int(os.environ.get('TP', 4)), max_model_len=16384, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=float(os.environ.get('TEMP', 0.7)), top_p=0.9, max_tokens=1536, logprobs=K)
    res = llm.generate(prompts, sp)
    P, O, TI, TL, off, meta = [], [], [], [], [0], []
    for r, p, g in zip(rows, prompts, res):
        o = g.outputs[0]
        if o.finish_reason != 'stop' or not o.token_ids:
            continue
        ti = np.zeros((len(o.token_ids), K), np.int32)
        tl = np.full((len(o.token_ids), K), -1e4, np.float16)
        for t, lps in enumerate(o.logprobs):
            best = sorted(lps.items(), key=lambda kv: kv[1].rank)[:K]
            for j, (tid, lp) in enumerate(best):
                ti[t, j], tl[t, j] = tid, max(lp.logprob, -1e4)
        P.append(np.asarray(p['prompt_token_ids'], np.int32)); O.append(np.asarray(o.token_ids, np.int32)); TI.append(ti); TL.append(tl)
        meta.append({'id': r['id'], 'domain': r.get('domain'), 'text': o.text, 'n_prompt': len(P[-1]), 'n_out': len(O[-1])})
    po = np.cumsum([0] + [len(x) for x in P]); oo = np.cumsum([0] + [len(x) for x in O])
    np.savez(f'{out}/part{part}.npz.tmp.npz', prompt=np.concatenate(P), out=np.concatenate(O), topk_ids=np.concatenate(TI),
             topk_lp=np.concatenate(TL), prompt_off=po, out_off=oo)
    with open(f'{out}/part{part}.jsonl', 'w') as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    os.replace(f'{out}/part{part}.npz.tmp.npz', f'{out}/part{part}.npz')
    print(f'TEACHER PART {part}: {len(meta)} of {len(rows)} answers kept, {int(oo[-1])} answer tokens')


if __name__ == '__main__':  # vLLM starts its workers with spawn
    main()
