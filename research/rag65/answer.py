"""Answer v2 (66) + v3 (200) for one worker's arms: closed-book and with the shared retriever.

usage: answer.py WORKER ENDPOINT OUT_DIR    slices: v2 (66), v3 (200), d1 = dataset_v1 development (106)    worker 0-2 = seeds 42/43/44 (full, x05, x02); 3 = base
Greedy, same system prompt and token budget for every arm. Resumable per arm.
"""
import json
import os
import sys
from pathlib import Path

from distillkit.config import load_config
from distillkit.teacher import Teacher
from retrieve import retrieve, with_context

worker, endpoint, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
cfg = load_config('configs/rag65_seed42.json')
PILOT = 'runs/rag65/indexes/documents.sqlite'
FACTORY = 'runs/rag65/indexes/factory_docs.sqlite'
SEEDS = [42, 43, 44]
models = ({'base': cfg.student.model} if worker == 3 else
          {f's{SEEDS[worker]}{v}': f's{SEEDS[worker]}{v}' for v in os.environ.get('VARIANTS', ',_x05,_x02').split(',')})
out.mkdir(parents=True, exist_ok=True)
for label, path in [('v2', 'data/eval/eval_all_v2.jsonl'), ('v3', 'data/eval/eval_v3.jsonl'), ('d1', 'data/eval/dataset_v1/development_normal.jsonl')]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
    contexts = {r['id']: retrieve(r['question'], PILOT, FACTORY) for r in rows}
    for name, served in models.items():
        cfg.teacher.model, cfg.teacher.base_url = served, endpoint
        cfg.teacher.temperature, cfg.teacher.max_tokens, cfg.teacher.top_logprobs = 0, cfg.eval.max_new_tokens, 0
        cfg.teacher.extra_body = {'chat_template_kwargs': {'enable_thinking': False}}
        client = Teacher(cfg.teacher)
        for variant in ['closed', 'rag']:
            dest = out / f'{label}__{name}_{variant}.jsonl'
            done = set()
            if dest.exists():  # a job killed at its time limit can leave a partial last line
                good = []
                for x in dest.read_text().splitlines():
                    try:
                        done.add(json.loads(x)['id'])
                        good.append(x)
                    except (ValueError, KeyError):
                        pass
                dest.write_text(''.join(g + '\n' for g in good))
            todo = [r for r in rows if r['id'] not in done]

            def one(r):
                q = with_context(r['question'], contexts[r['id']]) if variant == 'rag' else r['question']
                c = client.complete([{'role': 'system', 'content': cfg.student.system_prompt},
                                     {'role': 'user', 'content': q}], temperature=0)
                return {'id': r['id'], 'answer': c.content, 'finish_reason': c.finish_reason,
                        'retrieval': [d['url'] for d in contexts[r['id']]] if variant == 'rag' else []}
            with dest.open('a') as f:
                for result in client.map(one, todo):
                    f.write(json.dumps(result) + '\n')
                    f.flush()
            print(label, name, variant, 'answers', len(rows), flush=True)
