"""Teacher ceiling probe: answer the eval sets with a candidate teacher exactly like a student arm.

usage: answer_teacher.py ARM ENDPOINT SERVED_MODEL OUT_DIR THINKING(0|1) VARIANTS(rag,closed)
Same questions, retriever, system prompt and greedy decoding as research/rag65/answer.py, so the judge
compares the teacher with closed-book base on the same footing as the students. With thinking on, the
reasoning block is removed and only the final answer is judged.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, 'research/rag65')
from distillkit.config import load_config
from distillkit.teacher import Teacher
from retrieve import retrieve, with_context

arm, endpoint, served, out, thinking, variants = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5] == '1', sys.argv[6].split(',')
cfg = load_config('configs/rag65_seed42.json')
PILOT, FACTORY = 'runs/rag65/indexes/documents.sqlite', 'runs/rag65/indexes/factory_docs.sqlite'
cfg.teacher.model, cfg.teacher.base_url = served, endpoint
cfg.teacher.temperature, cfg.teacher.top_logprobs = 0, 0
cfg.teacher.max_tokens = 6144 if thinking else cfg.eval.max_new_tokens
cfg.teacher.extra_body = {'chat_template_kwargs': {'enable_thinking': thinking}}
cfg.teacher.concurrency = 32
client = Teacher(cfg.teacher)
out.mkdir(parents=True, exist_ok=True)
THINK = re.compile(r'^.*?</think>\s*', re.S)
for label, path in [('v2', 'data/eval/eval_all_v2.jsonl'), ('v3', 'data/eval/eval_v3.jsonl'), ('d1', 'data/eval/dataset_v1/development_normal.jsonl')]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
    contexts = {r['id']: retrieve(r['question'], PILOT, FACTORY) for r in rows}
    for variant in variants:
        dest = out / f'{label}__{arm}_{variant}.jsonl'
        done = {json.loads(x)['id'] for x in dest.read_text().splitlines()} if dest.exists() else set()
        todo = [r for r in rows if r['id'] not in done]

        def one(r):
            q = with_context(r['question'], contexts[r['id']]) if variant == 'rag' else r['question']
            c = client.complete([{'role': 'system', 'content': cfg.student.system_prompt},
                                 {'role': 'user', 'content': q}], temperature=0)
            answer = THINK.sub('', c.content or '') if thinking else c.content
            return {'id': r['id'], 'answer': answer, 'finish_reason': c.finish_reason,
                    'retrieval': [d['url'] for d in contexts[r['id']]] if variant == 'rag' else []}
        with dest.open('a') as f:
            for result in client.map(one, todo):
                f.write(json.dumps(result) + '\n')
        print(label, arm, variant, 'answers', len(rows), flush=True)
