"""G2 tool-call validity: first-response tool decisions on dataset_v1/development_tools.jsonl (23 items).

usage: tool_eval.py ENDPOINT OUT.json SERVED_MODEL [SERVED_MODEL ...]
The prompt is rendered with the training chat template, tools and system prompt plus the
tool-first-response-v2 policy, then sent to /v1/completions so the raw <tool_call> text is scored.
Greedy decoding. distillkit.eval_tools_v2.score checks the decision (call / no call), validity,
grounding, mock execution and expected arguments.
"""
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from distillkit.config import load_config
from distillkit.eval_tools_v2 import POLICY, PROTOCOL, score, system_prompt
from distillkit.records import openai_tools
from distillkit.tools import SCHEMAS

endpoint, out, models = sys.argv[1].rstrip('/'), Path(sys.argv[2]), sys.argv[3:]
cfg = load_config('configs/rag65_seed42.json')
cfg.eval.tool_protocol, cfg.eval.tool_application_policy = PROTOCOL, POLICY
tok = AutoTokenizer.from_pretrained(cfg.student.model, local_files_only=True)
items = [json.loads(x) for x in Path('data/eval/dataset_v1/development_tools.jsonl').read_text().splitlines() if x.strip()]
tools = openai_tools(SCHEMAS)
report = {}
for model in models:
    results = []
    for item in items:
        messages = [{'role': 'system', 'content': system_prompt(cfg, True)}, {'role': 'user', 'content': item['question']}]
        prompt = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        body = json.dumps({'model': model, 'prompt': prompt, 'max_tokens': 768, 'temperature': 0}).encode()
        req = urllib.request.Request(endpoint + '/completions', body, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as resp:
            text = json.loads(resp.read())['choices'][0]['text']
        results.append({'id': item['id'], 'expect': item['expect'], 'output': text, **score(text, item)})
    calls = [r for r in results if r['n_calls'] or r['malformed_call']]
    report[model] = {
        'items': len(results),
        'first_response_pass': round(sum(r['first_response_pass'] for r in results) / len(results), 4),
        'decision_ok': round(sum(r['decision_ok'] for r in results) / len(results), 4),
        'valid_call_rate': round(sum(bool(r.get('valid')) for r in calls) / len(calls), 4) if calls else None,
        'decisions': dict(Counter(f"{r['expect']}->{r['decision']}" for r in results)),
        'results': results}
    print(model, {k: v for k, v in report[model].items() if k != 'results'}, flush=True)
out.write_text(json.dumps(report, indent=2))
