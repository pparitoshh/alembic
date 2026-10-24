"""Round 38 data: tool calling. The v3 factory tool workflows (248 call traces with simulated tool results, 26
answer-directly, 362 ask-for-details; 6 Slurm tools) become one training example per assistant turn, so the student
learns the tool-call turns themselves, not only the final answer. Each example keeps the row's tool schemas
(OpenAI format) in `tools`, which train_kd.py renders into the prompt as at inference. 15% of the source documents
are held out for research/tools38/tool_eval.py. Prose replay rows from round 31's data keep the HPC answering.

usage: build_tool_data.py V3_RELEASE.jsonl REPLAY.jsonl OUT_DIR [N_REPLAY] [UPSAMPLE]
"""
import hashlib
import json
import random
import sys
from pathlib import Path

src, replay, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
n_replay = int(sys.argv[4]) if len(sys.argv) > 4 else 12000
upsample = int(sys.argv[5]) if len(sys.argv) > 5 else 3
out.mkdir(parents=True, exist_ok=True)


def held_out(doc_id):
    return int(hashlib.sha256(doc_id.encode()).hexdigest(), 16) % 100 < 15


def call_text(tc):
    f = tc['function']
    args = f['arguments'] if isinstance(f['arguments'], str) else json.dumps(f['arguments'], ensure_ascii=False)
    return '<tool_call>\n{"name": "' + f['name'] + '", "arguments": ' + args + '}\n</tool_call>'


rows = [json.loads(l) for l in open(src)]
rows = [r for r in rows if r.get('mode') in ('call', 'none', 'ask') and r.get('tools')]
train, test, n_turns = [], [], 0
for r in rows:
    tools = [{'type': 'function', 'function': t} for t in r['tools']]
    if held_out(r['doc_id']):
        test.append({'id': r['id'], 'mode': r['mode'], 'tools': tools, 'messages': r['messages']})
        continue
    msgs = r['messages']
    for i, m in enumerate(msgs):
        if m['role'] != 'assistant':
            continue
        target = (m.get('content') or '').strip()
        if m.get('tool_calls'):
            target = ((target + '\n') if target else '') + '\n'.join(call_text(tc) for tc in m['tool_calls'])
        ex = {'id': f"{r['id']}#t{i}", 'question': r['question'], 'tools': tools,
              'messages': msgs[:i] + [{'role': 'assistant', 'content': target}], 'mode': r['mode']}
        train += [ex] * upsample
        n_turns += 1
rng = random.Random(38)
rep = [json.loads(l) for l in open(replay)]
rng.shuffle(rep)
rep = rep[:n_replay]
data = train + rep
rng.shuffle(data)
with open(out / 'data_src.jsonl', 'w') as f:
    for ex in data:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
with open(out / 'tool_test.jsonl', 'w') as f:
    for ex in test:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
print(f'BUILD READY: {len(rows)} tool workflows -> {n_turns} assistant turns x{upsample} for training, '
      f'{len(test)} held-out workflows, {len(rep)} replay rows, {len(data)} training rows')
