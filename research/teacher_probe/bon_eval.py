"""Best-of-n ceiling probe on the eval sets: how good are best-of-3 teacher answers, as the selector picks them?

usage: bon_eval.py prepare ARM_FILE_TEMPLATE OUT_DIR   ARM_FILE_TEMPLATE contains {label}; writes OUT_DIR/data.jsonl
       bon_eval.py arm OUT_DIR ANSWERS_DIR [CANDS SELECT ARM]       writes {label}__{ARM}.jsonl (default bon_rag)
The 'current' answer of each row is the teacher's greedy answer with retrieval (from the probe); candidates.py
adds 3 sampled answers and select.py picks the best, exactly as in training-data best-of-n. The picked answers
are then judged against closed-book base like any arm, in the same run as the teacher itself.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, 'research/rag65')
from retrieve import retrieve, with_context

SETS = [('v2', 'data/eval/eval_all_v2.jsonl'), ('v3', 'data/eval/eval_v3.jsonl'),
        ('d1', 'data/eval/dataset_v1/development_normal.jsonl')]
PILOT, FACTORY = 'runs/rag65/indexes/documents.sqlite', 'runs/rag65/indexes/factory_docs.sqlite'

if sys.argv[1] == 'prepare':
    prefix, out = sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'data.jsonl', 'w') as f:
        for label, path in SETS:
            arm_file = Path(prefix.replace('{label}', label))  # e.g. runs/teacher_probe2/answers/{label}__t32_rag.jsonl
            greedy = {json.loads(x)['id']: json.loads(x)['answer'] for x in arm_file.read_text().splitlines()}
            for x in Path(path).read_text().splitlines():
                r = json.loads(x)
                user = with_context(r['question'], retrieve(r['question'], PILOT, FACTORY))
                f.write(json.dumps({'id': f'{label}:{r["id"]}', 'mode': 'prose',
                                    'messages': [{'role': 'user', 'content': user},
                                                 {'role': 'assistant', 'content': greedy[r['id']]}]}) + '\n')
    print('prepared', sum(1 for _ in open(out / 'data.jsonl')))
elif sys.argv[1] == 'arm':
    out, answers = Path(sys.argv[2]), Path(sys.argv[3])
    cdir, sdir, arm = sys.argv[4:7] if len(sys.argv) > 6 else ('cands', 'select', 'bon_rag')
    data = {json.loads(x)['id']: json.loads(x) for x in (out / 'data.jsonl').read_text().splitlines()}
    cands = {}
    for f in (out / cdir).glob('cands_*.jsonl'):
        for x in f.read_text().splitlines():
            r = json.loads(x)
            cands[r['id']] = r['candidates']
    picks = {}
    for f in (out / sdir).glob('select_*.jsonl'):
        for x in f.read_text().splitlines():
            r = json.loads(x)
            picks[r['id']] = r['best']
    by_label, chosen_new = {}, 0
    for rid, row in data.items():
        label, qid = rid.split(':', 1)
        best = picks.get(rid, 'current')
        answer = row['messages'][1]['content'] if best == 'current' else cands[rid][int(best[3:])]
        chosen_new += best != 'current'
        by_label.setdefault(label, []).append({'id': qid, 'answer': answer, 'finish_reason': 'stop', 'pick': best})
    for label, rows in by_label.items():
        (answers / f'{label}__{arm}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print('bon arm written:', {k: len(v) for k, v in by_label.items()}, 'new answer picked for', chosen_new, 'of', len(data))
