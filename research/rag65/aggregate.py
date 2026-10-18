"""Win rate vs closed-book base per arm and slice, mean over seeds, question bootstrap, absolute checks."""
from collections import defaultdict
import json
import random
import re
import sys
from pathlib import Path

from distillkit.checks import check_answer, load_flags

judged, answers, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
S = {'A': 1., 'T': .5, 'B': 0.}
R = {'B': 1., 'T': .5, 'A': 0.}
scores, invalid = defaultdict(dict), defaultdict(int)
for p in judged.glob('*.jsonl'):
    label, arm, _ = p.stem.split('__')
    for r in map(json.loads, p.read_text().splitlines()):
        v = r['judge']['arm_vs_base']
        if None in v:
            invalid[(label, arm)] += 1
        else:
            scores[(label, arm)][r['id']] = (S[v[0]] + R[v[1]]) / 2
expected = {'v2': 66, 'v3': 200, 'd1': 106}
flags = load_flags(None)
report = {'judge': 'openai/gpt-oss-20b, both orders, vs closed-book Qwen3-4B-Instruct-2507',
          'document_assisted_arms_are_labelled_rag': True, 'arms': {}, 'seed_means': {}}
for (label, arm), per in sorted(scores.items()):
    complete = len(per) == expected[label] and not invalid[(label, arm)]
    vals = list(per.values())
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(5000))
    ans = [json.loads(x) for x in (answers / f'{label}__{arm}.jsonl').read_text().splitlines()]
    checks = [check_answer(a['answer'], flags) for a in ans]
    report['arms'][f'{label}/{arm}'] = {
        'win_rate': round(sum(vals) / len(vals), 4), 'complete': complete, 'invalid': invalid[(label, arm)],
        'question_bootstrap_95': [round(boot[125], 4), round(boot[4874], 4)],
        'check_pass_rate': round(sum(c['passed'] for c in checks) / len(checks), 4),
        'answers_with_invalid_flags': sum(bool(c['bad_flags']) for c in checks),
        'truncated': sum(a['finish_reason'] == 'length' for a in ans)}
groups = defaultdict(list)
for key, v in report['arms'].items():
    label, arm = key.split('/')
    m = re.match(r's(\d+)(.*)', arm)
    if m and v['complete']:
        groups[f'{label}/student{m.group(2)}'].append(v['win_rate'])
for key, vals in sorted(groups.items()):
    report['seed_means'][key] = {'mean': round(sum(vals) / len(vals), 4), 'seeds': len(vals),
                                 'spread': round(max(vals) - min(vals), 4)}
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report['seed_means'], indent=2))
print(json.dumps({k: v['win_rate'] for k, v in report['arms'].items() if 'base' in k}, indent=2))
