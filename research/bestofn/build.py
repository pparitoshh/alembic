"""Best-of-n, step 3: the round-7 training set and DPO pairs.

usage: build.py DATA.jsonl CANDS_DIR SELECT_DIR OUT_DIR
A row's answer is replaced only when the reviewer picked a new candidate, judged it fully supported by the
quoted sources with no unsupported claims, and it passes the same deterministic checks as verify (shell
syntax, known Slurm flags). Otherwise the verified answer stays. DPO pairs (chosen = best, rejected = worst)
are written only when best and worst differ and the chosen answer is accepted.
"""
from collections import Counter
import copy
import hashlib
import json
import sys
from pathlib import Path

from distillkit.checks import check_answer, load_flags

data, cands_dir, select_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
if (out / 'verified.jsonl').exists():
    raise SystemExit('training set exists; use a new directory')


def load(d, pattern):
    rows = {}
    for f in sorted(d.glob(pattern)):
        for x in f.read_text().splitlines():
            try:
                r = json.loads(x)
                rows[r['id']] = r
            except (ValueError, KeyError):
                pass
    return rows


cands, picks, flags = load(cands_dir, 'cands_*.jsonl'), load(select_dir, 'select_*.jsonl'), load_flags()
stats, dpo = Counter(), []
with open(out / 'verified.jsonl', 'w') as f:
    for line in open(data):
        row = json.loads(line)
        pick = picks.get(row['id'])
        if pick is None:
            stats['not_selected'] += 1
        else:
            options = {'current': row['messages'][1]['content'],
                       **{f'new{k}': a for k, a in enumerate(cands[row['id']]['candidates'])}}
            best, worst = options.get(pick['best']), options.get(pick['worst'])
            if pick['best'] == 'current':
                stats['kept_current_best'] += 1
            elif not pick['best_fully_supported'] or pick['unsupported_claims_in_best']:
                stats['new_best_unsupported_kept_current'] += 1
            elif best is None or not check_answer(best, flags)['passed']:
                stats['new_best_failed_checks_kept_current'] += 1
            else:
                row = copy.deepcopy(row)
                row['messages'][1]['content'] = best
                row['bestofn'] = {'teacher': cands[row['id']]['teacher'], 'reviewer': pick['reviewer'],
                                  'chosen': pick['best']}
                stats['replaced_with_new_best'] += 1
            chosen = row['messages'][1]['content']
            if worst is not None and pick['worst'] != pick['best'] and worst != chosen:
                dpo.append({'id': row['id'], 'prompt': row['messages'][:1], 'chosen': chosen, 'rejected': worst})
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
with open(out / 'dpo_pairs.jsonl', 'w') as f:
    for p in dpo:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')
summary = {'stats': dict(stats), 'dpo_pairs': len(dpo),
           'sha256': hashlib.sha256((out / 'verified.jsonl').read_bytes()).hexdigest()}
(out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
