#!/bin/bash
# Leaderboard: equal answer budget: judge the *_b768 copies (answers cut at 768 tokens, research/exp30/budget768.py), only
# its missing units with all three judges, then the 2-of-3 majority and the table (same run as the 13 models).
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
LOG=$R/ctl/leaderboard/b768.log; log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
cd $C; D=runs/leaderboard; A=$D/answers
left() { $VPY - <<PY | tail -1
from pathlib import Path
a = Path('$A'); j = Path('$D/$1'); need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
( for a in 1 2 3 4; do [ "$(left judged)" = 0 ] && break; log "gpt-oss judge attempt $a ($(left judged) units)"
    REPO=$C RUN=leaderboard bash $R/ctl/priority_submit.sh --nodes=2 --export=ALL,STAGE=judge,RUN=leaderboard research/rag65/stage.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_gemma)" = 0 ] && break; log "gemma judge attempt $a ($(left judged_gemma) units)"
    REPO=$C RUN=leaderboard bash $R/ctl/priority_submit.sh --nodes=2 --export=ALL,RUN=leaderboard research/teacher_probe/judge_gemma.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_qwen)" = 0 ] && break; log "qwen3-235b judge attempt $a ($(left judged_qwen) units)"
    REPO=$C RUN=leaderboard bash $R/ctl/priority_submit.sh --nodes=2 --export=ALL,RUN=leaderboard research/teacher_probe/judge_qwen235.sbatch; done ) &
wait
export PYTHONPATH=src:research/rag65
$VPY research/rag65/aggregate.py $D/judged $A $D/RESULT.json > /dev/null 2>&1
$VPY research/rag65/aggregate.py $D/judged_gemma $A $D/RESULT_gemma.json > /dev/null 2>&1
$VPY research/rag65/majority.py $D >> $LOG 2>&1
for j in judged_qwen judged_majority; do $VPY research/rag65/aggregate.py $D/$j $A $D/RESULT_${j#judged_}.json > /dev/null 2>&1; done
$VPY - > $R/ctl/leaderboard/table_majority_all.txt <<PY
import json
R = {n: json.load(open(f'$D/{f}'))['arms'] for n, f in [('gpt-oss', 'RESULT.json'), ('gemma', 'RESULT_gemma.json'), ('qwen235', 'RESULT_qwen.json'), ('MAJORITY', 'RESULT_majority.json')]}
arms = sorted({k.split('/', 1)[1] for k in R['MAJORITY']})
def m(r, a):
    v = [r.get(f'{l}/{a}', {}).get('win_rate') for l in ('v3', 'v2', 'd1')]
    return v, (sum(v) / 3 if all(x is not None for x in v) else None)
rows = sorted(arms, key=lambda a: -(m(R['MAJORITY'], a)[1] or 0))
print(f'{"model":18s} ' + ' | '.join(f'{n:>8s}' for n in R) + ' | majority v3 / v2 / d1')
for a in rows:
    means = [m(R[n], a)[1] for n in R]; v = m(R['MAJORITY'], a)[0]
    print(f'{a:18s} ' + ' | '.join('    -   ' if x is None else f'{x:8.3f}' for x in means) + ' | ' + ' / '.join('-' if x is None else f'{x:.3f}' for x in v))
PY
log "DONE: $(tr '\n' '|' < $R/ctl/leaderboard/table_majority_all.txt | cut -c1-3000)"
# Equal-budget table: each model's *_b768 copy replaces it where one exists (models within the budget are unchanged).
 - > /leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/ctl/leaderboard/table_majority_b768.txt <<PY
import json
R = {n: json.load(open(f'/{f}'))['arms'] for n, f in [('gpt-oss', 'RESULT.json'), ('gemma', 'RESULT_gemma.json'), ('qwen235', 'RESULT_qwen.json'), ('MAJORITY', 'RESULT_majority.json')]}
arms = sorted({k.split('/', 1)[1] for k in R['MAJORITY']})
pick = [a for a in arms if not a.endswith('_b768') and a + '_b768' not in arms] + [a for a in arms if a.endswith('_b768')]
def m(r, a):
    v = [r.get(f'{l}/{a}', {}).get('win_rate') for l in ('v3', 'v2', 'd1')]
    return v, (sum(v) / 3 if all(x is not None for x in v) else None)
rows = sorted(pick, key=lambda a: -(m(R['MAJORITY'], a)[1] or 0))
print('EQUAL 768-TOKEN ANSWER BUDGET (answers over budget cut at 768 tokens, re-judged)')
print(f'{model:20s} ' + ' | '.join(f'{n:>8s}' for n in R) + ' | majority v3 / v2 / d1')
for a in rows:
    means = [m(R[n], a)[1] for n in R]; v = m(R['MAJORITY'], a)[0]
    print(f'{a:20s} ' + ' | '.join('    -   ' if x is None else f'{x:8.3f}' for x in means) + ' | ' + ' / '.join('-' if x is None else f'{x:.3f}' for x in v))
PY
log B768 TABLE: 
