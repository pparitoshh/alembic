#!/bin/bash
# Oct 25: re-judge the equal-1536 variant-e leaderboard answers (runs/lb1536e, unchanged) against the REVIEWED test sets
# (data/eval/clean: duplicates and broken-premise rows dropped, wrong references fixed; AI review by Claude, approved by
# Vaibhav). Same three judges + majority. Log: ctl/lbclean.log
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=lbclean; O=$C/runs/$OUT; A=$O/answers; LOG=$R/ctl/$OUT.log; E=$C/data/eval/clean
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
cd $C; mkdir -p $A
until [ -s $E/v2.jsonl ] && [ -s $E/v3.jsonl ] && [ -s $E/d1.jsonl ]; do sleep 60; done
export EVAL_DIR=$E
cp -n $C/runs/lb1536e/answers/*.jsonl $A/
log "start: $(wc -l < $E/v3.jsonl) v3, $(wc -l < $E/v2.jsonl) v2, $(wc -l < $E/d1.jsonl) d1 questions; $(ls $A | wc -l) answer files"
left() { $VPY - "$A" "$O/$1" "$E" <<'PY' | tail -1
import sys
from pathlib import Path
a = Path(sys.argv[1]); j = Path(sys.argv[2]); e = Path(sys.argv[3]); need = 0
for label in ('v2', 'v3', 'd1'):
    n = sum(1 for _ in open(e / (label + '.jsonl')))
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
( for a in 1 2 3 4; do [ "$(left judged)" = 0 ] && break; log "gpt-oss judge attempt $a ($(left judged) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,STAGE=judge,RUN=$OUT,EVAL_DIR=$E research/rag65/stage.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_gemma)" = 0 ] && break; log "gemma judge attempt $a ($(left judged_gemma) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT,EVAL_DIR=$E research/teacher_probe/judge_gemma.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_qwen)" = 0 ] && break; log "qwen3-235b judge attempt $a ($(left judged_qwen) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT,EVAL_DIR=$E research/teacher_probe/judge_qwen235.sbatch; done ) &
wait
export PYTHONPATH=src:research/rag65
$VPY research/rag65/majority.py runs/$OUT >> $LOG 2>&1
for J in judged judged_gemma judged_qwen judged_majority; do $VPY research/rag65/aggregate.py $O/$J $A $O/RESULT_$J.json > /dev/null 2>&1; done
summary=$(python3 - "$O" <<'PY'
import json, sys
for j in ('judged', 'judged_gemma', 'judged_qwen', 'judged_majority'):
    a = json.load(open(sys.argv[1] + '/RESULT_' + j + '.json'))['arms']
    arms = [x for x in ('le_t235_rag', 'le_tnext_rag', 'le_gptoss120_rag', 'le_gemma4_rag', 'le_gptoss20_rag', 'le_q32_rag',
                        'le_r43_rag', 'le_r27_rag') if all(l + '/' + x in a for l in ('v3', 'v2', 'd1'))]
    def m(arm): return [a[l + '/' + arm]['win_rate'] for l in ('v3', 'v2', 'd1')]
    print(j, ' | '.join('%s %.3f (%.3f/%.3f/%.3f)' % (arm, sum(m(arm)) / 3, *m(arm)) for arm in arms))
PY
)
log "DONE: $summary"
