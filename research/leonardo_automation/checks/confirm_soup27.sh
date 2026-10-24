#!/bin/bash
# Confirmation of round 27 (uniform soup) against the round 20 release: the same answers judged again in a fresh,
# separate run by all three judges (gpt-oss-20b, Gemma 4, Qwen3-235B) plus their majority, so round 27's gain over
# round 20 is checked against judge noise before it stays the Hugging Face release. Low priority.
# usage: confirm_soup27.sh   (log: ctl/confirm_soup27.log; results runs/confirm_soup27/)
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=confirm_soup27; O=$C/runs/$OUT; LOG=$R/ctl/$OUT.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
hp_clear() {
    local need=0 have=0 r
    for r in $(cat $R/ctl/HIGH_PRIORITY 2>/dev/null); do
        grep -q ' DONE: \| STOP: ' $R/ctl/$r/controller.log 2>/dev/null && continue
        need=$(( need + 1 ))
        squeue --me -h -o %j 2>/dev/null | grep -v -- -review | grep -q "^dk-$r" && have=$(( have + 1 ))
    done
    [ $have -ge $need ] && [ "$(squeue --me -h | wc -l)" -le 1 ] && [ -z "$(squeue --me -h -t PD -o %i)" ]
}
yield() { n=0; while [ $n -lt 3 ]; do sleep 60; if hp_clear; then n=$(( n + 1 )); else n=0; fi; done; }
run() { yield; j=$(cd $C && sbatch --parsable --nice=500 "$@" 2>/dev/null | tail -1)
        [[ "$j" =~ ^[0-9]+$ ]] || { log "submission refused; retrying"; sleep 120; return; }
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $O/answers
S=runs/vs_release_soup27/answers
for l in v2 v3 d1; do
    cp -n $S/${l}__base_closed.jsonl $S/${l}__base_rag.jsonl $S/${l}__rel_rag.jsonl $O/answers/
    cp -n $S/${l}__s42_rag.jsonl $O/answers/${l}__r27_rag.jsonl
done
left() { $VPY - <<PY | tail -1
from pathlib import Path
a=Path('runs/$OUT/answers'); j=Path('runs/$OUT/$1')
need=0
for label,n in [('v2',66),('v3',200),('d1',106)]:
    for p in a.glob(label+'__*.jsonl'):
        arm=p.name[len(label)+2:-6]
        if arm=='base_closed': continue
        need+=sum(not (j/f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0,n,34))
print(need)
PY
}
log "start"
( for a in 1 2 3 4; do [ "$(left judged)" = 0 ] && break; log "gpt-oss judge attempt $a"
    REPO=$C RUN=$OUT run --nodes=2 --job-name=dk-cfs27-judge --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_gemma)" = 0 ] && break; log "gemma judge attempt $a"
    REPO=$C RUN=$OUT run --nodes=2 --job-name=dk-cfs27-gemma --export=ALL,RUN=$OUT research/teacher_probe/judge_gemma.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_qwen)" = 0 ] && break; log "qwen3-235b judge attempt $a"
    REPO=$C RUN=$OUT run --nodes=2 --job-name=dk-cfs27-qwen --export=ALL,RUN=$OUT research/teacher_probe/judge_qwen235.sbatch; done ) &
wait
export PYTHONPATH=src:research/rag65
$VPY research/rag65/majority.py runs/$OUT >> $LOG 2>&1
for J in judged judged_gemma judged_qwen judged_majority; do
    [ -d $O/$J ] && $VPY research/rag65/paired.py $O/$J $O/PAIRED_$J.json r27_rag:rel_rag > $O/paired_$J.out 2>&1
done
log "DONE: $(python3 -c "
import json, os
for j in ('judged', 'judged_gemma', 'judged_qwen', 'judged_majority'):
    p = '$O/PAIRED_' + j + '.json'
    if os.path.exists(p):
        d = json.load(open(p))
        print(j, '; '.join(f\"{k.split('/')[0]} {v['a']:.3f} vs {v['b']:.3f} {v['diff']:+.3f}\" for k, v in d.items()))")"
