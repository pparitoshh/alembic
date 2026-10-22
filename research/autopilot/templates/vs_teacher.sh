#!/bin/bash
# After a round finishes, judge its answers head to head against the teacher (Qwen3-Next-80B + RAG, teacher_probe2):
# the reference arm "base_closed" is the teacher's answer, so 0.50 = as good as the teacher.
# usage: vs_teacher_auto.sh ROUND_DIR (e.g. rag65kd5)   log: ctl/vs_teacher_<ROUND>.log
set -uo pipefail
RD=$1
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=vs_teacher_$RD; LOG=$R/ctl/$OUT.log; log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
log "waiting for $RD to finish"
until grep -q ' DONE: \| STOP: ' $R/ctl/$RD/controller.log 2>/dev/null; do sleep 120; done
grep -q ' STOP: ' $R/ctl/$RD/controller.log && { log "STOP: $RD stopped; nothing to judge"; exit 1; }
# Reserve the next free slot at once (the factory yields while ctl/PRIORITY exists): the setup below takes minutes on
# the login node and the factory took both slots in that gap on Oct 21. priority_submit.sh rewrites and clears it.
echo "$OUT (setting up)" > $R/ctl/PRIORITY; trap 'rm -f $R/ctl/PRIORITY' EXIT
cd $C; A=runs/$OUT/answers; mkdir -p $A
for l in v2 v3 d1; do
    cp -n runs/teacher_probe2/answers/${l}__tnext_rag.jsonl $A/${l}__base_closed.jsonl
    for f in runs/$RD/answers/${l}__s4?_rag.jsonl runs/$RD/answers/${l}__s4?_x05_rag.jsonl; do [ -s $f ] && cp -n $f $A/; done
    cp -n runs/rag65kd3/answers/${l}__s44_x05_rag.jsonl $A/${l}__r11s44_x05_rag.jsonl
done
left() { $VPY - <<PY | tail -1
from pathlib import Path
a = Path('runs/$OUT/answers'); j = Path('runs/$OUT/judged'); need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
for attempt in 1 2 3 4; do
    [ "$(left)" = 0 ] && break
    log "judge attempt $attempt ($(left) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=3 --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch
done
PYTHONPATH=src:research/rag65 $VPY research/rag65/aggregate.py runs/$OUT/judged runs/$OUT/answers runs/$OUT/RESULT.json > runs/$OUT/aggregate.out 2>&1
log "DONE: $(python3 -c "
import json
a=json.load(open('runs/$OUT/RESULT.json'))['arms']
print('; '.join(f\"{k} {v['win_rate']:.3f}\" for k,v in sorted(a.items())))")"
