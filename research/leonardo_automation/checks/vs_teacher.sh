#!/bin/bash
# Student vs teacher, head to head: the judge's reference arm ("base_closed", which every arm is compared with) is
# the teacher's answer (Qwen3-Next-80B + RAG, teacher_probe2), so 0.50 = as good as the teacher, > 0.50 = better.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
LOG=$R/ctl/vs_teacher.log; log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
cd $C; A=runs/vs_teacher/answers; mkdir -p $A
for l in v2 v3 d1; do
    cp -n runs/teacher_probe2/answers/${l}__tnext_rag.jsonl $A/${l}__base_closed.jsonl   # the reference = teacher
    cp -n runs/rag65kd3/answers/${l}__s44_x05_rag.jsonl $A/${l}__r11s44_x05_rag.jsonl
    for s in 42 43 44; do cp -n runs/rag65kd3/answers/${l}__s${s}_x05_rag.jsonl $A/${l}__r11s${s}_x05_rag.jsonl; done
    cp -n runs/rag65kd3/answers/${l}__k1s44_rag.jsonl $A/${l}__k1s44_rag.jsonl
    cp -n runs/rag65kd3/answers/${l}__r5s43_x02_rag.jsonl $A/${l}__r5s43_x02_rag.jsonl
    cp -n runs/rag65kd3/answers/${l}__base_rag.jsonl $A/${l}__base_rag.jsonl
    cp -n runs/teacher_probe4/answers/${l}__t235_rag.jsonl $A/${l}__t235_rag.jsonl 2>/dev/null
done
left() { $VPY - <<'PY' | tail -1
from pathlib import Path
a = Path('runs/vs_teacher/answers'); j = Path('runs/vs_teacher/judged'); need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
for attempt in 1 2 3; do
    [ "$(left)" = 0 ] && break
    log "judge attempt $attempt ($(left) units left)"
    REPO=$C RUN=vs_teacher bash $R/ctl/priority_submit.sh --nodes=3 --export=ALL,STAGE=judge,RUN=vs_teacher research/rag65/stage.sbatch
done
PYTHONPATH=src:research/rag65 $VPY research/rag65/aggregate.py runs/vs_teacher/judged runs/vs_teacher/answers runs/vs_teacher/RESULT.json > runs/vs_teacher/aggregate.out 2>&1
log "DONE: $(python3 -c "
import json
a=json.load(open('runs/vs_teacher/RESULT.json'))['arms']
print('; '.join(f\"{k} {v['win_rate']:.3f}\" for k,v in sorted(a.items())))")"
