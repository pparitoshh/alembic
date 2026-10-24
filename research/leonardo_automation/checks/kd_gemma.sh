#!/bin/bash
# Second-judge (Gemma) check of the logit-distillation win: KD seeds (full strength) vs round 5 (x0.2) and base+RAG,
# judged again with Gemma 4 26B into runs/kd_gemma/judged_gemma, then paired.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
LOG=$R/ctl/kd_gemma.log; log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
cd $C; A=runs/kd_gemma/answers; mkdir -p $A
for f in runs/rag65kd/answers/*__base_closed.jsonl runs/rag65kd/answers/*__base_rag.jsonl runs/rag65kd/answers/*__s4?_rag.jsonl runs/rag65kd/answers/*__r5s4?_x02_rag.jsonl; do cp -n $f $A/; done
left() { $VPY - <<'PY' | tail -1
from pathlib import Path
a = Path('runs/kd_gemma/answers'); j = Path('runs/kd_gemma/judged_gemma'); need = 0
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
    log "gemma judge attempt $attempt ($(left) units left)"
    REPO=$C RUN=kd_gemma bash $R/ctl/priority_submit.sh --nodes=3 --export=ALL,RUN=kd_gemma research/teacher_probe/judge_gemma.sbatch
done
PYTHONPATH=src:research/rag65 $VPY research/rag65/paired.py runs/kd_gemma/judged_gemma runs/kd_gemma/PAIRED_gemma.json \
    s42_rag:r5s42_x02_rag s43_rag:r5s43_x02_rag s44_rag:r5s44_x02_rag s44_rag:base_rag > runs/kd_gemma/paired.out 2>&1
log "DONE: $(tr -d '\n' < runs/kd_gemma/paired.out | head -c 2500)"
