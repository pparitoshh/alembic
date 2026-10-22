#!/bin/bash
# Second-judge (Gemma) check of the round-12a win (distillation weight 0.8, LoRA r32): its seeds (full strength) vs
# round 11 (same seed, and seed 44 x0.5 = the current release), judged again with Gemma 4 26B, then paired.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
LOG=$R/ctl/r12a_gemma.log; log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
cd $C; A=runs/r12a_gemma/answers; mkdir -p $A
for f in runs/rag65kd4/answers/*__base_closed.jsonl runs/rag65kd4/answers/*__base_rag.jsonl runs/rag65kd4/answers/*__s4?_rag.jsonl runs/rag65kd4/answers/*__k3s4?_rag.jsonl runs/rag65kd4/answers/*__k3s44_x05_rag.jsonl; do cp -n $f $A/; done
left() { $VPY - <<'PY' | tail -1
from pathlib import Path
a = Path('runs/r12a_gemma/answers'); j = Path('runs/r12a_gemma/judged_gemma'); need = 0
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
    REPO=$C RUN=r12a_gemma bash $R/ctl/priority_submit.sh --nodes=3 --export=ALL,RUN=r12a_gemma research/teacher_probe/judge_gemma.sbatch
done
PYTHONPATH=src:research/rag65 $VPY research/rag65/paired.py runs/r12a_gemma/judged_gemma runs/r12a_gemma/PAIRED_gemma.json \
    s42_rag:k3s42_rag s43_rag:k3s43_rag s44_rag:k3s44_rag s42_rag:k3s44_x05_rag s43_rag:k3s44_x05_rag s44_rag:k3s44_x05_rag s43_rag:base_rag > runs/r12a_gemma/paired.out 2>&1
log "DONE: $(tr -d '\n' < runs/r12a_gemma/paired.out | head -c 2500)"
