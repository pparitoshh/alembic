#!/bin/bash
# Cross-check of the bigger-teacher probe with the Gemma judge (gpt-oss-120b might be favoured by the gpt-oss-20b
# judge): judge runs/teacher_probe4 answers again into judged_gemma, then pair goss120/t235 against tnext.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=$C/runs/teacher_probe4
LOG=$R/ctl/teacher_probe4.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
left() { (cd $C && $VPY - <<'PY' | tail -1
from pathlib import Path
a = Path('runs/teacher_probe4/answers'); j = Path('runs/teacher_probe4/judged_gemma')
need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
); }
for attempt in 1 2 3 4; do
    [ "$(left)" = 0 ] && break
    log "gemma judge attempt $attempt ($(left) units left)"
    (cd $C && REPO=$C RUN=teacher_probe4 bash $R/ctl/priority_submit.sh --nodes=2 --export=ALL,RUN=teacher_probe4 research/teacher_probe/judge_gemma.sbatch)
done
cd $C && PYTHONPATH=src:research/rag65 $VPY research/rag65/paired.py "$OUT/judged_gemma" "$OUT/PAIRED_gemma.json" \
    goss120_rag:tnext_rag t235_rag:tnext_rag tnext_rag:base_rag > "$OUT/paired_gemma.out" 2>&1
log "GEMMA DONE: $(tr -d '\n' < "$OUT/paired_gemma.out" | head -c 2000)"
