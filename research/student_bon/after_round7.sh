#!/bin/bash
# After round 7 is scored: pick its best retrieval strength by the mean of v2/v3/d1 and run test-time best-of-n on it.
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
until grep -q ' DONE: ' $R/ctl/rag65v7/controller.log 2>/dev/null && grep -q ' DONE: ' $R/ctl/student_bon_r5.log 2>/dev/null; do sleep 120; done
V=$(python3 - <<PY
import json
m = json.load(open('$C/runs/rag65v7/RESULT.json'))['seed_means']
best = max(['', '_x05', '_x02'], key=lambda v: sum(m.get(f'{s}/student{v}_rag', {}).get('mean', 0) for s in ('v2', 'v3', 'd1')))
print(best or 'FULL')
PY
)
[ "$V" = FULL ] && V=""
echo "$(date -u +%FT%TZ) round 7 best variant: '${V:-x1.0}'" >> $R/ctl/student_bon_r7.log
exec bash $C/research/student_bon/run.sh rag65v7 "$V" student_bon_r7
