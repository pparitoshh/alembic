#!/bin/bash
# GOAL.md G5 (filtered vs unfiltered data) and G14 (curriculum): rounds ab39f (7,000 verified v11 rows), ab40u (7,000
# raw generations) and ab41c (ab39f's rows sorted short -> long), all trained on base Qwen3-4B with 3 seeds, judged in
# ONE run by gpt-oss-20b, Gemma 4 and Qwen3-235B (+ 2-of-3 majority). Arms f_s4X / u_s4X / c_s4X (full-strength
# adapters); paired comparisons u vs f and c vs f per seed. log: ctl/vs_ablation.log, results runs/vs_ablation/
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=vs_ablation; LOG=$R/ctl/$OUT.log; log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
log "waiting for ab39f, ab40u, ab41c"
for r in ab39f ab40u ab41c; do until grep -q ' DONE: \| STOP: ' $S/ctl/$r/controller.log 2>/dev/null; do sleep 120; done
    grep -q ' STOP: ' $S/ctl/$r/controller.log && { log "STOP: $r stopped"; exit 1; }; done
cd $C; A=runs/$OUT/answers; mkdir -p $A
cp -n runs/vs_release_exp30/answers/*__base_closed.jsonl runs/vs_release_exp30/answers/*__base_rag.jsonl $A/
for pair in f:ab39f u:ab40u c:ab41c; do t=${pair%%:*}; r=${pair#*:}
    for l in v2 v3 d1; do for s in 42 43 44; do cp -n $S/code/runs/$r/answers/${l}__s${s}_rag.jsonl $A/${l}__${t}_s${s}_rag.jsonl; done; done; done
left() { $VPY - <<PY | tail -1
from pathlib import Path
a = Path('runs/$OUT/answers'); j = Path('runs/$OUT/$1'); need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
( for a in 1 2 3 4; do [ "$(left judged)" = 0 ] && break; log "gpt-oss judge attempt $a ($(left judged) left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_gemma)" = 0 ] && break; log "gemma judge attempt $a"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=3 --export=ALL,RUN=$OUT research/teacher_probe/judge_gemma.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_qwen)" = 0 ] && break; log "qwen3-235b judge attempt $a"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT research/teacher_probe/judge_qwen235.sbatch; done ) &
wait
$VPY research/rag65/majority.py runs/$OUT >> $LOG 2>&1
pairs=""; for s in 42 43 44; do pairs="$pairs u_s${s}_rag:f_s${s}_rag c_s${s}_rag:f_s${s}_rag"; done
export PYTHONPATH=src:research/rag65
for J in judged judged_gemma judged_qwen judged_majority; do
    $VPY research/rag65/aggregate.py runs/$OUT/$J runs/$OUT/answers runs/$OUT/RESULT_$J.json > /dev/null 2>&1
    $VPY research/rag65/paired.py runs/$OUT/$J runs/$OUT/PAIRED_$J.json $pairs > runs/$OUT/paired_$J.out 2>&1; done
log "DONE: $(python3 -c "
import json
d = json.load(open('runs/$OUT/PAIRED_judged_majority.json'))
print('; '.join(f\"{k} {v['diff']:+.3f}\" for k, v in d.items()))")"
