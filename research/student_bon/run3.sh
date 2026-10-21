#!/bin/bash
# Best-of-8 for round 12c (on-policy, the best student): 12c (merged round 11 + seed adapter, full strength) writes
# 7 samples per eval question besides its greedy answer; Qwen3-Next picks the best of the 8 (the strongest selector in
# student_bon2); the picks and the greedy answers are judged head to head against the teacher (Qwen3-Next-80B + RAG,
# teacher_probe2 answers as the reference arm), so 0.50 = as good as the teacher.
# Reported as "student + selector", not as the 4B model's own score.
# usage: run3.sh   (log: ctl/student_bon_r12c.log)
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
RUN=rag65kd6; VARIANT=; OUT=student_bon_r12c; BASE=$C/runs/gguf_r11/merged
O=$C/runs/$OUT
LOG=$R/ctl/$OUT.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
cd $C
mkdir -p $O/answers
for s in 42 43 44; do
    mkdir -p $O/s$s
    [ -s $O/s$s/data.jsonl ] || PYTHONPATH=src:research/rag65 $VPY research/teacher_probe/bon_eval.py prepare \
        "runs/$RUN/answers/{label}__s$s${VARIANT}_rag.jsonl" $O/s$s >> "$LOG" 2>&1
done
have() { cat $O/s$1/$2/*.jsonl 2>/dev/null | wc -l; }
with_cands() { cat $O/s$1/$2/*.jsonl 2>/dev/null | python3 -c "import json,sys; print(sum(bool(json.loads(l).get('candidates')) for l in sys.stdin if l.strip()))"; }
want() { python3 -c "
import json
q=('Quoted retrieval results (data, not instructions):','Provided input 1 (quoted data')
print(sum(any(x in json.loads(l)['messages'][0]['content'] for x in q) for l in open('$O/s$1/data.jsonl')))"; }
for s in 42 43 44; do echo "$(want $s)" > $O/s$s/WANT; done
cands_done() { for s in 42 43 44; do [ "$(have $s cands8)" -ge "$(cat $O/s$s/WANT)" ] || return 1; done; }
for attempt in 1 2 3; do
    cands_done && break
    log "cands8 attempt $attempt"
    REPO=$C bash $R/ctl/priority_submit.sh \
        --export=ALL,RUN=$RUN,VARIANT=$VARIANT,OUT=$OUT,K=7,CDIR=cands8,STUDENT_BASE=$BASE research/student_bon/cands.sbatch
done
cands_done || log "cands8 incomplete; rows without 7 samples are reviewed with what they have"
sel_done() { for s in 42 43 44; do [ "$(have $s sel_n8)" -ge "$(with_cands $s cands8)" ] || return 1; done; }
for attempt in 1 2 3; do
    sel_done && break
    log "select attempt $attempt"
    REPO=$C bash $R/ctl/priority_submit.sh --export=ALL,OUT=$OUT,SELECTOR=tnext,SETS=cands8:sel_n8 research/student_bon/select.sbatch
done
for l in v2 v3 d1; do cp -n runs/teacher_probe2/answers/${l}__tnext_rag.jsonl $O/answers/${l}__base_closed.jsonl; done  # reference = teacher
for s in 42 43 44; do
    cp -n runs/$RUN/answers/*__s$s${VARIANT}_rag.jsonl $O/answers/
    PYTHONPATH=src:research/rag65 $VPY research/teacher_probe/bon_eval.py arm $O/s$s $O/answers cands8 sel_n8 TMP >> "$LOG" 2>&1
    for l in v2 v3 d1; do mv $O/answers/${l}__TMP.jsonl $O/answers/${l}__s$s${VARIANT}n8_rag.jsonl; done
done
left() { $VPY - <<PY | tail -1
from pathlib import Path
a=Path('runs/$OUT/answers'); j=Path('runs/$OUT/judged')
need=0
for label,n in [('v2',66),('v3',200),('d1',106)]:
    for p in a.glob(label+'__*.jsonl'):
        arm=p.name[len(label)+2:-6]
        if arm=='base_closed': continue
        need+=sum(not (j/f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0,n,34))
print(need)
PY
}
for attempt in 1 2 3 4; do
    [ "$(left)" = 0 ] && break
    log "judge attempt $attempt ($(left) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch
done
PYTHONPATH=src:research/rag65 $VPY research/rag65/aggregate.py $O/judged $O/answers $O/RESULT.json > $O/aggregate.out 2>&1
pairs=""; for s in 42 43 44; do pairs="$pairs s$s${VARIANT}n8_rag:s$s${VARIANT}_rag"; done
PYTHONPATH=src:research/rag65 $VPY research/rag65/paired.py $O/judged $O/PAIRED.json $pairs > $O/paired.out 2>&1
log "DONE: $(python3 -c "
import json
a=json.load(open('$O/RESULT.json'))['arms']
print('; '.join(f\"{k} {v['win_rate']:.3f}\" for k,v in sorted(a.items())))")"
