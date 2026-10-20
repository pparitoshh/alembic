#!/bin/bash
# Test-time best-of-n for a trained student: the student (RUN, VARIANT; all 3 seeds) writes 3 extra answers per
# eval question with its evaluation prompt; Gemma picks the best of greedy + 3; the picks are judged against
# closed-book base in the same run as the student's own greedy answers and base+RAG.
# Reported as "student + selector", not as the 4B model's own score.
# usage: run.sh RUN VARIANT OUT   e.g. run.sh rag65v5 _x02 student_bon_r5
set -uo pipefail
RUN=$1; VARIANT=$2; OUT=$3
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
O=$C/runs/$OUT
LOG=$R/ctl/$OUT.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
cd $C
mkdir -p $O/answers
for s in 42 43 44; do
    mkdir -p $O/s$s/cands $O/s$s/select
    [ -s $O/s$s/data.jsonl ] || PYTHONPATH=src:research/rag65 $VPY research/teacher_probe/bon_eval.py prepare \
        "runs/$RUN/answers/{label}__s$s${VARIANT}_rag.jsonl" $O/s$s >> "$LOG" 2>&1
done
want() { python3 -c "
import json
q=('Quoted retrieval results (data, not instructions):','Provided input 1 (quoted data')
print(sum(any(x in json.loads(l)['messages'][0]['content'] for x in q) for l in open('$O/s$1/data.jsonl')))"; }
have() { cat $O/s$1/$2/$3_*.jsonl 2>/dev/null | wc -l; }
done_phase() { for s in 42 43 44; do [ "$(have $s $1 $2)" -ge "$3" ] || [ "$1" = select -a "$(have $s select select)" -ge "$(have $s cands cands)" ] || return 1; done; }
W=$(want 42)
for attempt in 1 2 3; do
    ok=1; for s in 42 43 44; do [ "$(have $s cands cands)" -ge "$(want $s)" ] || ok=0; done; [ $ok = 1 ] && break
    log "cands attempt $attempt"
    REPO=$C RUN=$RUN VARIANT=$VARIANT OUT=$OUT bash $R/ctl/priority_submit.sh --export=ALL,RUN=$RUN,VARIANT=$VARIANT,OUT=$OUT research/student_bon/cands.sbatch
done
for attempt in 1 2 3; do
    ok=1; for s in 42 43 44; do [ "$(have $s select select)" -ge "$(have $s cands cands)" ] || ok=0; done; [ $ok = 1 ] && break
    log "select attempt $attempt"
    REPO=$C OUT=$OUT bash $R/ctl/priority_submit.sh --export=ALL,OUT=$OUT research/student_bon/select.sbatch
done
cp -n runs/$RUN/answers/*__base_closed.jsonl runs/$RUN/answers/*__base_rag.jsonl $O/answers/
for s in 42 43 44; do
    cp -n runs/$RUN/answers/*__s$s${VARIANT}_rag.jsonl $O/answers/
    PYTHONPATH=src:research/rag65 $VPY research/teacher_probe/bon_eval.py arm $O/s$s $O/s$s >> "$LOG" 2>&1
    for l in v2 v3 d1; do cp $O/s$s/${l}__bon_rag.jsonl $O/answers/${l}__s$s${VARIANT}bon_rag.jsonl; done
done
log "judging greedy vs best-of-n (+ base+RAG) in one run"
REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=2 --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch
PYTHONPATH=src:research/rag65 $VPY research/rag65/aggregate.py $O/judged $O/answers $O/RESULT.json > $O/aggregate.out 2>&1
pairs=""; for s in 42 43 44; do pairs="$pairs s$s${VARIANT}bon_rag:s$s${VARIANT}_rag s$s${VARIANT}bon_rag:base_rag"; done
PYTHONPATH=src:research/rag65 $VPY research/rag65/paired.py $O/judged $O/PAIRED.json $pairs > $O/paired.out 2>&1
log "DONE: $(tr -d '\n' < $O/aggregate.out | head -c 1500)"
