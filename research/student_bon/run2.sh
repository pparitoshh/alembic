#!/bin/bash
# Selector upgrade on round 5 (x0.2, 3 seeds): more candidates and a stronger reviewer, all judged in one run.
#   g4 = greedy + 3 samples, Gemma reviewer (the current main result, re-judged here)
#   g8 = greedy + 7 samples, Gemma      n4 = greedy + 3, Qwen3-Next reviewer      n8 = greedy + 7, Qwen3-Next
# usage: run2.sh   (log: ctl/student_bon2.log)
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
RUN=rag65v5; VARIANT=_x02; OUT=student_bon2
O=$C/runs/$OUT; P5=$C/runs/student_bon_r5
LOG=$R/ctl/$OUT.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
cd $C
mkdir -p $O/answers
for s in 42 43 44; do  # same questions, greedy answers and 3-sample candidates as the round-5 selector run
    mkdir -p $O/s$s
    cp -n $P5/s$s/data.jsonl $O/s$s/
    [ -d $O/s$s/cands ] || cp -r $P5/s$s/cands $O/s$s/cands
    [ -d $O/s$s/sel_g4 ] || cp -r $P5/s$s/select $O/s$s/sel_g4
done
have() { cat $O/s$1/$2/*.jsonl 2>/dev/null | wc -l; }
# Rows whose samples all ran past the length limit have no candidates: select.py skips them (greedy answer stays).
with_cands() { cat $O/s$1/$2/*.jsonl 2>/dev/null | python3 -c "import json,sys; print(sum(bool(json.loads(l).get('candidates')) for l in sys.stdin if l.strip()))"; }
all_ge() { for s in 42 43 44; do [ "$(have $s $1)" -ge "$(with_cands $s $2)" ] || return 1; done; }
want() { python3 -c "
import json
q=('Quoted retrieval results (data, not instructions):','Provided input 1 (quoted data')
print(sum(any(x in json.loads(l)['messages'][0]['content'] for x in q) for l in open('$O/s$1/data.jsonl')))"; }
for s in 42 43 44; do echo "$(want $s)" > $O/s$s/WANT; done
cands8_done() { for s in 42 43 44; do [ "$(have $s cands8)" -ge "$(cat $O/s$s/WANT)" ] || return 1; done; }
# Jobs run in the background so the 1-node phases use both debug slots.
for attempt in 1 2 3; do
    cands8_done && break
    log "cands8 attempt $attempt"
    REPO=$C RUN=$RUN VARIANT=$VARIANT OUT=$OUT bash $R/ctl/priority_submit.sh \
        --export=ALL,RUN=$RUN,VARIANT=$VARIANT,OUT=$OUT,K=7,CDIR=cands8 research/student_bon/cands.sbatch
done
cands8_done || log "cands8 incomplete; rows without 7 samples are reviewed with what they have"
for attempt in 1 2 3; do
    ok_g=1; all_ge sel_g8 cands8 || ok_g=0
    ok_n=1; { all_ge sel_n4 cands && all_ge sel_n8 cands8; } || ok_n=0
    [ $ok_g = 1 ] && [ $ok_n = 1 ] && break
    log "select attempt $attempt (gemma done=$ok_g, qwen3-next done=$ok_n)"
    [ $ok_g = 1 ] || REPO=$C OUT=$OUT bash $R/ctl/priority_submit.sh \
        --export=ALL,OUT=$OUT,SELECTOR=gemma,SETS=cands8:sel_g8 research/student_bon/select.sbatch &
    [ $ok_n = 1 ] || { sleep 90; REPO=$C OUT=$OUT bash $R/ctl/priority_submit.sh \
        --export=ALL,OUT=$OUT,SELECTOR=tnext,SETS="cands:sel_n4 cands8:sel_n8" research/student_bon/select.sbatch; } &
    wait
done
cp -n runs/$RUN/answers/*__base_closed.jsonl runs/$RUN/answers/*__base_rag.jsonl $O/answers/
for s in 42 43 44; do
    cp -n runs/$RUN/answers/*__s$s${VARIANT}_rag.jsonl $O/answers/
    for a in g4:cands:sel_g4 g8:cands8:sel_g8 n4:cands:sel_n4 n8:cands8:sel_n8; do
        IFS=: read arm cd sd <<< "$a"
        PYTHONPATH=src:research/rag65 $VPY research/teacher_probe/bon_eval.py arm $O/s$s $O/answers $cd $sd TMP >> "$LOG" 2>&1
        for l in v2 v3 d1; do mv $O/answers/${l}__TMP.jsonl $O/answers/${l}__s$s${VARIANT}${arm}_rag.jsonl; done
    done
done
left() { $VPY - <<'PY' | tail -1
from pathlib import Path
a=Path('runs/student_bon2/answers'); j=Path('runs/student_bon2/judged')
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
pairs=""
for s in 42 43 44; do for a in g8 n4 n8; do pairs="$pairs s$s${VARIANT}${a}_rag:s$s${VARIANT}g4_rag"; done; pairs="$pairs s$s${VARIANT}g4_rag:s$s${VARIANT}_rag"; done
PYTHONPATH=src:research/rag65 $VPY research/rag65/paired.py $O/judged $O/PAIRED.json $pairs > $O/paired.out 2>&1
log "DONE: $(tr -d '\n' < $O/aggregate.out | head -c 2000)"
