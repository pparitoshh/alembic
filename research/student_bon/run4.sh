#!/bin/bash
# Best-of-8 for the release 12e (round 12a merged + round 12e seed adapters, full strength): the student writes 7
# samples per eval question besides its greedy answer; Qwen3-Next (not an evaluation judge) picks the best of the 8.
# Picks and greedy answers are judged against closed-book base Qwen3-4B (the leaderboard scale) by gpt-oss-20b,
# Gemma 4 and Qwen3-235B in parallel, plus their 2-of-3 majority. Reported as "12e + best-of-8 selector", never as the
# 4B model's own single-answer score. Low priority: submits only while at most one other job of ours is queued.
# usage: run4.sh   (log: ctl/student_bon_r12e.log)
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
RUN=rag65kd9; VARIANT=; OUT=student_bon_r12e; BASE=$C/runs/r12a_merged
O=$C/runs/$OUT
LOG=$R/ctl/$OUT.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
# High-priority rounds keep their slot between jobs (data building, adapter scaling can take > 5 min on the login
# node; on Oct 22 a time-based quiet window still let low-priority jobs in). Low priority runs only while every
# unfinished round in ctl/HIGH_PRIORITY has its own job queued, at most one job of ours is queued, and no judge waits.
hp_clear() {
    local need=0 have=0 r
    for r in $(cat $R/ctl/HIGH_PRIORITY 2>/dev/null); do
        [ "$r" = "${RD:-}" ] && break  # only rounds listed before this one (strict order, no deadlock)
        grep -q ' DONE: \| STOP: ' $R/ctl/$r/controller.log 2>/dev/null && continue
        need=$(( need + 1 ))
        squeue --me -h -o %j 2>/dev/null | grep -v -- -review | grep -q "^dk-$r" && have=$(( have + 1 ))
    done
    [ $have -ge $need ] && [ "$(squeue --me -h | wc -l)" -le 1 ] && [ -z "$(squeue --me -h -t PD -o %i)" ]  # a running priority job holds ctl/PRIORITY for its whole run; only a pending one should block
}
yield() { n=0; while [ $n -lt 3 ]; do sleep 60; if hp_clear; then n=$(( n + 1 )); else n=0; fi; done; }
while squeue --me -h -o %j | grep -q "^dk-bon12e"; do sleep 60; done  # a restart waits for its running job first
run() { yield; j=$(cd $C && sbatch --parsable --nice=500 "$@" 2>/dev/null | tail -1)  # wait for the job to end
        [[ "$j" =~ ^[0-9]+$ ]] || { log "submission refused; retrying"; sleep 120; return; }
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C
test -s $BASE/config.json || { log "STOP: $BASE missing"; exit 1; }
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
for attempt in 1 2 3 4; do
    cands_done && break
    log "cands8 attempt $attempt"
    REPO=$C RUN=$RUN VARIANT=$VARIANT OUT=$OUT K=7 CDIR=cands8 STUDENT_BASE=$BASE run --job-name=dk-bon12e-cands research/student_bon/cands.sbatch
done
cands_done || log "cands8 incomplete; rows without 7 samples are reviewed with what they have"
sel_done() { for s in 42 43 44; do [ "$(have $s sel_n8)" -ge "$(with_cands $s cands8)" ] || return 1; done; }
for attempt in 1 2 3 4; do
    sel_done && break
    log "select attempt $attempt"
    REPO=$C OUT=$OUT SELECTOR=tnext SETS=cands8:sel_n8 run --job-name=dk-bon12e-select research/student_bon/select.sbatch
done
cp -n runs/$RUN/answers/*__base_closed.jsonl runs/$RUN/answers/*__base_rag.jsonl $O/answers/  # reference = base closed-book
for s in 42 43 44; do
    cp -n runs/$RUN/answers/*__s$s${VARIANT}_rag.jsonl $O/answers/
    [ -s $O/answers/v3__s$s${VARIANT}n8_rag.jsonl ] && continue
    PYTHONPATH=src:research/rag65 $VPY research/teacher_probe/bon_eval.py arm $O/s$s $O/answers cands8 sel_n8 TMP >> "$LOG" 2>&1
    for l in v2 v3 d1; do mv $O/answers/${l}__TMP.jsonl $O/answers/${l}__s$s${VARIANT}n8_rag.jsonl; done
done
left() { $VPY - <<PY | tail -1
from pathlib import Path
a=Path('runs/$OUT/answers'); j=Path('runs/$OUT/$1')
need=0
for label,n in [('v2',66),('v3',200),('d1',106)]:
    for p in a.glob(label+'__*.jsonl'):
        arm=p.name[len(label)+2:-6]
        if arm=='base_closed': continue
        need+=sum(not (j/f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0,n,34))
print(need)
PY
}
( for attempt in 1 2 3 4; do [ "$(left judged)" = 0 ] && break; log "gpt-oss judge attempt $attempt"
    REPO=$C RUN=$OUT run --nodes=4 --job-name=dk-bon12e-judge --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch; done ) &
( for attempt in 1 2 3 4; do [ "$(left judged_gemma)" = 0 ] && break; log "gemma judge attempt $attempt"
    REPO=$C RUN=$OUT run --nodes=3 --job-name=dk-bon12e-gemma --export=ALL,RUN=$OUT research/teacher_probe/judge_gemma.sbatch; done ) &
( for attempt in 1 2 3 4; do [ "$(left judged_qwen)" = 0 ] && break; log "qwen3-235b judge attempt $attempt"
    REPO=$C RUN=$OUT run --nodes=4 --job-name=dk-bon12e-qwen --export=ALL,RUN=$OUT research/teacher_probe/judge_qwen235.sbatch; done ) &
wait
export PYTHONPATH=src:research/rag65
$VPY research/rag65/majority.py runs/$OUT >> $LOG 2>&1
pairs=""; for s in 42 43 44; do pairs="$pairs s$s${VARIANT}n8_rag:s$s${VARIANT}_rag"; done
for J in judged judged_gemma judged_qwen judged_majority; do
    [ -d $O/$J ] || continue
    $VPY research/rag65/aggregate.py $O/$J $O/answers $O/RESULT_$J.json > $O/aggregate_$J.out 2>&1
    $VPY research/rag65/paired.py $O/$J $O/PAIRED_$J.json $pairs > $O/paired_$J.out 2>&1
done
log "DONE: $(python3 -c "
import json, os
for j in ('judged', 'judged_gemma', 'judged_qwen', 'judged_majority'):
    p = '$O/RESULT_' + j + '.json'
    if os.path.exists(p):
        a = json.load(open(p))['arms']
        print(j, '; '.join(f\"{k} {v['win_rate']:.3f}\" for k, v in sorted(a.items()) if k.startswith('v3/')))")"
