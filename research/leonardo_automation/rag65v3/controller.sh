#!/bin/bash
# Round 3: longer teacher-regenerated (REGEN_PROMPT=v2) RAFT answers + CC-BY v2 -> train 3 seeds -> scale -> answer -> judge -> aggregate.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
HERE="$(cd "$(dirname "$0")" && pwd)"
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
B7=/leonardo_work/EUHPC_D30_031/runs/a08trc09/student-mixed-v2-20261015T042338Z/bundle/data/b7_original.jsonl
remote() { bash -c "$*" 2>&1; }  # runs on the Leonardo login node itself
heartbeat() { touch "$HERE/HEARTBEAT"; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
wait_job() { while remote "squeue -h -j $1" | grep -q .; do heartbeat; sleep 90; done; log "job $1: $(remote "sacct -n -X -j $1 --format=State,Elapsed" | head -1 | xargs)"; }
submit() { remote "cd $C && export REPO=$C RUN=rag65v3 REGEN_PROMPT=v2 && sbatch --parsable $*" | tail -1; }
ENV="cd $C && export PYTHONPATH=src:research/rag65 HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 &&"
D=runs/rag65v3
PROSE=16314

for round in 1 2 3 4; do
    n=$(remote "cat $C/$D/regen/regen_*.jsonl 2>/dev/null | wc -l" | tail -1)
    [ "${n:-0}" -ge $((PROSE * 995 / 1000)) ] && break
    j=$(submit --nodes=8 --export=ALL,STAGE=regen research/rag65/stage.sbatch); log "regen round $round: job $j (have ${n:-0}/$PROSE)"
    wait_job "$j"
done

if ! remote "test -f $C/$D/data/verified.jsonl && echo ok" | grep -q ok; then
    remote "$ENV $VPY research/rag65/build_regen.py runs/rag65/raft/verified.jsonl $D/regen runs/ai_factory_ccby_v2/release/verified.jsonl $D/data" > "$HERE/data.out"
fi
log "DATA: $(tr -d '\n ' < "$HERE/data.out" | head -c 600)"
remote "cd $C && for s in 42 43 44; do mkdir -p $D/train/seed\$s && cp -n $D/data/verified.jsonl $D/train/seed\$s/; done
        mkdir -p $D/answers && cp -n runs/rag65/answers/*__base_*.jsonl $D/answers/"

for round in 1 2 3 4 5 6 7 8; do
    done_n=$(remote "ls $C/$D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l" | tail -1)
    [ "$done_n" = 3 ] && break
    j=$(remote "squeue --me -h -n dk-rag65-train -o %i" | head -1)
    if [ -n "$j" ]; then log "train round $round: adopting running job $j"; else
    j=$(submit research/rag65/train.sbatch); log "train round $round: job $j ($done_n/3 seeds done)"; fi
    wait_job "$j"
    remote "tail -2 $R/logs/train-$j-0.out" >> "$HERE/controller.log"
    # A TIMEOUT ends with SIGTERM tracebacks by design; only a non-timeout failure is an error.
    state=$(remote "sacct -n -X -j $j --format=State" | head -1 | xargs)
    if [ "$state" != TIMEOUT ] && remote "grep -l Traceback $R/logs/train-$j-*.err" | grep -q err; then
        remote "grep -h -B2 -A8 Traceback $R/logs/train-$j-*.err | tail -30" >> "$HERE/controller.log"
        log "STOP: training raised an exception (see above)"; exit 1
    fi
done
done_n=$(remote "ls $C/$D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l" | tail -1)
[ "$done_n" = 3 ] || { log "STOP: only $done_n/3 seeds trained after 8 rounds"; exit 1; }

log "scaling adapters"
remote "$ENV for s in 42 43 44; do for x in 0.5 0.2; do $VPY research/rag65/make_scaled.py $D/train/seed\$s/adapter \$x; done; done && ls $D/train/seed42"

for round in 1 2 3; do
    n=$(remote "cd $C/$D/answers 2>/dev/null && cat *.jsonl | wc -l" | tail -1)
    [ "${n:-0}" -ge $((20 * 372)) ] && break
    j=$(submit --nodes=1 --export=ALL,STAGE=answer research/rag65/stage.sbatch); log "answer round $round: job $j (have ${n:-0} answers)"
    wait_job "$j"
done

for round in 1 2 3 4; do
    left=$(remote "cd $C && $VPY - <<'PY'
from pathlib import Path
a=Path('runs/rag65v3/answers'); j=Path('runs/rag65v3/judged')
need=0
for label,n in [('v2',66),('v3',200),('d1',106)]:
    for p in a.glob(label+'__*.jsonl'):
        arm=p.name[len(label)+2:-6]
        if arm=='base_closed': continue
        need+=sum(not (j/f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0,n,34))
print(need)
PY" | tail -1)
    [ "$left" = 0 ] && break
    j=$(submit --nodes=8 --export=ALL,STAGE=judge research/rag65/stage.sbatch); log "judge round $round: job $j ($left units left)"
    wait_job "$j"
done

remote "$ENV $VPY research/rag65/aggregate.py $D/judged $D/answers $D/RESULT.json" > "$HERE/aggregate.out"
log "DONE: $(tr -d '\n' < "$HERE/aggregate.out" | head -c 900)"
