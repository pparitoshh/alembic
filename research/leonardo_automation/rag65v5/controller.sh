#!/bin/bash
# Round 5: round-4 set + top-up (ai_factory_all_v2) and coverage (ai_factory_all_v3) releases -> train 3 seeds -> scale -> answer -> judge -> aggregate.
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
submit() { remote "cd $C && export REPO=$C RUN=rag65v5 && sbatch --parsable $*" | tail -1; }
ENV="cd $C && export PYTHONPATH=src:research/rag65 HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 &&"
D=runs/rag65v5
PROSE=16314

log "round 5: waiting for the round-4 scores and the top-up and coverage merges"
until grep -q ' DONE: ' "$HERE/../rag65v4/controller.log" 2>/dev/null && grep -q ' MERGED: ' "$HERE/../factory_v2/controller.log" 2>/dev/null \
        && grep -q ' MERGED: ' "$HERE/../factory_v3/controller.log" 2>/dev/null; do heartbeat; sleep 120; done
if ! remote "test -f $C/$D/data/verified.jsonl && echo ok" | grep -q ok; then
    remote "$ENV $VPY research/rag65/build_v5.py runs/rag65v4/data/verified.jsonl $D/data runs/ai_factory_all_v2/release/verified.jsonl:factory_v2 runs/ai_factory_all_v3/release/verified.jsonl:factory_v3" > "$HERE/data.out"
fi
log "DATA: $(tr -d '\n ' < "$HERE/data.out" | head -c 700)"
remote "cd $C && for s in 42 43 44; do mkdir -p $D/train/seed\$s && cp -n $D/data/verified.jsonl $D/train/seed\$s/; done
        mkdir -p $D/answers && cp -n runs/rag65/answers/*__base_*.jsonl $D/answers/
        # Earlier rounds' answers are re-judged in this run, so rounds compare paired, without judge drift.
        for f in runs/rag65v4/answers/*__s4?_*.jsonl; do b=\$(basename \$f); cp -n \$f $D/answers/\${b/__s/__r4s}; done
        for r in 3; do for f in runs/rag65v\$r/answers/*__s4?_x0[25]_*.jsonl runs/rag65v\$r/answers/*__s4?_rag.jsonl runs/rag65v\$r/answers/*__s4?_closed.jsonl; do
            b=\$(basename \$f); cp -n \$f $D/answers/\${b/__s/__r\${r}s}; done; done"

for round in $(seq 1 30); do
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
[ "$done_n" = 3 ] || { log "STOP: only $done_n/3 seeds trained after 30 rounds"; exit 1; }

log "scaling adapters"
remote "$ENV for s in 42 43 44; do for x in 0.5 0.2; do $VPY research/rag65/make_scaled.py $D/train/seed\$s/adapter \$x; done; done && ls $D/train/seed42"
# Seed soup (exact mean of the three seeds' updates, rank 48), served from seed 42's directory only.
remote "$ENV for x in 1.0 0.5; do $VPY research/rag65/make_soup.py $D/train/seed42/adapter_soup\$([ \$x = 0.5 ] && echo 05) \$x $D/train/seed42/adapter $D/train/seed43/adapter $D/train/seed44/adapter; done && ls $D/train/seed42"

for round in 1 2 3; do
    n=$(remote "cd $C/$D/answers 2>/dev/null && cat *__s4*.jsonl | wc -l" | tail -1)
    [ "${n:-0}" -ge $((20 * 372)) ] && break  # 3 seeds x 3 strengths x 2 modes + 2 soup arms
    j=$(remote "squeue --me -h -n dk-rag65 -o %i" | head -1)
    if [ -n "$j" ]; then log "answer round $round: adopting running job $j"; else
    j=$(submit --nodes=1 --export=ALL,STAGE=answer research/rag65/stage.sbatch); log "answer round $round: job $j (have ${n:-0} answers)"; fi
    wait_job "$j"
done

for round in 1 2 3 4 5 6; do
    left=$(remote "cd $C && $VPY - <<'PY'
from pathlib import Path
a=Path('runs/rag65v5/answers'); j=Path('runs/rag65v5/judged')
need=0
for label,n in [('v2',66),('v3',200),('d1',106)]:
    for p in a.glob(label+'__*.jsonl'):
        arm=p.name[len(label)+2:-6]
        if arm=='base_closed': continue
        need+=sum(not (j/f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0,n,34))
print(need)
PY" | tail -1)
    [ "$left" = 0 ] && break
    j=$(remote "squeue --me -h -n dk-rag65 -o %i" | head -1)
    if [ -n "$j" ]; then log "judge round $round: adopting running job $j"; else
    j=$(submit --nodes=8 --export=ALL,STAGE=judge research/rag65/stage.sbatch); log "judge round $round: job $j ($left units left)"; fi
    wait_job "$j"
done

remote "$ENV $VPY research/rag65/aggregate.py $D/judged $D/answers $D/RESULT.json" > "$HERE/aggregate.out"
pairs=""
for x in 42 43 44; do pairs="$pairs s${x}_rag:base_rag s${x}_rag:r4s${x}_rag s${x}_x05_rag:base_rag s${x}_x05_rag:r4s${x}_x05_rag s${x}_x05_rag:r3s${x}_x05_rag s${x}_x02_rag:base_rag s${x}_x02_rag:r4s${x}_x02_rag s${x}_closed:r4s${x}_closed s${x}_x02_closed:r4s${x}_x02_closed"; done
pairs="$pairs s42_soup05_rag:base_rag s42_soup05_rag:s42_x05_rag s42_soup_rag:s42_rag s42_soup05_closed:s42_x05_closed"
remote "$ENV $VPY research/rag65/paired.py $D/judged $D/PAIRED.json $pairs" > "$HERE/paired.out"
grep -q '"v3/student_rag"' "$HERE/aggregate.out" || { log "STOP: aggregate failed: $(tail -5 "$HERE/aggregate.out")"; exit 1; }
log "DONE: $(tr -d '\n' < "$HERE/aggregate.out" | head -c 900)"
[ -x "$HERE/next.sh" ] && { log "starting next stage"; exec "$HERE/next.sh"; }
