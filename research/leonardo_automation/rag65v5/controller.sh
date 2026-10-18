#!/bin/bash
# Round 5: round-4 set + top-up release (ai_factory_all_v2), ~58.6K rows; started standalone, alongside round 4's judging -> train 3 seeds -> scale -> answer -> judge -> aggregate.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
HERE="$(cd "$(dirname "$0")" && pwd)"
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
B7=/leonardo_work/EUHPC_D30_031/runs/a08trc09/student-mixed-v2-20261015T042338Z/bundle/data/b7_original.jsonl
remote() { bash -c "$*" 2>&1; }  # runs on the Leonardo login node itself
heartbeat() { touch "$HERE/HEARTBEAT"; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
# Job ids are validated: error text from a failed squeue/sbatch must never be waited on as a job.
wait_job() {
    [[ "$1" =~ ^[0-9]+$ ]] || { log "no valid job id ('$1'); retrying"; sleep 60; return; }
    while squeue -h -j "$1" 2>/dev/null | grep -q .; do heartbeat; sleep 90; done
    log "job $1: $(sacct -n -X -j "$1" --format=State,Elapsed 2>/dev/null | head -1 | xargs)"
}
submit() { local j; j=$(remote "cd $C && export REPO=$C RUN=rag65v5 && sbatch --parsable $* 2>/dev/null" | tail -1); [[ "$j" =~ ^[0-9]+$ ]] && echo "$j"; }
adopt() { squeue --me -h -n "$1" -o %i 2>/dev/null | grep -E '^[0-9]+$' | head -1; }
# Missing answer lines over every expected arm file (3 seeds x 3 strengths + seed-42 soups, closed and rag).
answers_missing() { remote "cd $C/$D && $VPY - <<'PY'
import json
from pathlib import Path
need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for s in (42, 43, 44):
        for v in ['', '_x05', '_x02'] + (['_soup', '_soup05'] if s == 42 else []):
            for mode in ('closed', 'rag'):
                f = Path(f'answers/{label}__s{s}{v}_{mode}.jsonl')
                ids = set()
                for x in (f.read_text().splitlines() if f.exists() else []):
                    try:
                        ids.add(json.loads(x)['id'])
                    except (ValueError, KeyError):
                        pass
                need += max(0, n - len(ids))
print(need)
PY" | tail -1; }
ENV="cd $C && export PYTHONPATH=src:research/rag65 HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 &&"
D=runs/rag65v5
PROSE=16314

log "round 5: waiting for the top-up merge (round-4 answers already exist)"
until grep -q ' MERGED: ' "$HERE/../factory_v2/controller.log" 2>/dev/null; do heartbeat; sleep 120; done
if ! remote "test -f $C/$D/data/verified.jsonl && echo ok" | grep -q ok; then
    remote "$ENV $VPY research/rag65/build_v5.py runs/rag65v4/data/verified.jsonl $D/data runs/ai_factory_all_v2/release/verified.jsonl:factory_v2" > "$HERE/data.out"
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
    j=$(adopt dk-rag65-train)
    if [ -n "$j" ]; then log "train round $round: adopting running job $j"; else
    j=$(submit research/rag65/train.sbatch); log "train round $round: job $j ($done_n/3 seeds done)"; fi
    wait_job "$j"
    remote "tail -2 $R/logs/train-$j-0.out" >> "$HERE/controller.log"
    # A TIMEOUT ends with SIGTERM tracebacks by design; only a non-timeout failure is an error.
    state=$(sacct -n -X -j "$j" --format=State 2>/dev/null | head -1 | xargs)
    if [ "$state" != TIMEOUT ] && remote "grep -l Traceback $R/logs/train-$j-*.err" | grep -q err; then
        remote "grep -h -B2 -A8 Traceback $R/logs/train-$j-*.err | tail -30" >> "$HERE/controller.log"
        log "STOP: training raised an exception (see above)"; exit 1
    fi
    # Failures without a traceback (node kill, OOM kill) are retried, but not forever.
    case "$state" in TIMEOUT|COMPLETED*) fails=0 ;; *) fails=$(( ${fails:-0} + 1 )) ;; esac
    [ "${fails:-0}" -ge 3 ] && { log "STOP: 3 consecutive training jobs failed ($state)"; exit 1; }
done
done_n=$(remote "ls $C/$D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l" | tail -1)
[ "$done_n" = 3 ] || { log "STOP: only $done_n/3 seeds trained after 30 rounds"; exit 1; }

log "scaling adapters"
remote "$ENV for s in 42 43 44; do for x in 0.5 0.2; do $VPY research/rag65/make_scaled.py $D/train/seed\$s/adapter \$x; done; done && ls $D/train/seed42"
# Seed soup (exact mean of the three seeds' updates, rank 48), served from seed 42's directory only.
remote "$ENV for x in 1.0 0.5; do $VPY research/rag65/make_soup.py $D/train/seed42/adapter_soup\$([ \$x = 0.5 ] && echo 05) \$x $D/train/seed42/adapter $D/train/seed43/adapter $D/train/seed44/adapter; done && ls $D/train/seed42"
for a in seed42/adapter seed42/adapter_x05 seed42/adapter_x02 seed42/adapter_soup seed42/adapter_soup05 \
         seed43/adapter seed43/adapter_x05 seed43/adapter_x02 seed44/adapter seed44/adapter_x05 seed44/adapter_x02; do
    test -s "$C/$D/train/$a/adapter_model.safetensors" || { log "STOP: $a has no weights"; exit 1; }
done

for round in $(seq 1 8); do  # ~2,300 answers per 30-min job; 20 arms x 372 need ~4 jobs
    m=$(answers_missing)
    [ "$m" = 0 ] && break
    j=$(adopt dk-rag65)
    if [ -n "$j" ]; then log "answer round $round: adopting running job $j"; else
    j=$(submit --nodes=1 --export=ALL,STAGE=answer research/rag65/stage.sbatch); log "answer round $round: job $j ($m answers missing)"; fi
    wait_job "$j"
done

m=$(answers_missing)
[ "$m" = 0 ] || { log "STOP: $m answers still missing after 8 answer jobs; not judging incomplete arms"; exit 1; }
for round in $(seq 1 8); do
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
    j=$(adopt dk-rag65)
    if [ -n "$j" ]; then log "judge round $round: adopting running job $j"; else
    j=$(submit --nodes=8 --export=ALL,STAGE=judge research/rag65/stage.sbatch); log "judge round $round: job $j ($left units left)"; fi
    wait_job "$j"
done

[ "$left" = 0 ] || { log "STOP: $left judge units left after 8 judge jobs"; exit 1; }
remote "$ENV $VPY research/rag65/aggregate.py $D/judged $D/answers $D/RESULT.json" > "$HERE/aggregate.out"
pairs=""
for x in 42 43 44; do pairs="$pairs s${x}_rag:base_rag s${x}_rag:r4s${x}_rag s${x}_x05_rag:base_rag s${x}_x05_rag:r4s${x}_x05_rag s${x}_x05_rag:r3s${x}_x05_rag s${x}_x02_rag:base_rag s${x}_x02_rag:r4s${x}_x02_rag s${x}_closed:r4s${x}_closed s${x}_x02_closed:r4s${x}_x02_closed"; done
pairs="$pairs s42_soup05_rag:base_rag s42_soup05_rag:s42_x05_rag s42_soup_rag:s42_rag s42_soup05_closed:s42_x05_closed"
remote "$ENV $VPY research/rag65/paired.py $D/judged $D/PAIRED.json $pairs" > "$HERE/paired.out"
grep -q '"v3/student_rag"' "$HERE/aggregate.out" || { log "STOP: aggregate failed: $(tail -5 "$HERE/aggregate.out")"; exit 1; }
log "DONE: $(tr -d '\n' < "$HERE/aggregate.out" | head -c 900)"
[ -x "$HERE/next.sh" ] && { log "starting next stage"; exec "$HERE/next.sh"; }
