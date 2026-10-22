#!/bin/bash
# Round 11: logit distillation on round 5's (Slurm-heavy) question set: the 46,732 single-turn prose rows of the
# best-of-n set (18,974 Qwen3-Next answers + 27,758 old-teacher answers; tool rows left out, they need tool-schema
# rendering). Qwen3-Next scores every answer token, old answers included. Round 10 showed that more AI Factory rows
# did not help these Slurm-heavy evals; this tests relevant data. Judged with distillation 1 (k1s*) and round 5.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
HERE="$(cd "$(dirname "$0")" && pwd)"
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
D=runs/rag65kd3
KD=$C/$D/teacher_top20
DATA=$C/runs/rag65kd3/data/verified.jsonl
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
heartbeat() { touch "$HERE/HEARTBEAT"; }
adopt() { squeue --me -h -n "$1" -o %i 2>/dev/null | grep -E '^[0-9]+$' | head -1; }
wait_job() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do heartbeat; sleep 90; done
             log "job $1: $(sacct -n -X -j "$1" --format=State,Elapsed 2>/dev/null | head -1 | xargs)"; }
submit() { (cd $C && REPO=$C RUN=rag65kd3 CFG=rag65kd3 KD_DIR=$KD DATA=$DATA sbatch --parsable "$@" 2>/dev/null | tail -1); }
cd $C
mkdir -p $D/train $KD $D/data
for s in 42 43 44; do
    [ -s configs/rag65kd3_seed$s.json ] || sed "s#runs/rag65kd/train#runs/rag65kd3/train#" configs/rag65kd_seed$s.json > configs/rag65kd3_seed$s.json
    mkdir -p $D/train/seed$s && cp -n $DATA $D/train/seed$s/verified.jsonl
done
for f in $C/runs/rag65kd/teacher_top20/kd_*.npz; do b=$(basename $f); cp -n $f $KD/kd_r9_${b#kd_}; done
NEW=$C/$D/data/score_new.jsonl
mkdir -p $C/$D/data
[ -s $DATA ] || python3 -c "
import json
with open('$DATA', 'w') as f, open('$NEW', 'w') as g:
    for l in open('$C/runs/bestofn/out/verified.jsonl'):
        r = json.loads(l)
        if [m['role'] for m in r['messages']] == ['user', 'assistant'] and not r.get('tools'):
            f.write(l)
            if not r.get('bestofn'):
                g.write(l)  # old-teacher answers: still need the teacher's distributions
"
for s in 42 43 44; do cp -n $DATA $D/train/seed$s/verified.jsonl; done  # the seed loop above ran before $DATA existed
rows=$(wc -l < $DATA)
scored() { $VPY -c "
import glob, numpy as np
print(sum(len(np.load(f)['row_ids']) for f in glob.glob('$KD/kd_*.npz')))" 2>/dev/null || echo 0; }
log "teacher scoring: $rows rows, $(scored) done"
for attempt in 1 2 3 4 5 6 7 8; do
    [ "$(scored)" -ge $(( rows * 99 / 100 )) ] && break  # rows longer than 4,096 tokens are skipped
    log "scoring attempt $attempt ($(scored)/$rows)"
    REPO=$C KD_DIR=$KD DATA=$NEW bash $R/ctl/priority_submit.sh --nodes=6 research/kd/score.sbatch
    grep -h -A8 Traceback $R/logs/kd-score-*.err 2>/dev/null | tail -10 >> "$HERE/controller.log"
done
[ "$(scored)" -ge $(( rows * 99 / 100 )) ] || { log "STOP: only $(scored)/$rows rows scored"; exit 1; }
log "SCORED: $(scored)/$rows rows, $(du -sh $KD | cut -f1)"

fails=0
for round in $(seq 1 90); do
    done_n=$(ls $D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)
    [ "$done_n" = 3 ] && break
    j=$(adopt dk-rag65kd3-train)
    if [ -z "$j" ]; then
        last=$(for i in 0 1 2; do [ -f $D/train/seed$(( 42 + i ))/TRAINING_DONE.json ] || echo $i; done | tail -1)
        [[ "$last" =~ ^[0-2]$ ]] || last=2
        echo $(( last + 1 )) > "$HERE/NODES_NEEDED"
        j=$(submit --nodes=$(( last + 1 )) --job-name=dk-rag65kd3-train research/kd/train.sbatch)
        [[ "$j" =~ ^[0-9]+$ ]] || { log "train submission refused (QoS slots busy); retrying"; sleep 120; continue; }
        log "train round $round: job $j ($done_n/3 seeds done, $(( last + 1 )) node(s))"
    fi
    wait_job "$j"
    state=$(sacct -n -X -j "$j" --format=State 2>/dev/null | head -1 | xargs)
    tail -2 $R/logs/train-$j-0.out 2>/dev/null | cut -c1-300 >> "$HERE/controller.log"
    if [ "$state" != TIMEOUT ] && grep -l Traceback $R/logs/train-$j-*.err > /dev/null 2>&1; then
        grep -h -B2 -A12 Traceback $R/logs/train-$j-*.err | tail -30 >> "$HERE/controller.log"
        log "STOP: KD training raised an exception (see above)"; exit 1
    fi
    case "$state" in TIMEOUT|COMPLETED*) fails=0 ;; RUNNING*|PENDING*|COMPLETING*|'') ;; *) fails=$(( fails + 1 )) ;; esac
    [ $fails -ge 3 ] && { log "STOP: 3 consecutive KD training jobs failed ($state)"; exit 1; }
done
[ "$(ls $D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)" = 3 ] || { log "STOP: KD training unfinished"; exit 1; }
log "scaling adapters"
export PYTHONPATH=src:research/rag65
for s in 42 43 44; do for x in 0.5 0.2; do $VPY research/rag65/make_scaled.py $D/train/seed$s/adapter $x; done; done

log "trained; answering"
mkdir -p $D/answers
cp -n runs/rag65v5/answers/*__base_*.jsonl $D/answers/
for f in runs/rag65v5/answers/*__s4?_*.jsonl; do b=$(basename $f); cp -n $f $D/answers/${b/__s/__r5s}; done
for f in runs/rag65kd/answers/*__s4?_*.jsonl; do b=$(basename $f); cp -n $f $D/answers/${b/__s/__k1s}; done
missing() { $VPY - <<'PY' | tail -1
import json
from pathlib import Path
need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for s in (42, 43, 44):
        for v in ['', '_x05', '_x02']:
            for mode in ('closed', 'rag'):
                f = Path(f'runs/rag65kd3/answers/{label}__s{s}{v}_{mode}.jsonl')
                ids = set()
                for x in (f.read_text().splitlines() if f.exists() else []):
                    try:
                        ids.add(json.loads(x)['id'])
                    except (ValueError, KeyError):
                        pass
                need += max(0, n - len(ids))
print(need)
PY
}
for attempt in 1 2 3 4 5 6; do
    [ "$(missing)" = 0 ] && break
    j=$(adopt dk-rag65kd3); [ -n "$j" ] || j=$(submit --nodes=2 --job-name=dk-rag65kd3 --export=ALL,STAGE=answer research/rag65/stage.sbatch)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "answer submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "answer job $j ($(missing) answers missing)"; wait_job "$j"
done
[ "$(missing)" = 0 ] || { log "STOP: $(missing) answers missing"; exit 1; }
left() { $VPY - <<'PY' | tail -1
from pathlib import Path
a = Path('runs/rag65kd3/answers'); j = Path('runs/rag65kd3/judged')
need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
for attempt in 1 2 3 4 5 6 7 8; do
    [ "$(left)" = 0 ] && break
    j=$(adopt dk-rag65kd3); [ -n "$j" ] || j=$(submit --nodes=4 --job-name=dk-rag65kd3 --export=ALL,STAGE=judge research/rag65/stage.sbatch)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "judge submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "judge job $j ($(left) units left)"; wait_job "$j"
done
[ "$(left)" = 0 ] || { log "STOP: $(left) judge units left"; exit 1; }
$VPY research/rag65/aggregate.py $D/judged $D/answers $D/RESULT.json > "$HERE/aggregate.out" 2>&1
pairs=""
for x in 42 43 44; do pairs="$pairs s${x}_rag:k1s${x}_rag s${x}_rag:r5s${x}_x02_rag s${x}_x05_rag:k1s${x}_rag"; done
$VPY research/rag65/paired.py $D/judged $D/PAIRED.json $pairs > "$HERE/paired.out" 2>&1
log "DONE: $(tr -d '\n' < "$HERE/aggregate.out" | head -c 1500)"
