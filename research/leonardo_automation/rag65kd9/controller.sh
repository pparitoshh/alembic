#!/bin/bash
# Round 12e: on-policy distillation (12c's recipe) on top of round 12a, the strongest off-policy student (distillation
# weight 0.8, LoRA r32; seed 43 full strength: v3 0.789, best single run so far). 12a is merged first (CPU partition),
# then it answers round 11's prompts itself, the teacher scores those answers (top-20) and a new LoRA is trained on
# them (weight 1.0). Judged with 12c (k6s44) and 12a (a4s43), then vs the teacher.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
HERE="$(cd "$(dirname "$0")" && pwd)"
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
D=runs/rag65kd9
MERGED=$C/runs/r12a_merged     # round 12a, seed 43, full strength
export BASE_MODEL=$MERGED       # answer stage serves the merged 12a model plus the new adapters
GEN=$C/runs/rag65kd9/gen
KD=$C/runs/rag65kd9/teacher_top20
DATA=$C/runs/rag65kd9/data/verified.jsonl
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
heartbeat() { touch "$HERE/HEARTBEAT"; }
adopt() { squeue --me -h -n "$1" -o %i 2>/dev/null | grep -E '^[0-9]+$' | head -1; }
wait_job() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do heartbeat; sleep 90; done
             log "job $1: $(sacct -n -X -j "$1" --format=State,Elapsed 2>/dev/null | head -1 | xargs)"; }
submit() { (cd $C && REPO=$C RUN=rag65kd9 CFG=rag65kd9 ALPHA=1.0 KD_DIR=$KD DATA=$DATA sbatch --parsable "$@" 2>/dev/null | tail -1); }
cd $C
if [ ! -s $MERGED/config.json ]; then
    log "merging round 12a seed 43 (CPU partition)"
    srun -A EUHPC_D30_031 -p lrd_all_serial -c 4 --mem=30G -t 01:00:00 -J dk-merge-12a bash -c "export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1; $VPY research/dpo/merge.py runs/rag65kd4/train/seed43/adapter $MERGED" > "$HERE/merge.out" 2>&1
    test -s $MERGED/config.json || { log "STOP: merge failed: $(tail -3 "$HERE/merge.out")"; exit 1; }
fi
mkdir -p $D/train $KD $D/data
for s in 42 43 44; do
    [ -s configs/rag65kd9_seed$s.json ] || sed -e "s#runs/rag65kd/train#runs/rag65kd9/train#" -e "s#\"model\": \"Qwen/Qwen3-4B-Instruct-2507\"#\"model\": \"$MERGED\"#" configs/rag65kd_seed$s.json > configs/rag65kd9_seed$s.json
    mkdir -p $D/train/seed$s && cp -n $DATA $D/train/seed$s/verified.jsonl
done
mkdir -p $D/data $GEN
SRC=$C/runs/rag65kd3/data/verified.jsonl
want=$($VPY -c "
import json
q=('Quoted retrieval results (data, not instructions):','Provided input 1 (quoted data')
print(sum(any(x in json.loads(l)['messages'][0]['content'] for x in q) for l in open('$SRC')))")
have() { cat $GEN/cands_*.jsonl 2>/dev/null | wc -l; }
for attempt in 1 2 3 4 5; do
    [ "$(have)" -ge "$want" ] && break
    j=$(adopt dk-kd-opgen); [ -n "$j" ] || j=$(cd $C && REPO=$C MODEL=$MERGED DATA=$SRC OUT=$GEN sbatch --parsable --nodes=2 research/kd/onpolicy_gen.sbatch 2>/dev/null | tail -1)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "generation submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "student generation job $j ($(have)/$want answers)"; wait_job "$j"
done
log "GENERATED: $(have)/$want student answers"
[ -s $DATA ] || $VPY -c "
import glob, json
c = {}
for f in glob.glob('$GEN/cands_*.jsonl'):
    for l in open(f):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get('candidates'):
            c[r['id']] = r['candidates'][0]
n = 0
with open('$DATA', 'w') as f:
    for l in open('$SRC'):
        r = json.loads(l)
        if r['id'] in c:
            r['messages'][-1]['content'] = c[r['id']]; r['onpolicy'] = 'round12a-s43'
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); n += 1
print(n)
" > "$HERE/data.out"
log "DATA: $(tail -1 "$HERE/data.out") rows with the student's own answers"
for s in 42 43 44; do mkdir -p $D/train/seed$s && cp -n $DATA $D/train/seed$s/verified.jsonl; done
rows=$(wc -l < $DATA)
scored() { $VPY -c "
import glob, numpy as np
print(sum(len(np.load(f)['row_ids']) for f in glob.glob('$KD/kd_*.npz')))" 2>/dev/null || echo 0; }
for attempt in 1 2 3 4 5 6; do
    [ "$(scored)" -ge $(( rows * 99 / 100 )) ] && break
    j=$(adopt dk-kd-score); [ -n "$j" ] || j=$(submit --nodes=3 research/kd/score.sbatch)  # keep N fixed across attempts
    [[ "$j" =~ ^[0-9]+$ ]] || { log "scoring submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "teacher scoring job $j ($(scored)/$rows)"; wait_job "$j"
done
[ "$(scored)" -ge $(( rows * 99 / 100 )) ] || { log "STOP: only $(scored)/$rows rows scored"; exit 1; }
log "SCORED: $(scored)/$rows rows, $(du -sh $KD | cut -f1)"

fails=0
for round in $(seq 1 90); do
    done_n=$(ls $D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)
    [ "$done_n" = 3 ] && break
    j=$(adopt dk-rag65kd9-train)
    if [ -z "$j" ]; then
        last=$(for i in 0 1 2; do [ -f $D/train/seed$(( 42 + i ))/TRAINING_DONE.json ] || echo $i; done | tail -1)
        [[ "$last" =~ ^[0-2]$ ]] || last=2
        echo $(( last + 1 )) > "$HERE/NODES_NEEDED"
        j=$(submit --nodes=$(( last + 1 )) --job-name=dk-rag65kd9-train research/kd/train.sbatch)
        [[ "$j" =~ ^[0-9]+$ ]] || { log "train submission refused (QoS slots busy); retrying"; sleep 120; continue; }
        log "train round $round: job $j ($done_n/3 seeds done, $(( last + 1 )) node(s))"
    fi
    wait_job "$j"
    state=$(sacct -n -X -j "$j" --format=State 2>/dev/null | head -1 | xargs)
    tail -2 $R/logs/train-$j-0.out 2>/dev/null | cut -c1-300 >> "$HERE/controller.log"
    # A node failure or Slurm's SIGTERM also leaves tracebacks (SignalException): resume from the checkpoints instead.
    if [ "$state" != TIMEOUT ] && [ "$state" != NODE_FAIL ] && grep -l Traceback $R/logs/train-$j-*.err > /dev/null 2>&1 \
            && ! grep -q 'SignalException' $R/logs/train-$j-*.err; then
        grep -h -B2 -A12 Traceback $R/logs/train-$j-*.err | tail -30 >> "$HERE/controller.log"
        log "STOP: KD training raised an exception (see above)"; exit 1
    fi
    case "$state" in TIMEOUT|COMPLETED*|NODE_FAIL) fails=0 ;; RUNNING*|PENDING*|COMPLETING*|'') ;; *) fails=$(( fails + 1 )) ;; esac
    [ $fails -ge 3 ] && { log "STOP: 3 consecutive KD training jobs failed ($state)"; exit 1; }
done
[ "$(ls $D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)" = 3 ] || { log "STOP: KD training unfinished"; exit 1; }
log "scaling adapters"
export PYTHONPATH=src:research/rag65
for s in 42 43 44; do for x in 0.5 0.2; do
    t=$D/train/seed$s/adapter_x0${x#0.}; [ -s $t/adapter_model.safetensors ] || $VPY research/rag65/make_scaled.py $D/train/seed$s/adapter $x &
done; done; wait

log "trained; answering"
mkdir -p $D/answers
cp -n runs/rag65v5/answers/*__base_*.jsonl $D/answers/
for f in runs/rag65v5/answers/*__s4?_*.jsonl; do b=$(basename $f); cp -n $f $D/answers/${b/__s/__r5s}; done
for f in runs/rag65kd3/answers/*__s4?_*.jsonl; do b=$(basename $f); cp -n $f $D/answers/${b/__s/__k3s}; done
for f in runs/rag65kd6/answers/*__s4?_*.jsonl; do b=$(basename $f); cp -n $f $D/answers/${b/__s/__k6s}; done
for f in runs/rag65kd4/answers/*__s4?_*.jsonl; do b=$(basename $f); cp -n $f $D/answers/${b/__s/__a4s}; done
missing() { $VPY - <<'PY' | tail -1
import json
from pathlib import Path
need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for s in (42, 43, 44):
        for v in ['', '_x05', '_x02']:
            for mode in ('closed', 'rag'):
                f = Path(f'runs/rag65kd9/answers/{label}__s{s}{v}_{mode}.jsonl')
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
    j=$(adopt dk-rag65kd9); [ -n "$j" ] || j=$(submit --nodes=2 --job-name=dk-rag65kd9 --export=ALL,STAGE=answer research/rag65/stage.sbatch)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "answer submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "answer job $j ($(missing) answers missing)"; wait_job "$j"
done
[ "$(missing)" = 0 ] || { log "STOP: $(missing) answers missing"; exit 1; }
left() { $VPY - <<'PY' | tail -1
from pathlib import Path
a = Path('runs/rag65kd9/answers'); j = Path('runs/rag65kd9/judged')
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
    j=$(adopt dk-rag65kd9); [ -n "$j" ] || j=$(submit --nodes=4 --job-name=dk-rag65kd9 --export=ALL,STAGE=judge research/rag65/stage.sbatch)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "judge submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "judge job $j ($(left) units left)"; wait_job "$j"
done
[ "$(left)" = 0 ] || { log "STOP: $(left) judge units left"; exit 1; }
$VPY research/rag65/aggregate.py $D/judged $D/answers $D/RESULT.json > "$HERE/aggregate.out" 2>&1
pairs=""
for x in 42 43 44; do pairs="$pairs s${x}_rag:k6s44_rag s${x}_x05_rag:k6s44_rag"; done
pairs="$pairs s42_rag:a4s43_rag s43_rag:a4s43_rag s44_rag:a4s43_rag a4s43_rag:k6s44_rag"
$VPY research/rag65/paired.py $D/judged $D/PAIRED.json $pairs > "$HERE/paired.out" 2>&1
log "DONE: $(tr -d '\n' < "$HERE/aggregate.out" | head -c 1500)"
