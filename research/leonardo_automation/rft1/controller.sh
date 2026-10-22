#!/bin/bash
# Round 16 (RFT-1): rejection sampling against the teacher, aimed at the head-to-head score. On 1 in 3 of the 18,974
# training questions that the teacher (Qwen3-Next) answered itself (runs/rag65v9), 12e (the Hugging Face release)
# samples 4 answers; the evaluation judge (gpt-oss-20b, both orders) compares each with the teacher's answer, using the
# question's source excerpt as the reference; for each question the best answer that beats the teacher (score >= 0.75)
# is kept, and 12e is fine-tuned on these winners (plain cross-entropy, LoRA r16). Judged against 12e and round 11,
# then head to head against the teacher.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
HERE="$(cd "$(dirname "$0")" && pwd)"
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
D=runs/rft1
MERGED=$C/runs/gguf_r12e/merged  # 12e (round 12a seed 43 + 12e seed 44), the current release
export BASE_MODEL=$MERGED         # answer stage serves the merged 12e model plus the new adapters
SUB=$C/runs/rft1/data/questions.jsonl
GEN=$C/runs/rft1/gen
JUDGED=$C/runs/rft1/judged_teacher  # not runs/rft1/judged, the eval judge folder (mixed files crashed aggregate.py, Oct 22)
KD=$C/runs/rft1                   # unused: plain fine-tuning (ALPHA 0)
DATA=$C/runs/rft1/data/verified.jsonl
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
heartbeat() { touch "$HERE/HEARTBEAT"; }
adopt() { squeue --me -h -n "$1" -o %i 2>/dev/null | grep -E '^[0-9]+$' | head -1; }
wait_job() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do heartbeat; sleep 90; done
             log "job $1: $(sacct -n -X -j "$1" --format=State,Elapsed 2>/dev/null | head -1 | xargs)"; }
submit() { (cd $C && REPO=$C RUN=rft1 CFG=rft1 ALPHA=0 KD_DIR=$KD DATA=$DATA sbatch --parsable "$@" 2>/dev/null | tail -1); }
cd $C
test -s $MERGED/config.json || { log "STOP: merged 12e model missing ($MERGED)"; exit 1; }
mkdir -p $D/train $D/data $GEN $JUDGED
[ -s $SUB ] || $VPY -c "
import json
with open('$SUB', 'w') as f:
    for i, l in enumerate(open('runs/rag65v9/data/verified.jsonl')):
        if i % 3 == 0:
            f.write(l)
"
want=$(wc -l < $SUB)
have() { cat $GEN/cands_*.jsonl 2>/dev/null | wc -l; }
log "questions: $want; sampling 4 answers each with 12e"
for attempt in 1 2 3 4 5; do
    [ "$(have)" -ge $(( want * 98 / 100 )) ] && break
    j=$(adopt dk-rft1-gen); [ -n "$j" ] || j=$(cd $C && REPO=$C MODEL=$MERGED DATA=$SUB OUT=$GEN K=4 sbatch --parsable --nodes=2 --job-name=dk-rft1-gen research/kd/onpolicy_gen.sbatch 2>/dev/null | tail -1)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "sampling submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "sampling job $j ($(have)/$want questions)"; wait_job "$j"
done
log "SAMPLED: $(have)/$want questions"
units_left() { $VPY -c "
import glob, json
ids = set()
for f in glob.glob('$GEN/cands_*.jsonl'):
    for l in open(f):
        try:
            c = json.loads(l)
        except ValueError:
            continue
        if c.get('candidates'):
            ids.add(c['id'])
n = (len(ids) + 24) // 25
print(n - len(glob.glob('$JUDGED/unit_*.jsonl')))" 2>/dev/null || echo 999; }
for attempt in 1 2 3 4 5 6; do
    [ "$(units_left)" -le 0 ] && break
    j=$(adopt dk-rft1-judge); [ -n "$j" ] || j=$(cd $C && REPO=$C DATA=$SUB CANDS=$GEN OUT=$JUDGED sbatch --parsable --nodes=4 --job-name=dk-rft1-judge research/rft/judge.sbatch 2>/dev/null | tail -1)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "judge submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "judge job $j ($(units_left) units left)"; wait_job "$j"
done
[ "$(units_left)" -le 0 ] || { log "STOP: $(units_left) judge units left"; exit 1; }
[ -s $DATA ] || $VPY research/rft/build.py $SUB $GEN $JUDGED $DATA 0.75 > "$HERE/build.out" 2>&1
if [ "$(wc -l < $DATA)" -lt 1500 ]; then
    log "only $(wc -l < $DATA) clear wins; keeping ties and better (>= 0.5)"
    $VPY research/rft/build.py $SUB $GEN $JUDGED $DATA 0.5 > "$HERE/build.out" 2>&1
fi
log "WINNERS: $(tr '\n' ' ' < "$HERE/build.out")"
[ "$(wc -l < $DATA)" -ge 500 ] || { log "STOP: too few winning answers to train on"; exit 1; }
for s in 42 43 44; do
    [ -s configs/rft1_seed$s.json ] || sed -e "s#runs/rag65kd/train#runs/rft1/train#" -e "s#\"model\": \"Qwen/Qwen3-4B-Instruct-2507\"#\"model\": \"$MERGED\"#" configs/rag65kd_seed$s.json > configs/rft1_seed$s.json
    mkdir -p $D/train/seed$s && cp -n $DATA $D/train/seed$s/verified.jsonl
done

fails=0
for round in $(seq 1 90); do
    done_n=$(ls $D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)
    [ "$done_n" = 3 ] && break
    j=$(adopt dk-rft1-train)
    if [ -z "$j" ]; then
        last=$(for i in 0 1 2; do [ -f $D/train/seed$(( 42 + i ))/TRAINING_DONE.json ] || echo $i; done | tail -1)
        [[ "$last" =~ ^[0-2]$ ]] || last=2
        echo $(( last + 1 )) > "$HERE/NODES_NEEDED"
        j=$(submit --nodes=$(( last + 1 )) --job-name=dk-rft1-train research/kd/train.sbatch)
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
for f in runs/rag65kd9/answers/*__s44_rag.jsonl; do b=$(basename $f); cp -n $f $D/answers/${b/__s/__e9s}; done

missing() { $VPY - <<'PY' | tail -1
import json
from pathlib import Path
need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for s in (42, 43, 44):
        for v in ['', '_x05', '_x02']:
            for mode in ('closed', 'rag'):
                f = Path(f'runs/rft1/answers/{label}__s{s}{v}_{mode}.jsonl')
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
    j=$(adopt dk-rft1); [ -n "$j" ] || j=$(submit --nodes=2 --job-name=dk-rft1 --export=ALL,STAGE=answer research/rag65/stage.sbatch)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "answer submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "answer job $j ($(missing) answers missing)"; wait_job "$j"
done
[ "$(missing)" = 0 ] || { log "STOP: $(missing) answers missing"; exit 1; }
left() { $VPY - <<'PY' | tail -1
from pathlib import Path
a = Path('runs/rft1/answers'); j = Path('runs/rft1/judged')
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
    j=$(adopt dk-rft1); [ -n "$j" ] || j=$(submit --nodes=4 --job-name=dk-rft1 --export=ALL,STAGE=judge research/rag65/stage.sbatch)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "judge submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "judge job $j ($(left) units left)"; wait_job "$j"
done
[ "$(left)" = 0 ] || { log "STOP: $(left) judge units left"; exit 1; }
$VPY research/rag65/aggregate.py $D/judged $D/answers $D/RESULT.json > "$HERE/aggregate.out" 2>&1
pairs=""
for x in 42 43 44; do pairs="$pairs s${x}_rag:e9s44_rag s${x}_x05_rag:e9s44_rag s${x}_rag:k3s44_x05_rag"; done
$VPY research/rag65/paired.py $D/judged $D/PAIRED.json $pairs > "$HERE/paired.out" 2>&1
log "DONE: $(tr -d '\n' < "$HERE/aggregate.out" | head -c 1500)"
