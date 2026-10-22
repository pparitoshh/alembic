#!/bin/bash
# Shraddha's account: one distillation round from Qwen3-235B teacher scores (round 21's data, read-only from
# a08trc09's runs), trained on the current release (round 20, merged), then its evaluation answers. Judging and any
# release happen on a08trc09's side, which watches ctl/<round>/controller.log for the DONE line.
# usage: run_round.sh sh24|sh25        (run on a Leonardo login node with setsid nohup; resumable: rerun to continue)
#   sh24: 12a's recipe   (distillation weight 0.8, LoRA r32/a64)
#   sh25: round-14 style (distillation weight 1.0, LoRA r64/a128)
set -uo pipefail
RD=${1:?round name: sh24 or sh25}
S=/leonardo_work/EUHPC_D30_031/shared_split
A=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017   # a08trc09's pipeline (read-only for us)
C=$S/code; HERE=$S/ctl/$RD; mkdir -p $HERE
VPY=/leonardo_work/EUHPC_D30_031/shared_split/venv-train/bin/python
MERGED=$A/code/runs/gguf_op235/merged          # round 20, the current Hugging Face release
export BASE_MODEL=$MERGED
DATA_SRC=$A/code/runs/kd235/data/verified.jsonl   # Qwen3-235B answers (round 21)
KD=$A/code/runs/kd235/teacher_top20               # Qwen3-235B top-20 scores (round 21)
case "$RD" in
    sh24) ALPHA=0.8; LR=32; LA=64 ;;
    sh25) ALPHA=1.0; LR=64; LA=128 ;;
    *) echo "unknown round $RD"; exit 1 ;;
esac
D=runs/$RD; DATA=$C/$D/data/verified.jsonl
export VLLM_CACHE_ROOT=$S/cache/$USER/vllm TRITON_CACHE_DIR=$S/cache/$USER/triton   # home quota is only 50 GB
mkdir -p $VLLM_CACHE_ROOT $TRITON_CACHE_DIR
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
heartbeat() { touch "$HERE/HEARTBEAT"; }
adopt() { squeue --me -h -n "$1" -o %i 2>/dev/null | grep -E '^[0-9]+$' | head -1; }
wait_job() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do heartbeat; sleep 90; done
             log "job $1: $(sacct -n -X -j "$1" --format=State,Elapsed 2>/dev/null | head -1 | xargs)"; }
submit() { (cd $C && REPO=$C RUN=$RD CFG=$RD ALPHA=$ALPHA KD_DIR=$KD DATA=$DATA sbatch --parsable "$@" 2>/dev/null | tail -1); }
cd $C
test -s $MERGED/config.json || { log "STOP: cannot read the round 20 model at $MERGED (permissions?)"; exit 1; }
log "start ($RD: weight $ALPHA, LoRA r$LR/a$LA, on round 20)"
log "waiting for round 21's Qwen3-235B scores (a08trc09's ctl/kd235/controller.log: SCORED)"
until grep -q ' SCORED: ' $A/ctl/kd235/controller.log 2>/dev/null; do
    grep -q ' STOP: ' $A/ctl/kd235/controller.log 2>/dev/null && { log "STOP: round 21 stopped before its scores were ready"; exit 1; }
    heartbeat; sleep 120
done
mkdir -p $D/train $D/data
[ -s $DATA ] || cp $DATA_SRC $DATA
for s in 42 43 44; do
    [ -s configs/${RD}_seed$s.json ] || sed -e "s#runs/rag65kd/train#$D/train#" -e "s#\"model\": \"Qwen/Qwen3-4B-Instruct-2507\"#\"model\": \"$MERGED\"#" -e "s#\"lora_r\": 16#\"lora_r\": $LR#" -e "s#\"lora_alpha\": 32#\"lora_alpha\": $LA#" configs/rag65kd_seed$s.json > configs/${RD}_seed$s.json
    mkdir -p $D/train/seed$s && cp -n $DATA $D/train/seed$s/verified.jsonl
done
grep -q "\"lora_r\": $LR" configs/${RD}_seed42.json && grep -q "$MERGED" configs/${RD}_seed42.json || { log "STOP: config not as intended"; exit 1; }
fails=0
for round in $(seq 1 90); do
    done_n=$(ls $D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)
    [ "$done_n" = 3 ] && break
    j=$(adopt dk-$RD-train)
    if [ -z "$j" ]; then
        last=$(for i in 0 1 2; do [ -f $D/train/seed$(( 42 + i ))/TRAINING_DONE.json ] || echo $i; done | tail -1)
        [[ "$last" =~ ^[0-2]$ ]] || last=2
        j=$(submit --nodes=$(( last + 1 )) --job-name=dk-$RD-train research/kd/train.sbatch)
        [[ "$j" =~ ^[0-9]+$ ]] || { log "train submission refused (QoS slots busy); retrying"; sleep 120; continue; }
        log "train round $round: job $j ($done_n/3 seeds done, $(( last + 1 )) node(s))"
    fi
    wait_job "$j"
    state=$(sacct -n -X -j "$j" --format=State 2>/dev/null | head -1 | xargs)
    tail -2 $S/logs/train-$j-0.out 2>/dev/null | cut -c1-300 >> "$HERE/controller.log"
    if [ "$state" != TIMEOUT ] && [ "$state" != NODE_FAIL ] && grep -l Traceback $S/logs/train-$j-*.err > /dev/null 2>&1 \
            && ! grep -q 'SignalException' $S/logs/train-$j-*.err; then
        grep -h -B2 -A12 Traceback $S/logs/train-$j-*.err | tail -30 >> "$HERE/controller.log"
        log "STOP: training raised an exception (see above)"; exit 1
    fi
    case "$state" in TIMEOUT|COMPLETED*|NODE_FAIL) fails=0 ;; RUNNING*|PENDING*|COMPLETING*|'') ;; *) fails=$(( fails + 1 )) ;; esac
    [ $fails -ge 3 ] && { log "STOP: 3 consecutive training jobs failed ($state)"; exit 1; }
done
[ "$(ls $D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)" = 3 ] || { log "STOP: training unfinished"; exit 1; }
log "scaling adapters"
export PYTHONPATH=src:research/rag65
for s in 42 43 44; do t=$D/train/seed$s/adapter_x05; [ -s $t/adapter_model.safetensors ] || $VPY research/rag65/make_scaled.py $D/train/seed$s/adapter 0.5 & done; wait
log "trained; answering"
mkdir -p $D/answers
cp -n $A/code/runs/rag65v5/answers/*__base_closed.jsonl $A/code/runs/rag65v5/answers/*__base_rag.jsonl $D/answers/
missing() { $VPY - <<PY | tail -1
import json
from pathlib import Path
need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for s in (42, 43, 44):
        for v in ['', '_x05']:
            f = Path(f'$D/answers/{label}__s{s}{v}_rag.jsonl')
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
    j=$(adopt dk-$RD); [ -n "$j" ] || j=$(VARIANTS=,_x05 ANSWER_MODES=rag submit --nodes=1 --job-name=dk-$RD --export=ALL,STAGE=answer research/rag65/stage.sbatch)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "answer submission refused (QoS slots busy); retrying"; sleep 120; continue; }
    log "answer job $j ($(missing) answers missing)"; wait_job "$j"
done
[ "$(missing)" = 0 ] || { log "STOP: $(missing) answers missing"; exit 1; }
chmod -R g+rX $C/$D 2>/dev/null
log "DONE: answers ready in $C/$D/answers; a08trc09 judges them (3 judges + majority) and releases a winner"
