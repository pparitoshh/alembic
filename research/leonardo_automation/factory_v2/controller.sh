#!/bin/bash
# Top-up batch (ai_factory_all_v2): generate/review its 100 shards in at most ONE debug job slot, so batch 1
# and the round-4 controller always keep the other; yields entirely while a round-4 job is pending.
# Merges after batch 1's release exists, with batch 1 as an existing bank so cross-batch repeats drop.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
HERE="$(cd "$(dirname "$0")" && pwd)"
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
B7=/leonardo_work/EUHPC_D30_031/runs/a08trc09/student-mixed-v2-20261015T042338Z/bundle/data/b7_original.jsonl
BATCH=ai_factory_all_v2
SHARDS=100
MAX_ATTEMPTS=5
remote() { bash -c "$*" 2>&1; }
heartbeat() { touch "$HERE/HEARTBEAT"; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
mkdir -p "$HERE/inflight" "$HERE/attempts"

until test -f $C/runs/$BATCH/summary.json; do heartbeat; sleep 60; done
log "prepared: $(tr -d '\n ' < $C/runs/$BATCH/summary.json | head -c 700)"

while true; do
    heartbeat
    # A failed squeue must not look like "no jobs": skip the round. A job is forgotten only once sacct
    # reports a final state (a transient squeue miss once released shards that were still under review).
    running=$(squeue --me -h -o %i 2>/dev/null) || { sleep 60; continue; }
    for f in "$HERE"/inflight/*; do
        [ -e "$f" ] || continue
        j=$(basename "$f")
        if ! grep -qw "$j" <<< "$running"; then
            st=$(sacct -n -X -j "$j" --format=State,Elapsed 2>/dev/null | head -1 | xargs)
            case "$st" in ''|RUNNING*|PENDING*|REQUEUED*|CONFIGURING*|COMPLETING*) continue ;; esac
            log "job $j: $st"
            rm -f "$f"
        fi
    done
    state=$(cd $C/runs/$BATCH && for k in $(seq -f %03g 0 $((SHARDS - 1))); do
        if [ -f shard_$k/clean.jsonl ]; then echo "c $k"; elif [ -f shard_$k/GENERATE_DONE ]; then echo "r $k"; else echo "g $k"; fi; done)
    clean=$(grep -c '^c ' <<< "$state")
    busy=$(cat "$HERE"/inflight/* 2>/dev/null | tr '\n' ' ')
    pick() {
        grep "^$1 " <<< "$state" | cut -d' ' -f2 | while read -r k; do
            grep -qw "$k" <<< "$busy" && continue
            [ "$(cat "$HERE/attempts/$1$k" 2>/dev/null || echo 0)" -ge $MAX_ATTEMPTS ] && continue
            echo "$k"
        done
    }
    gen=$(pick g); rev=$(pick r)
    if [ -z "$gen$rev" ] && [ -z "$(ls "$HERE"/inflight)" ]; then
        log "all shards finished or exhausted: $clean/$SHARDS clean"; break
    fi
    jobs=$(remote "squeue --me -h -q boost_qos_dbg -o '%i %D'")
    njobs=$(grep -c . <<< "$jobs")
    used=$(awk '{s += $2} END {print s + 0}' <<< "$jobs")
    rag_pending=$(remote "squeue --me -h -t PD -o %j" | grep -c 'rag65')
    # At most 5 nodes: the round controller's 3-node training job must always fit in the 8-node QoS limit,
    # including in the gap between two training pieces.
    nodes=$(( 8 - used )); [ $nodes -gt 5 ] && nodes=5
    # While batch 1 is unmerged, take only the second slot: never the one batch 1 (and round 4) needs.
    v1_done=0; grep -q ' MERGED: ' "$HERE/../factory_v1/controller.log" && v1_done=1
    v1_jobs=$(ls "$HERE/../factory_v1/inflight" 2>/dev/null | wc -l)
    # ctl/PRIORITY (ctl/priority_submit.sh) reserves the next free debug slot for a one-off job.
    if [ ! -e "$HERE/../PRIORITY" ] && [ -z "$(ls "$HERE"/inflight)" ] && [ "$njobs" -lt 2 ] && [ "$rag_pending" -eq 0 ] && [ $nodes -ge 2 ] \
            && { [ $v1_done = 1 ] || [ "$v1_jobs" -ge 1 ]; }; then
        if [ "$(grep -c . <<< "$rev")" -ge $(( nodes * 4 )) ] || [ -z "$gen" ]; then phase=review; list=$rev; else phase=generate; list=$gen; fi
        list=$(head -n $(( nodes * 4 )) <<< "$list" | grep . | tr '\n' ' ')
        n=$(wc -w <<< "$list")
        if [ "$n" -gt 0 ]; then
            nodes=$(( (n + 3) / 4 ))
            j=$(remote "cd $C && export REPO=$C BATCH=$BATCH SHARD_LIST='$list' && sbatch --parsable --nodes=$nodes --export=ALL,PHASE=$phase,REVIEW_TP=2 research/ai_factory_pilot/factory_dbg.sbatch" | tail -1)
            if [[ "$j" =~ ^[0-9]+$ ]]; then
                echo "$list" > "$HERE/inflight/$j"
                for k in $list; do p=${phase:0:1}; echo $(( $(cat "$HERE/attempts/$p$k" 2>/dev/null || echo 0) + 1 )) > "$HERE/attempts/$p$k"; done
                log "$phase job $j: $n shards on $nodes nodes (clean $clean/$SHARDS)"
            else
                log "submit refused: $j"
            fi
        fi
    fi
    sleep 45  # short gaps between 30-min jobs leave GPUs idle
done

until test -f $C/runs/ai_factory_all_v1/release/verified.jsonl; do heartbeat; sleep 120; done
start_merge=$(date +%s)
remote "cd $C && export BATCH=$BATCH && PYTHONPATH=src HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 srun -A EUHPC_D30_031 -p lrd_all_serial -c 4 --mem=30G -t 01:30:00 $VPY research/ai_factory_pilot/merge_factory_v1.py $SHARDS $B7 runs/ai_factory_ccby_v1/release/verified.jsonl runs/ai_factory_ccby_v2/release/verified.jsonl runs/ai_factory_all_v1/release/verified.jsonl" > "$HERE/merge.out"
# MERGED only for a release written by this merge: later stages start from it.
rel=$C/runs/$BATCH/release/verified.jsonl
if [ ! -s "$rel" ] || [ "$(stat -c %Y "$rel")" -lt "$start_merge" ]; then
    log "STOP: merge did not produce a new release: $(tail -5 "$HERE/merge.out" | tr '\n' ' ')"; exit 1
fi
cp $C/runs/$BATCH/release/REPORT.json "$HERE/REPORT.json"
log "MERGED: $(tr -d '\n' < "$HERE/REPORT.json" | head -c 900)"
