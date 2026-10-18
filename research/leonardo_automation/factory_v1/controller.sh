#!/bin/bash
# All-factory batch: keep debug-QoS GPUs busy with generate/review jobs over 200 resumable shards, then merge.
# While the round-3 controller runs it owns one of the two debug job slots; after that both are ours.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
HERE="$(cd "$(dirname "$0")" && pwd)"
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
B7=/leonardo_work/EUHPC_D30_031/runs/a08trc09/student-mixed-v2-20261015T042338Z/bundle/data/b7_original.jsonl
SHARDS=200
MAX_ATTEMPTS=5
remote() { bash -c "$*" 2>&1; }  # runs on the Leonardo login node itself
heartbeat() { touch "$HERE/HEARTBEAT"; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
mkdir -p "$HERE/inflight" "$HERE/attempts"

until remote "test -f $C/runs/ai_factory_all_v1/summary.json && echo ok" | grep -q ok; do sleep 60; done
log "prepared: $(remote "tr -d '\n ' < $C/runs/ai_factory_all_v1/summary.json" | head -c 900)"

while true; do
    # Forget finished jobs; their shards become eligible again if still pending.
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
    state=$(remote "cd $C/runs/ai_factory_all_v1 && for k in \$(seq -f %03g 0 $((SHARDS - 1))); do
        if [ -f shard_\$k/clean.jsonl ]; then echo \"c \$k\"; elif [ -f shard_\$k/GENERATE_DONE ]; then echo \"r \$k\"; else echo \"g \$k\"; fi; done")
    clean=$(grep -c '^c ' <<< "$state")
    busy=$(cat "$HERE"/inflight/* 2>/dev/null | tr '\n' ' ')
    pick() {  # pending shards for a phase, not in flight, under the attempt cap
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
    max_jobs=2
    heartbeat
    # The round-3 controller owns one slot while its heartbeat is fresh.
    [ -n "$(find "$HERE/../rag65v3/HEARTBEAT" -mmin -10 2>/dev/null)" ] && max_jobs=1 && njobs=$(ls "$HERE"/inflight | wc -l)
    nodes=$(( 8 - used )); [ $nodes -gt 8 ] && nodes=8
    if [ "$njobs" -lt "$max_jobs" ] && [ $nodes -ge 2 ]; then
        # Review whenever a full job's worth is waiting (or nothing is left to generate).
        if [ "$(grep -c . <<< "$rev")" -ge $(( nodes * 4 )) ] || [ -z "$gen" ]; then phase=review; list=$rev; else phase=generate; list=$gen; fi
        list=$(head -n $(( nodes * 4 )) <<< "$list" | grep . | tr '\n' ' ')
        n=$(wc -w <<< "$list")
        if [ "$n" -gt 0 ]; then
            nodes=$(( (n + 3) / 4 ))
            j=$(remote "cd $C && export REPO=$C SHARD_LIST='$list' && sbatch --parsable --nodes=$nodes --export=ALL,PHASE=$phase research/ai_factory_pilot/factory_dbg.sbatch" | tail -1)
            if [[ "$j" =~ ^[0-9]+$ ]]; then
                echo "$list" > "$HERE/inflight/$j"
                for k in $list; do p=${phase:0:1}; echo $(( $(cat "$HERE/attempts/$p$k" 2>/dev/null || echo 0) + 1 )) > "$HERE/attempts/$p$k"; done
                log "$phase job $j: $n shards on $nodes nodes (clean $clean/$SHARDS)"
            else
                log "submit refused: $j"
            fi
        fi
    fi
    sleep 120
done

remote "cd $C && PYTHONPATH=src HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 srun -A EUHPC_D30_031 -p boost_usr_prod -q boost_qos_dbg -N1 -n1 -c32 --gres=gpu:1 --mem=200G -t 00:30:00 $VPY research/ai_factory_pilot/merge_factory_v1.py $SHARDS $B7 runs/ai_factory_ccby_v1/release/verified.jsonl runs/ai_factory_ccby_v2/release/verified.jsonl" > "$HERE/merge.out"
remote "cat $C/runs/ai_factory_all_v1/release/REPORT.json" > "$HERE/REPORT.json"
log "MERGED: $(tr -d '\n' < "$HERE/REPORT.json" | head -c 900)"
[ -x "$HERE/next.sh" ] && { log "starting next stage"; exec "$HERE/next.sh"; }
