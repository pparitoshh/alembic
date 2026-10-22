#!/bin/bash
# General-HPC batch (ai_factory_all_v5, research/ai_factory_pilot/prepare_general_v5.py): the 2026-10-19 general HPC
# documentation archive, prose and ask-back only, chunks never used by v1-v4. Clone of the factory_v4 controller:
# same GEN_TP=4 / REVIEW_TP=2 jobs, node caps and yield rules; v1-v4 are merged, so no batch-dependency checks.
# The merge uses every earlier release (CC-BY v1-v2, all-factory v1-v4) as a bank, so no question repeats, then
# annotate_general_v5.py adds topic / licence class and keeps 'other_needs_review' licences out of the permissive file.
# Installed as controller.sh.NEW so the watchdog does not start it; rename to controller.sh to activate.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
HERE="$(cd "$(dirname "$0")" && pwd)"
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
B7=/leonardo_work/EUHPC_D30_031/runs/a08trc09/student-mixed-v2-20261015T042338Z/bundle/data/b7_original.jsonl
BATCH=ai_factory_all_v5
SHARDS=$(ls -d $C/runs/$BATCH/shard_* 2>/dev/null | wc -l)
MAX_ATTEMPTS=10  # shards hold at most 450 questions (large documents are split): ~2 review jobs each
remote() { bash -c "$*" 2>&1; }
heartbeat() { touch "$HERE/HEARTBEAT"; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
mkdir -p "$HERE/inflight" "$HERE/attempts"

until test -f $C/runs/$BATCH/summary.json; do heartbeat; sleep 60; done
SHARDS=$(ls -d $C/runs/$BATCH/shard_* | wc -l)
log "prepared ($SHARDS shards): $(tr -d '\n ' < $C/runs/$BATCH/summary.json | head -c 700)"

frozen_logged=0
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
    # ctl/FREEZE_DATA (user decision Oct 19): no new data jobs while it exists. Unlike v3, do not merge early:
    # wait until it is removed (or the controller is stopped).
    if [ -e "$HERE/../FREEZE_DATA" ]; then
        [ $frozen_logged = 0 ] && log "data frozen (ctl/FREEZE_DATA): no new jobs until it is removed"
        frozen_logged=1; sleep 120; continue
    fi
    frozen_logged=0
    jobs=$(remote "squeue --me -h -q boost_qos_dbg -o '%i %D'")
    njobs=$(grep -c . <<< "$jobs")
    used=$(awk '{s += $2} END {print s + 0}' <<< "$jobs")
    rag_pending=$(remote "squeue --me -h -t PD -o %j" | grep -cE 'rag65|dk-kd')  # score jobs (rounds, KD scoring/generation) waiting
    # At most 5 nodes: the round controller's 3-node training job must always fit in the 8-node QoS limit,
    # including in the gap between two training pieces.
    nodes=$(( 8 - used )); [ $nodes -gt 5 ] && nodes=5
    # ctl/PRIORITY (ctl/priority_submit.sh) reserves the next free debug slot for a one-off job.
    # Both debug slots when no round job is queued or running (e.g. overnight between round 4 and round 5).
    # Two slots only in the window after round 5 is scored and before round 6 submits training: then no
    # round controller needs a slot. Otherwise one, so round jobs always get theirs between pieces.
    max_own=1
    grep -q ' DONE: ' "$HERE/../rag65v5/controller.log" 2>/dev/null && ! grep -q 'train round' "$HERE/../rag65v6/controller.log" 2>/dev/null && max_own=2
    # The 1.7B student (rag65s17), round 9 (rag65v9) and the KD runs (rag65kd, rag65kd2) train on up to 6 nodes: while
    # one is unfinished, take one slot and leave the nodes it needs, so its next training piece always fits and its
    # answer/judge jobs get the other slot. The round writes NODES_NEEDED before each piece (it shrinks as seeds finish).
    for r in rag65s17 rag65v9 rag65kd rag65kd2 rag65kd3 rag65kd4 rag65kd5 rag65kd6 rag65kd7 rag65kd8; do
        if [ -f "$HERE/../$r/controller.sh" ] && ! grep -q ' DONE: \| STOP: ' "$HERE/../$r/controller.log" 2>/dev/null; then
            # Until a round writes NODES_NEEDED: KD rounds train one node per seed (3), the others two per seed (6).
            dflt=6; [[ $r == rag65kd* ]] && dflt=3
            need=$(cat "$HERE/../$r/NODES_NEEDED" 2>/dev/null); [[ "$need" =~ ^[1-6]$ ]] || need=$dflt
            max_own=1; [ $nodes -gt $(( 8 - need )) ] && nodes=$(( 8 - need ))
            # A round waiting for a slot (its answer/judge submission was refused) goes first: take none.
            tail -1 "$HERE/../$r/controller.log" 2>/dev/null | grep -q 'submission refused' && max_own=0
            # Between two training pieces (training phase, no train job queued) the round is about to resubmit:
            # never take that slot (it lost 5-20 min twice on Oct 20).
            if grep -q 'train round' "$HERE/../$r/controller.log" && ! grep -q 'scaling adapters' "$HERE/../$r/controller.log" \
                    && ! remote "squeue --me -h -o %j" | grep -q "^dk-$r-train"; then
                max_own=0
            fi
        fi
    done
    # Score first (user, Oct 19): no new job while a score experiment is queued or running through priority_submit,
    # whose PRIORITY file disappears for a moment between two attempts.
    # Once the score job is submitted and in the queue, the other slot is free for us: yield only while it waits.
    pj=$(tail -1 "$HERE/../priority.log" 2>/dev/null | grep -o 'submitted [0-9]*' | awk '{print $2}')
    prio_running=0; [ -n "$pj" ] && squeue -h -j "$pj" 2>/dev/null | grep -q . && prio_running=1
    if pgrep -f "^bash $HERE/../priority_submit.sh|^bash $(dirname "$HERE")/priority_submit.sh" > /dev/null && [ $prio_running = 0 ]; then
        max_own=0
    fi
    prio_waiting=0; [ -e "$HERE/../PRIORITY" ] && [ $prio_running = 0 ] && prio_waiting=1
    if [ $prio_waiting = 0 ] && [ "$(ls "$HERE"/inflight | wc -l)" -lt $max_own ] && [ "$njobs" -lt 2 ] && [ "$rag_pending" -eq 0 ] && [ $nodes -ge 2 ]; then
        if [ "$(grep -c . <<< "$rev")" -ge $(( nodes * 4 )) ] || [ -z "$gen" ]; then phase=review; list=$rev; else phase=generate; list=$gen; fi
        list=$(head -n $(( nodes * 4 )) <<< "$list" | grep . | tr '\n' ' ')
        n=$(wc -w <<< "$list")
        if [ "$n" -gt 0 ]; then
            nodes=$(( (n + 3) / 4 ))
            j=$(remote "cd $C && export REPO=$C BATCH=$BATCH SHARD_LIST='$list' && sbatch --parsable --nodes=$nodes --export=ALL,PHASE=$phase,REVIEW_TP=2,GEN_TP=4 research/ai_factory_pilot/factory_dbg.sbatch" | tail -1)
            if [[ "$j" =~ ^[0-9]+$ ]]; then
                echo "$list" > "$HERE/inflight/$j"
                for k in $list; do p=${phase:0:1}; echo $(( $(cat "$HERE/attempts/$p$k" 2>/dev/null || echo 0) + 1 )) > "$HERE/attempts/$p$k"; done
                log "$phase job $j: $n shards on $nodes nodes (clean $clean/$SHARDS)"
            else
                log "submit refused: $j"
            fi
        fi
    fi
    # Zero-gap chaining (user, Oct 20: "not even for a single second should the job be empty"): while exactly one
    # of our jobs runs and none is queued, queue the next one behind it (afterany, same node count). It starts in our
    # own slot the moment the current job ends, never in a round's slot; --nice keeps any waiting score job first.
    own=$(ls "$HERE"/inflight)
    if [ $prio_waiting = 0 ] && [ "$rag_pending" -eq 0 ] && [ "$(grep -c . <<< "$own")" = 1 ]; then
        cur=$own
        if squeue -h -j "$cur" -t R -o %i 2>/dev/null | grep -q .; then
            cn=$(squeue -h -j "$cur" -o %D 2>/dev/null)
            busy=$(cat "$HERE"/inflight/* 2>/dev/null | tr '\n' ' ')
            gen=$(pick g); rev=$(pick r)
            if [ "$(grep -c . <<< "$rev")" -ge $(( cn * 4 )) ] || [ -z "$gen" ]; then phase=review; list=$rev; else phase=generate; list=$gen; fi
            list=$(head -n $(( cn * 4 )) <<< "$list" | grep . | tr '\n' ' ')
            n=$(wc -w <<< "$list")
            if [ "$n" -gt 0 ]; then
                nodes=$(( (n + 3) / 4 ))
                j=$(remote "cd $C && export REPO=$C BATCH=$BATCH SHARD_LIST='$list' && sbatch --parsable --nodes=$nodes --dependency=afterany:$cur --nice=10000 --export=ALL,PHASE=$phase,REVIEW_TP=2,GEN_TP=4 research/ai_factory_pilot/factory_dbg.sbatch" | tail -1)
                if [[ "$j" =~ ^[0-9]+$ ]]; then
                    echo "$list" > "$HERE/inflight/$j"
                    for k in $list; do p=${phase:0:1}; echo $(( $(cat "$HERE/attempts/$p$k" 2>/dev/null || echo 0) + 1 )) > "$HERE/attempts/$p$k"; done
                    log "$phase job $j queued after $cur: $n shards on $nodes nodes (clean $clean/$SHARDS)"
                fi
            fi
        fi
    fi
    sleep 45  # short gaps between 30-min jobs leave GPUs idle
done

# Every earlier release is a bank (v4 is MERGED, so all of them exist); fail closed if one is missing.
BANKS="runs/ai_factory_ccby_v1/release/verified.jsonl runs/ai_factory_ccby_v2/release/verified.jsonl runs/ai_factory_all_v1/release/verified.jsonl runs/ai_factory_all_v2/release/verified.jsonl runs/ai_factory_all_v3/release/verified.jsonl runs/ai_factory_all_v4/release/verified.jsonl"
for b in $BANKS; do
    test -s "$C/$b" || { log "STOP: bank release missing: $b"; exit 1; }
done
start_merge=$(date +%s)
remote "cd $C && export BATCH=$BATCH && PYTHONPATH=src HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 srun -A EUHPC_D30_031 -p lrd_all_serial -c 4 --mem=30G -t 01:30:00 $VPY research/ai_factory_pilot/merge_factory_v1.py $SHARDS $B7 $BANKS && PYTHONPATH=src $VPY research/ai_factory_pilot/annotate_general_v5.py" > "$HERE/merge.out"
# MERGED only for a release written by this merge: later stages start from it.
rel=$C/runs/$BATCH/release/verified.jsonl
if [ ! -s "$rel" ] || [ "$(stat -c %Y "$rel")" -lt "$start_merge" ] || ! grep -q '"annotated_by"' $C/runs/$BATCH/release/REPORT.json; then
    log "STOP: merge did not produce a new annotated release: $(tail -5 "$HERE/merge.out" | tr '\n' ' ')"; exit 1
fi
cp $C/runs/$BATCH/release/REPORT.json "$HERE/REPORT.json"
log "MERGED: $(tr -d '\n' < "$HERE/REPORT.json" | head -c 900)"
