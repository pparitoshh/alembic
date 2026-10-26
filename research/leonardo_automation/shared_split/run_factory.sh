#!/bin/bash
# Lowest-priority filler task: generate (Qwen3-235B teacher, factory_worker.sh) and review (Gemma 4) every shard of a
# prepared general-HPC batch in shared_split/code/runs/<batch>, then merge it into a release. Modelled on a08trc09's
# ctl/factory_v8/controller.sh (same jobs, per-shard markers GENERATE_DONE / REVIEW_V2_DONE, MAX_ATTEMPTS per shard and
# phase, same merge), without its rag65/priority/FREEZE rules. Runs under whichever account starts it.
# Slot rules: at most one job of its own at a time, sized to the free nodes (8 per user minus nodes already queued);
# a 1-node tail that would take the user's last slot is offered to another batch's next job (bundled tails, Oct 26);
# it YIELDS (submits nothing) while any ctl/*/WANT_SLOT exists, while the user already has 2 jobs, or while an
# unfinished review filler listed in ctl/QUEUE waits for a slot.
# usage: run_factory.sh <batch>   (started by queue_runner.sh; resumable: rerun to continue)
#        DRY_RUN=1 run_factory.sh <batch>   prints the state and what it WOULD submit or merge, then exits
#        (DRY_QUEUE='<jobid> <nodes>\n...' simulates the user's queue in a dry run; DRY_QUEUE= simulates an empty one)
set -uo pipefail
umask 002  # files (and, through sbatch's umask propagation, job outputs) stay group-writable for both accounts
BATCH=${1:?batch}
S=/leonardo_work/EUHPC_D30_031/shared_split
C=$S/code; HERE=$S/ctl/$BATCH
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/code   # read-only bank releases
VPY=$S/venv-train/bin/python
B7=/leonardo_work/EUHPC_D30_031/runs/a08trc09/student-mixed-v2-20261015T042338Z/bundle/data/b7_original.jsonl
MAX_ATTEMPTS=${MAX_ATTEMPTS:-10}  # per shard and phase, like v8 (shards hold at most 450 questions)
DRY=${DRY_RUN:-0}
JOBNAME=dk-$BATCH
mkdir -p "$HERE/inflight" "$HERE/attempts"
log() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; fi; }
heartbeat() { [ "$DRY" = 1 ] || touch "$HERE/HEARTBEAT"; }
group_rw() { find "$@" -user "$USER" \( ! -perm -g=rw -o -type d ! -perm -g=x \) -exec chmod g+rwX {} + 2>/dev/null; }

cd $C || { log "STOP: cannot enter $C"; exit 1; }
test -f runs/$BATCH/summary.json || { log "STOP: runs/$BATCH is not prepared (no summary.json)"; exit 1; }
SHARDS=$(ls -d runs/$BATCH/shard_* | wc -l)
for k in $(seq -f %03g 0 $((SHARDS - 1))); do
    test -s configs/${BATCH}_shard_$k.json || { log "STOP: configs/${BATCH}_shard_$k.json missing"; exit 1; }
done
test -d data/eval || { log "STOP: data/eval missing (distillkit decontam needs it)"; exit 1; }
log "start on $(hostname) as $USER: $SHARDS shards, $(python3 -c "import json; print(json.load(open('runs/$BATCH/summary.json'))['planned_questions'])") planned questions"

# A review filler earlier in ctl/QUEUE that is unfinished, alive and has no job of its own is waiting for a slot.
filler_waiting() {
    local kind name
    while read -r kind name; do
        [ "$kind" = review ] && [ -n "${name:-}" ] || continue
        grep -q ' DONE: \| STOP: ' $S/ctl/$name/controller.log 2>/dev/null && continue
        pgrep -f "^bash $S/ctl/run_review.sh $name\$" > /dev/null || continue
        squeue --me -h -n dk-$name -o %i 2>/dev/null | grep -q . && continue
        echo "$name"; return 0
    done < $S/ctl/QUEUE
    return 1
}

# Bundled tails (Oct 26): a batch whose next job would be a 1-node tail taking the user's last debug slot offers its
# shards in ctl/.donate_$USER/<batch> instead; the next job another batch of this account submits runs them on extra
# nodes (factory_dbg.sbatch SHARD_LIST2) and records the job in the donor's inflight/. Unclaimed offers expire after
# 10 minutes and the donor submits the tail itself.
DON=$S/ctl/.donate_$USER
[ "$DRY" = 1 ] || mkdir -p $DON
partner_ready() {  # another live GPU batch of this account without a job in flight, which will submit soon
    local b
    for b in $(pgrep -u $USER -af "^bash $S/ctl/run_factory.sh " | awk '{print $NF}'); do
        [ "$b" = "$BATCH" ] && continue
        grep -q ' merging \| DONE: \| STOP: ' $S/ctl/$b/controller.log 2>/dev/null && continue
        ls $S/ctl/$b/inflight/ 2>/dev/null | grep -q . && continue
        [ -e $DON/$b ] && continue
        return 0
    done
    return 1
}
donation_pending() {  # 0 while this batch's offered tail waits for, or is being taken by, another batch's job
    local f=$DON/$BATCH c age
    if [ -e $f ]; then
        age=$(( $(date +%s) - $(stat -c %Y $f 2>/dev/null || date +%s) ))
        [ $age -lt 600 ] && return 0
        if mv $f $HERE/.donation_expired 2>/dev/null; then
            echo $(( $(date +%s) + 1800 )) > $HERE/no_donate_until
            log "tail offer expired after ${age}s; submitting it itself"; return 1
        fi
        return 0  # taken meanwhile
    fi
    for c in $DON/$BATCH.claimed.*; do
        [ -e "$c" ] || continue
        [ $(( $(date +%s) - $(stat -c %Y $c 2>/dev/null || echo 0) )) -lt 300 ] && return 0
        rm -f $c; log "stale claim $(basename $c) removed"
    done
    return 1
}

last_wait=""
while true; do
    heartbeat
    # A filler never blocks a round: cancel our own still-PENDING jobs while any round waits for a slot.
    if [ "$DRY" != 1 ] && ls $S/ctl/*/WANT_SLOT > /dev/null 2>&1; then
        for p in $(squeue --me -h -t PD -n dk-$BATCH -o %i 2>/dev/null); do scancel $p; log "cancelled own pending job $p: a round waits for a slot"; done
    fi
    # Forget a job once it is out of the queue and sacct reports a final state (or, for a job sacct cannot show,
    # e.g. one submitted by the other account before a restart, once it is out of the queue for 40 minutes).
    ended=0
    for f in "$HERE"/inflight/*; do
        [ -e "$f" ] || continue
        j=$(basename "$f")
        squeue -h -j "$j" -o %T 2>/dev/null | grep -q . && continue
        st=$(sacct -n -X -j "$j" --format=State,Elapsed 2>/dev/null | head -1 | xargs)
        case "$st" in RUNNING*|PENDING*|REQUEUED*|CONFIGURING*|COMPLETING*) continue ;; esac
        if [ -z "$st" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 2400 ]; then continue; fi
        [ "$DRY" = 1 ] && { echo "[dry-run] would forget job $j (${st:-not in sacct})"; continue; }
        log "job $j: ${st:-ended (not in sacct)}"
        tail -3 $S/logs/factory-*-$j-0.err 2>/dev/null | cut -c1-300 >> "$HERE/controller.log"
        rm -f "$f"; ended=1
    done
    [ $ended = 1 ] && group_rw runs/$BATCH $S/logs $C/slurm/logs
    state=$(for k in $(seq -f %03g 0 $((SHARDS - 1))); do
        d=runs/$BATCH/shard_$k
        if [ -f $d/REVIEW_V2_DONE ] && [ -f $d/clean.jsonl ]; then echo "c $k"; elif [ -f $d/GENERATE_DONE ]; then echo "r $k"; else echo "g $k"; fi; done)
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
    ninflight=$(ls "$HERE"/inflight | wc -l)
    [ "$DRY" = 1 ] && echo "[dry-run] shards: $clean clean, $(grep -c '^r ' <<< "$state") to review, $(grep -c '^g ' <<< "$state") to generate; pickable: gen=[$(echo $gen)] rev=[$(echo $rev)]; own jobs in flight: $ninflight"
    if [ -z "$gen$rev" ] && [ "$ninflight" = 0 ]; then
        log "all shards finished or exhausted: $clean/$SHARDS clean"; break
    fi
    wait_reason=""
    if [ "$ninflight" -ge 1 ]; then
        wait_reason="own job $(ls "$HERE"/inflight | tr '\n' ' ')in flight"
    elif [ "$DRY" != 1 ] && donation_pending; then
        wait_reason="tail offered to another batch's next job"
    else
        if [ "$DRY" = 1 ] && [ -n "${DRY_QUEUE+x}" ]; then jobs=$(printf '%b' "$DRY_QUEUE")  # test: simulated 'jobid nodes' lines
        # One controller per account measures and submits at a time (Oct 25: two controllers both saw 5 free nodes
        # and both asked for 5, so one job sat pending and 3 nodes stayed idle). The lock is released after submitting.
        LOCKD=$S/ctl/.factory_lock_$USER
        until mkdir $LOCKD 2>/dev/null; do [ $(( $(date +%s) - $(stat -c %Y $LOCKD 2>/dev/null || date +%s) )) -gt 300 ] && rmdir $LOCKD 2>/dev/null; sleep $(( RANDOM % 5 + 2 )); done
        HAVE_LOCK=1; trap '[ "$HAVE_LOCK" = 1 ] && rmdir $LOCKD 2>/dev/null' EXIT
        else jobs=$(squeue --me -h -p boost_usr_prod -o '%i %D' 2>/dev/null) || { sleep 60; continue; }; fi  # GPU jobs only: CPU-partition jobs (merges, serial) do not count against the debug QoS (Oct 25)
        njobs=$(grep -c . <<< "$jobs"); used=$(awk '{s += $2} END {print s + 0}' <<< "$jobs"); free=$(( 8 - used ))
        # Oct 26: yield only while a score job is pending or none runs (between pieces); a running score job with none
        # waiting leaves the second slot to this batch (an hour-long 3-node training otherwise kept 5 nodes idle).
        if [ "$USER" = a08trc09 ] && [ -e /leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/ctl/PRIORITY ] \
            && { squeue --me -h -t PD -p boost_usr_prod -o %j 2>/dev/null | grep -qv '^dk-ai_factory' \
                 || ! squeue --me -h -t R -p boost_usr_prod -o %j 2>/dev/null | grep -qv '^dk-ai_factory'; }; then wait_reason="yield: a score job of a08trc09 waits for a slot (ctl/PRIORITY)"  # Oct 25: feeder batches on this account never delay score work
        elif ls $S/ctl/*/WANT_SLOT > /dev/null 2>&1; then wait_reason="yield: $(ls $S/ctl/*/WANT_SLOT | xargs -n1 dirname | xargs -n1 basename | tr '\n' ' ')wants a slot"
        elif [ "$njobs" -ge 2 ]; then wait_reason="yield: $USER already has $njobs jobs"
        elif [ "$free" -lt 1 ]; then wait_reason="yield: no free node ($used of 8 queued)"
        elif fw=$(filler_waiting); then wait_reason="yield: filler $fw waits for a slot"
        fi
    fi
    if [ -n "$wait_reason" ]; then
        [ "$DRY" = 1 ] && { echo "[dry-run] would submit nothing now ($wait_reason)"; }
        [ "$DRY" != 1 ] && [ "${wait_reason%% *}" = yield: ] && [ "$wait_reason" != "$last_wait" ] && log "$wait_reason"
        last_wait=$wait_reason
    else
        last_wait=""
        # Review first once there is a full job's worth of generated shards (or nothing left to generate), like v8.
        if [ "$(grep -c . <<< "$rev")" -ge $(( free * 4 )) ] || [ -z "$gen" ]; then phase=review; list=$rev; else phase=generate; list=$gen; fi
        list=$(head -n $(( free * 4 )) <<< "$list" | grep . | tr '\n' ' ')
        n=$(wc -w <<< "$list")
        nodes=$(( (n + 3) / 4 ))  # 4 shards per node in both phases (one TP4 teacher, or two TP2 reviewers)
        # Take another batch's offered tail when it fits next to at least one node of our own shards.
        D2=""; LIST2=""; PHASE2=none; N2=0
        if [ "$DRY" != 1 ]; then
            for f in $DON/*; do
                b=${f##*/}; [ -f "$f" ] && [ "$b" != "$BATCH" ] || continue
                case $b in *.claimed.*|.*) continue ;; esac
                mv "$f" "$f.claimed.$BATCH" 2>/dev/null || continue
                PHASE2=$(sed -n 1p "$f.claimed.$BATCH"); LIST2=$(sed -n 2p "$f.claimed.$BATCH"); N2=$(( ($(wc -w <<< "$LIST2") + 3) / 4 ))
                if [ "$N2" -ge 1 ] && [ $(( free - N2 )) -ge 1 ]; then D2=$b; break; fi
                mv "$f.claimed.$BATCH" "$f"; LIST2=""; PHASE2=none; N2=0  # does not fit: leave it for another job
            done
        fi
        if [ -n "$D2" ]; then
            if [ "$phase" = review ]; then src=$rev; else src=$gen; fi
            list=$(head -n $(( (free - N2) * 4 )) <<< "$src" | grep . | tr '\n' ' ')
            n=$(wc -w <<< "$list"); nodes=$(( (n + 3) / 4 + N2 ))
        elif [ "$DRY" != 1 ] && [ "$nodes" -le 1 ] && [ "$njobs" -ge 1 ] \
            && [ "$(date +%s)" -ge "$(cat $HERE/no_donate_until 2>/dev/null || echo 0)" ] && partner_ready; then
            printf '%s\n%s\n' "$phase" "$list" > $DON/.$BATCH.tmp && mv $DON/.$BATCH.tmp $DON/$BATCH
            log "tail of $n $phase shards offered to another batch's next job (keeps the last debug slot for a bigger job)"
            n=0  # submit nothing now
        fi
        cmd="cd $C && REPO=$C BATCH=$BATCH SHARD_LIST='$list' sbatch --parsable --nodes=$nodes --job-name=$JOBNAME --nice=2000 --export=ALL,PHASE=$phase,REVIEW_TP=2,GEN_TP=4 research/ai_factory_pilot/factory_dbg.sbatch"
        if [ "$DRY" = 1 ]; then
            echo "[dry-run] free nodes $free (user has $njobs jobs, $used nodes queued): WOULD submit $phase job for $n shards on $nodes node(s):"
            echo "[dry-run]   $cmd"
        elif [ "$n" -gt 0 ]; then
            j=$(cd $C && REPO=$C BATCH=$BATCH SHARD_LIST="$list" BATCH2=$D2 SHARD_LIST2="$LIST2" sbatch --parsable --nodes=$nodes \
                --job-name=$JOBNAME --nice=2000 --export=ALL,PHASE=$phase,PHASE2=$PHASE2,REVIEW_TP=2,GEN_TP=4 \
                research/ai_factory_pilot/factory_dbg.sbatch 2>&1 | tail -1)
            if [[ "$j" =~ ^[0-9]+$ ]]; then
                echo "$list" > "$HERE/inflight/$j"
                for k in $list; do p=${phase:0:1}; echo $(( $(cat "$HERE/attempts/$p$k" 2>/dev/null || echo 0) + 1 )) > "$HERE/attempts/$p$k"; done
                if [ -n "$D2" ]; then
                    echo "$LIST2" > $S/ctl/$D2/inflight/$j
                    for k in $LIST2; do p=${PHASE2:0:1}; echo $(( $(cat "$S/ctl/$D2/attempts/$p$k" 2>/dev/null || echo 0) + 1 )) > "$S/ctl/$D2/attempts/$p$k"; done
                    echo "$(date -u +%FT%TZ) $PHASE2 shards [$LIST2] run in job $j of $BATCH (bundled tail)" >> $S/ctl/$D2/controller.log
                    rm -f $DON/$D2.claimed.$BATCH
                fi
                log "$phase job $j: $n shards on $(( nodes - N2 )) node(s) (clean $clean/$SHARDS)${D2:+ + $(wc -w <<< "$LIST2") $PHASE2 shards of $D2 on $N2 node(s) (bundled tail)}"
            else
                [ -n "$D2" ] && mv $DON/$D2.claimed.$BATCH $DON/$D2 2>/dev/null
                log "submit refused: $j"; sleep 120
            fi
        fi
    fi
    [ "${HAVE_LOCK:-0}" = 1 ] && { rmdir $S/ctl/.factory_lock_$USER 2>/dev/null; HAVE_LOCK=0; }  # squeue already shows the new job
    [ "$DRY" = 1 ] && break
    sleep 30
done

# Merge (CPU): every earlier release is a bank (v1-v8 and the CC-BY batches, as in v8); fail closed if one is missing.
BANKS=""
for b in ai_factory_ccby_v1 ai_factory_ccby_v2 ai_factory_all_v1 ai_factory_all_v2 ai_factory_all_v3 ai_factory_all_v4 \
         ai_factory_all_v5 ai_factory_all_v6 ai_factory_all_v7 ai_factory_all_v8; do
    test -r "$R/runs/$b/release/verified.jsonl" || { log "STOP: bank release not readable: $R/runs/$b/release/verified.jsonl"; exit 1; }
    BANKS="$BANKS $R/runs/$b/release/verified.jsonl"
done
test -r $B7 || { log "STOP: cannot read $B7"; exit 1; }
# Shared-folder batches (v9, v10, ...) are banks too, so a later batch does not re-release their questions.
for b in ai_factory_all_v9 ai_factory_all_v10 ai_factory_all_v11; do
    f=$S/code/runs/$b/release/verified.jsonl
    [ "$b" != "$BATCH" ] && [ -s "$f" ] && BANKS="$BANKS $f"
done
MERGE="export BATCH=$BATCH PYTHONPATH=src HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; $VPY research/ai_factory_pilot/merge_factory_v1.py $SHARDS $B7$BANKS && $VPY research/ai_factory_pilot/annotate_general_v5.py"
if [ "$DRY" = 1 ]; then
    echo "[dry-run] after the last shard it WOULD merge (cwd $C):"
    echo "[dry-run]   srun -A EUHPC_D30_031 -p lrd_all_serial -c 4 --mem=30G -t 01:30:00 bash -c \"$MERGE\""
    exit 0
fi
log "merging $clean/$SHARDS clean shards"
start_merge=$(date +%s)
srun -A EUHPC_D30_031 -p lrd_all_serial -c 4 --mem=30G -t 01:30:00 -J $JOBNAME-merge bash -c "cd $C && $MERGE" > "$HERE/merge.out" 2>&1
rc=$?
rel=runs/$BATCH/release/verified.jsonl
if [ $rc != 0 ] && grep -qiE 'srun: error|Unable to allocate|Invalid account|partition' "$HERE/merge.out" && [ ! -s $rel ]; then
    log "serial partition unavailable ($(grep -iE 'srun: error' "$HERE/merge.out" | head -1)); merging on the login node (nice 19)"
    nice -n 19 bash -c "cd $C && $MERGE" >> "$HERE/merge.out" 2>&1
fi
if [ ! -s "$rel" ] || [ "$(stat -c %Y "$rel")" -lt "$start_merge" ] || ! grep -q '"annotated_by"' runs/$BATCH/release/REPORT.json; then
    log "STOP: merge did not produce a new annotated release: $(tail -5 "$HERE/merge.out" | tr '\n' ' ' | cut -c1-600)"; exit 1
fi
group_rw runs/$BATCH "$HERE"
cp runs/$BATCH/release/REPORT.json "$HERE/REPORT.json"
summary=$(python3 -c "
import json
r = json.load(open('runs/$BATCH/release/REPORT.json')); s = json.load(open('runs/$BATCH/summary.json'))
clean = sum(v for v in r['clean_rows_per_shard'].values() if isinstance(v, int))
print('%d rows released (%d permissive) from %d clean of %d planned; rejections %s; by_topic %s' % (
    r['released_rows'], r.get('permissive_rows', 0), clean, s['planned_questions'], json.dumps(r['rejections']),
    json.dumps(dict(list(r.get('by_topic', {}).items())[:8]))))")
log "DONE: $summary -> $C/$rel"
