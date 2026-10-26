#!/bin/bash
# Node feeder (Oct 25, user: "set so many batches in the queue so that this never happens again"). Every 5 minutes:
#  - a08trc08 (queue runner): when it has fewer than 6 busy nodes and fewer than 2 batches waiting in its queue (a batch's last shards run on 1-2 nodes), queue
#    the next batch so its start overlaps the tail;
#  - a08trc09 (this account, first since Oct 26): up to three factory batches whenever fewer than 6 nodes are busy here (the third takes
#    the others' 1-node tails into its jobs, run_factory.sh bundled tails),
#    next to the score work (run_factory.sh yields at the
#    2-job / 8-node limit and to WANT_SLOT, so it only fills nodes the score work leaves idle).
# New batches: all 156 taxonomy-v2 leaves with fresh angles (VARIANT_BASE +6 per batch), held-out families excluded,
# every earlier batch listed in SAME_SCHEME_BATCHES; from v34_prepare.sh. Four batches are kept prepared ahead (Oct 26; prep takes
# ~26 min on the login node), so idle nodes get work within a minute.
# Stop: touch ctl/feeder/STOP. Log: ctl/feeder/feeder.log
S=/leonardo_work/EUHPC_D30_031/shared_split; F=$S/ctl/feeder; mkdir -p $F
log() { echo "$(date -u +%FT%TZ) $*" >> $F/feeder.log; }
unfinished() { [ -d $S/ctl/$1 ] || return 0; ! grep -q ' DONE: \| STOP: ' $S/ctl/$1/controller.log 2>/dev/null; }
last_batch() { ls -d $S/code/runs/ai_factory_all_v* $S/ctl/ai_factory_all_v* 2>/dev/null | sed 's/.*_v//' | sort -n | tail -1; }

prepare() {  # $1 = batch number; writes v$1_prepare.sh, runs it; returns 0 when the plan exists
    local n=$1 k=$(( $1 - 34 )) same=""
    for v in $(seq 9 $(( n - 1 ))); do [ -d $S/code/runs/ai_factory_all_v$v ] && same="$same,\$S/code/runs/ai_factory_all_v$v"; done
    sed -e "s/ai_factory_all_v34/ai_factory_all_v$n/g; s/general-hpc-34/general-hpc-$n/" \
        -e "s/VARIANT_BASE=129 TOOL_VARIANT_BASE=329/VARIANT_BASE=$(( 129 + 6 * k )) TOOL_VARIANT_BASE=$(( 329 + 10 * k ))/" \
        -e "s#^export SAME_SCHEME_BATCHES=.*#export SAME_SCHEME_BATCHES=${same#,}#" \
        -e "s#^\# ai_factory_all_v34.*#\# ai_factory_all_v$n (feeder, Oct 25): all 156 leaves, new angles (VARIANT_BASE $(( 129 + 6 * k ))).#" \
        $S/ctl/v34_prepare.sh > $S/ctl/v${n}_prepare.sh.new && mv $S/ctl/v${n}_prepare.sh.new $S/ctl/v${n}_prepare.sh
    chmod g+rwx $S/ctl/v${n}_prepare.sh
    log "preparing ai_factory_all_v$n"
    (cd $S/ctl && bash v${n}_prepare.sh > v${n}_prepare.out 2>&1)
    grep -q '^rc=0' $S/ctl/v${n}_prepare.out && test -s $S/code/runs/ai_factory_all_v$n/summary.json
}

nodes_of() { squeue -u $1 -h -o '%D' 2>/dev/null | awk '{s += $1} END {print s + 0}'; }
assigned() { grep -q "^factory ai_factory_all_v$1\$" $S/ctl/QUEUE || [ -d $S/ctl/ai_factory_all_v$1 ]; }
backlog() { local k=0 b; for b in $(awk '$1 == "factory" {print $2}' $S/ctl/QUEUE); do [ -d $S/ctl/$b ] || k=$((k + 1)); done; echo $k; }
# controllers of this account still using GPUs (one that is merging its release runs on CPU only, Oct 26)
mine_ctl() {
    local k=0 b
    for b in $(pgrep -u $USER -af '^bash /leonardo_work/EUHPC_D30_031/shared_split/ctl/run_factory.sh ' | awk '{print $NF}'); do
        grep -q ' merging \| DONE: \| STOP: ' $S/ctl/$b/controller.log 2>/dev/null || k=$((k + 1))
    done
    echo $k
}
assign() {  # $1 = prepared batch number: here while this account has idle nodes (up to 3 controllers, Oct 26),
            # otherwise to a08trc08's queue
    local n=$1
    if [ "$(nodes_of $USER)" -lt 6 ] && [ "$(mine_ctl)" -lt 3 ]; then
        mkdir -p $S/ctl/ai_factory_all_v$n
        (setsid nohup bash $S/ctl/run_factory.sh ai_factory_all_v$n >> $S/ctl/ai_factory_all_v$n/nohup.out 2>&1 < /dev/null &)
        log "ai_factory_all_v$n started on a08trc09 ($(nodes_of $USER) nodes busy here)"
    else
        cp $S/ctl/QUEUE $S/ctl/QUEUE.feed && echo "factory ai_factory_all_v$n" >> $S/ctl/QUEUE.feed \
            && mv $S/ctl/QUEUE.feed $S/ctl/QUEUE && chmod g+rw $S/ctl/QUEUE
        log "ai_factory_all_v$n queued on a08trc08 ($(nodes_of a08trc08) nodes busy there)"
    fi
}

log "start (buffer of 4 prepared batches; a08trc08 < 6 busy nodes or no factory batch on a08trc09 -> assign one)"
while [ ! -e $F/STOP ]; do
    # buffer = prepared (summary.json) but not yet assigned
    buffer=()
    for d in $S/code/runs/ai_factory_all_v*; do
        n=${d##*_v}
        [ -s $d/summary.json ] && [ "$n" -ge 36 ] && ! assigned $n && ! pgrep -f "v${n}_prepare.sh" > /dev/null && buffer+=($n)
    done
    if [ ${#buffer[@]} -gt 0 ] && { { [ "$(nodes_of $USER)" -lt 6 ] && [ "$(mine_ctl)" -lt 3 ]; } || { [ "$(nodes_of a08trc08)" -lt 6 ] && [ "$(backlog)" -lt 2 ]; }; }; then
        assign ${buffer[0]}; buffer=("${buffer[@]:1}")
        sleep 120  # let the queue runner / controller start before measuring again
        continue
    fi
    preparing=$(pgrep -f "bash v[0-9]*_prepare.sh" | wc -l)
    if [ $(( ${#buffer[@]} + preparing )) -lt 4 ] && [ "$preparing" -lt 2 ]; then
        n=$(( $(last_batch) + 1 ))
        while [ -e $S/ctl/v${n}_prepare.sh ] || [ -d $S/code/runs/ai_factory_all_v$n ]; do n=$((n + 1)); done
        ( prepare $n || log "STOP-SKIP: preparation of v$n failed (see ctl/v${n}_prepare.out)" ) &
        sleep 20
        continue
    fi
    sleep 60
done
log "stopped (STOP file)"
