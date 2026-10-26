#!/bin/bash
# Keeps every task in ctl/QUEUE running until it logs DONE or STOP: starts tasks that have not started, restarts
# controllers that died. Lines: "round <name>" (run_round.sh), "review <name>" (run_review.sh) or "factory <batch>"
# (run_factory.sh); order = priority.
# The main session appends new tasks to ctl/QUEUE. usage: setsid nohup bash queue_runner.sh &   (one per account)
S=/leonardo_work/EUHPC_D30_031/shared_split
log() { echo "$(date -u +%FT%TZ) $*" >> $S/ctl/queue_runner_$USER.log; }
log "queue runner started on $(hostname)"
while true; do
    touch $S/ctl/QUEUE_HEARTBEAT_$USER
    while read -r kind name; do
        [ -n "${name:-}" ] || continue
        case "$kind" in round) script=run_round.sh ;; round2) script=run_round2.sh ;; review) script=run_review.sh ;; factory) script=run_factory.sh ;; *) continue ;; esac
        mkdir -p $S/ctl/$name
        grep -q ' DONE: \| STOP: ' $S/ctl/$name/controller.log 2>/dev/null && continue
        pgrep -f "^bash $S/ctl/$script $name\$" > /dev/null && continue
        log "starting $kind $name"
        (setsid nohup bash $S/ctl/$script $name >> $S/ctl/$name/nohup.out 2>&1 < /dev/null &)
    done < $S/ctl/QUEUE
    sleep 120
done
