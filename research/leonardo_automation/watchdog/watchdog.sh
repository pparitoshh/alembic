#!/bin/bash
# Restarts a dead controller. Chain: factory_v1 -> (exec) rag65v4 -> (exec) rag65v5; factory_v2/v3 run alongside.
# A controller is restarted only when it is unfinished, has not logged STOP and its process is gone. Factory
# controllers keep in-flight jobs in files; round controllers wait until no dk-rag65 job is queued.
# Controllers run as setsid/nohup processes, not tmux: a tmux server carries its first session's command
# line, so `pkill -f <controller>` once killed the server and every controller with it.
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
CTL=$R/ctl
log() { echo "$(date -u +%FT%TZ) $*" >> "$CTL/watchdog/watchdog.log"; }
alive() { pgrep -f "bash $CTL/$1/controller.sh" > /dev/null; }  # also matches /bin/bash after an exec
start() {  # $1 = controller dir
    setsid nohup bash "$CTL/$1/controller.sh" >> "$CTL/$1/nohup.out" 2>&1 < /dev/null &
    log "restarted $1 (pid $!)"
}
stopped() { grep -q ' STOP: ' "$CTL/$1/controller.log" 2>/dev/null; }
while true; do
    touch "$CTL/watchdog/HEARTBEAT"
    queue=$(squeue --me -h -o '%j' 2>/dev/null) || { sleep 300; continue; }  # never act on a failed squeue
    if ! grep -q ' MERGED: ' "$CTL/factory_v2/controller.log" 2>/dev/null && ! alive factory_v2 && ! stopped factory_v2; then
        start factory_v2
    fi
    if [ -f "$CTL/factory_v3/controller.sh" ] && ! grep -q ' MERGED: ' "$CTL/factory_v3/controller.log" 2>/dev/null \
            && ! alive factory_v3 && ! stopped factory_v3; then
        start factory_v3
    fi
    # Main chain.
    if ! alive factory_v1 && ! alive rag65v4 && ! alive rag65v5; then
        if ! grep -q ' MERGED: ' "$CTL/factory_v1/controller.log"; then
            stopped factory_v1 || start factory_v1
        elif ! grep -q ' DONE: ' "$CTL/rag65v4/controller.log" 2>/dev/null; then
            stopped rag65v4 || grep -q 'dk-rag65' <<< "$queue" || start rag65v4
        elif ! grep -q ' DONE: ' "$CTL/rag65v5/controller.log" 2>/dev/null; then
            stopped rag65v5 || grep -q 'dk-rag65' <<< "$queue" || start rag65v5
        fi
    fi
    sleep 300
done
