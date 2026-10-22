#!/bin/bash
# Restarts a dead controller. Chain: factory_v1 -> (exec) rag65v4 -> (exec) rag65v5; factory_v2/v3 run alongside.
# A controller is restarted only when it is unfinished, has not logged STOP and its process is gone. Factory
# controllers keep in-flight jobs in files; round controllers wait until no dk-rag65 job is queued.
# Controllers run as setsid/nohup processes, not tmux: a tmux server carries its first session's command
# line, so `pkill -f <controller>` once killed the server and every controller with it.
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
CTL=$R/ctl
log() { echo "$(date -u +%FT%TZ) $*" >> "$CTL/watchdog/watchdog.log"; }
alive() { pgrep -f "/$1/controller.sh" > /dev/null; }  # by name: also matches .../factory_v1/../rag65v4/ after an exec
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
    if [ -f "$CTL/factory_v4/controller.sh" ] && ! grep -q ' MERGED: ' "$CTL/factory_v4/controller.log" 2>/dev/null \
            && ! alive factory_v4 && ! stopped factory_v4; then
        start factory_v4
    fi
    if [ -f "$CTL/factory_v5/controller.sh" ] && ! grep -q ' MERGED: ' "$CTL/factory_v5/controller.log" 2>/dev/null \
            && ! alive factory_v5 && ! stopped factory_v5; then
        start factory_v5
    fi
    if [ -f "$CTL/factory_v6/controller.sh" ] && ! grep -q ' MERGED: ' "$CTL/factory_v6/controller.log" 2>/dev/null \
            && ! alive factory_v6 && ! stopped factory_v6; then
        start factory_v6
    fi
    if [ -f "$CTL/factory_v7/controller.sh" ] && ! grep -q ' MERGED: ' "$CTL/factory_v7/controller.log" 2>/dev/null \
            && ! alive factory_v7 && ! stopped factory_v7; then
        start factory_v7
    fi
    if [ -f "$CTL/factory_v8/controller.sh" ] && ! grep -q ' MERGED: ' "$CTL/factory_v8/controller.log" 2>/dev/null \
            && ! alive factory_v8 && ! stopped factory_v8; then
        start factory_v8
    fi
    if [ -f "$CTL/bestofn/controller.sh" ] && ! grep -q ' BUILT: ' "$CTL/bestofn/controller.log" 2>/dev/null \
            && ! alive bestofn && ! stopped bestofn; then
        start bestofn
    fi
    for c in bestofn2 rag65v8 dpo rag65s17 rag65v9 rag65kd rag65kd2 rag65kd3 rag65kd4 rag65kd5 rag65kd6 rag65kd7 rag65kd8 rag65kd9 rag65kd10 rag65kd11 rft1 t120a gguf_r12c gguf_eval gguf_imat goals $(cat "$CTL/auto_rounds.txt" 2>/dev/null); do  # + auto-chained rounds  # standalone: best-of-n #2, round 8, DPO, 1.7B student, GGUF evals, goals chain
        if [ -f "$CTL/$c/controller.sh" ] && ! grep -q ' BUILT: \| DONE: ' "$CTL/$c/controller.log" 2>/dev/null \
                && ! alive $c && ! stopped $c; then
            start $c
        fi
    done
    # Round 7 runs standalone from the best-of-n build, alongside round 6.
    if [ -f "$CTL/rag65v7/controller.sh" ] && ! grep -q ' DONE: ' "$CTL/rag65v7/controller.log" 2>/dev/null \
            && ! alive rag65v7 && ! stopped rag65v7; then
        start rag65v7
    fi
    # Round 5 runs standalone alongside round 4's scoring; restart it on its own when it dies.
    if [ -f "$CTL/rag65v5/controller.sh" ] && ! grep -q ' DONE: ' "$CTL/rag65v5/controller.log" 2>/dev/null \
            && ! alive rag65v5 && ! stopped rag65v5 && alive rag65v4; then
        start rag65v5
    fi
    # Main chain.
    if ! alive factory_v1 && ! alive rag65v4 && ! alive rag65v5 && ! alive rag65v6 && ! alive rag65v7; then
        if ! grep -q ' MERGED: ' "$CTL/factory_v1/controller.log"; then
            stopped factory_v1 || start factory_v1
        elif ! grep -q ' DONE: ' "$CTL/rag65v4/controller.log" 2>/dev/null; then
            stopped rag65v4 || grep -q 'dk-rag65' <<< "$queue" || start rag65v4
        elif ! grep -q ' DONE: ' "$CTL/rag65v5/controller.log" 2>/dev/null; then
            stopped rag65v5 || grep -q 'dk-rag65' <<< "$queue" || start rag65v5
        elif ! grep -q ' DONE: ' "$CTL/rag65v6/controller.log" 2>/dev/null; then
            stopped rag65v6 || grep -q 'dk-rag65' <<< "$queue" || start rag65v6
        fi
    fi
    sleep 60  # a dead controller costs at most a minute (user, Oct 20: no idle second)
done
