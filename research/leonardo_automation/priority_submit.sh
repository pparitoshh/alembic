#!/bin/bash
# usage: priority_submit.sh SBATCH_ARGS...   (run with setsid nohup; env such as REPO/RUN is passed through)
# Reserves the next free debug-QoS slot: factory controllers stop submitting while ctl/PRIORITY exists.
# Submits once a slot is free, waits for the job to end, then releases the reservation.
CTL="$(cd "$(dirname "$0")" && pwd)"
log() { echo "$(date -u +%FT%TZ) $*" >> "$CTL/priority.log"; }
echo "$*" > "$CTL/PRIORITY"
trap 'rm -f "$CTL/PRIORITY"' EXIT
until [ "$(squeue --me -h -q boost_qos_dbg 2>/dev/null | wc -l)" -lt 2 ] && squeue --me -h > /dev/null 2>&1; do sleep 60; done
j=$(sbatch --parsable "$@" 2>&1 | tail -1)
[[ "$j" =~ ^[0-9]+$ ]] || { log "submit refused: $j ($*)"; exit 1; }
log "submitted $j: $*"
while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done
log "job $j: $(sacct -n -X -j "$j" --format=State,Elapsed | head -1 | xargs)"
