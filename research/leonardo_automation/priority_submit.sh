#!/bin/bash
# usage: priority_submit.sh SBATCH_ARGS...   (run with setsid nohup; env such as REPO/RUN is passed through)
# Reserves the next free debug-QoS slot: factory controllers stop submitting while ctl/PRIORITY exists.
# Submits straight into the queue, waits for the job to end, then releases the reservation.
CTL="$(cd "$(dirname "$0")" && pwd)"
log() { echo "$(date -u +%FT%TZ) $*" >> "$CTL/priority.log"; }
echo "$*" > "$CTL/PRIORITY"
trap 'rm -f "$CTL/PRIORITY"' EXIT
# Take a free slot only while no round is between training pieces. A round is in its training phase from its
# first 'train round' line until 'scaling adapters'; during that phase its train job must be in the queue,
# otherwise the free slot is the gap the round controller is about to fill.
in_gap() {
    for c in "$CTL"/rag65*/controller.log; do  # rounds (rag65v*) and the 1.7B student (rag65s17)
        grep -q 'train round' "$c" 2>/dev/null || continue
        grep -q 'scaling adapters\| DONE: \| STOP: ' "$c" && continue
        squeue --me -h -t PD,R -o %j 2>/dev/null | grep -q 'rag65.*-train' || return 0  # a COMPLETING piece is not queued
    done
    return 1
}
# A refused submission (a finishing job still holds its slot) waits and retries; it is not a failure.
while true; do
    # Queue at once (user, Oct 20: always have the next job waiting): the job waits in Slurm's queue, not here, and
    # starts the moment a slot frees, ahead of the factory's low-priority chained jobs (--nice).
    until squeue --me -h > /dev/null 2>&1 && ! in_gap; do sleep 60; done
    j=$(sbatch --parsable "$@" 2>&1 | tail -1)
    [[ "$j" =~ ^[0-9]+$ ]] && break
    log "submit refused, retrying: $j ($*)"; sleep 120
done
log "submitted $j: $*"
while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done
log "job $j: $(sacct -n -X -j "$j" --format=State,Elapsed | head -1 | xargs)"
