#!/bin/bash
# Login-node watchdog: keeps one cp-agent job queued or running (resubmits when it ends), every 2 minutes.
source /etc/profile.d/modules.sh 2>/dev/null; module load slurm
Q=$HOME/cp/queue; mkdir -p $Q $HOME/cp/logs
while [ ! -e $Q/STOP ]; do
    touch $HOME/cp/logs/WATCHDOG_HEARTBEAT
    if ! squeue --me -h -n cp-agent -o %i | grep -q .; then
        n=$(cat $Q/NGPU 2>/dev/null || echo 8)  # queue/NGPU: GPUs for the next agent job (team share split)
        j=$(sbatch --parsable --gres=gpu:$n --cpus-per-task=$(( 28 * n )) --export=ALL,NGPU=$n $HOME/cp/bin/agent.sbatch 2>&1)
        echo "$(date -u +%FT%TZ) watchdog: submitted agent $j" >> $HOME/cp/logs/tasks.log
    fi
    sleep 120
done
