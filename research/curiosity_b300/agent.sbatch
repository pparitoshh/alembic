#!/bin/bash
#SBATCH --job-name=cp-agent
#SBATCH --partition=hackathon
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=224
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/eai-jjppv/cp/logs/agent-%j.out
# The team's whole GPU share (8 B300 on one node) as one long job that runs ~/cp/queue/*.sh in name order, one at a
# time: each task gets all 8 GPUs and starts/stops its own servers. Finished tasks move to queue/done or queue/failed.
# Stop: touch ~/cp/queue/STOP (the watchdog then stops resubmitting too).
Q=$HOME/cp/queue; mkdir -p $Q/running $Q/done $Q/failed $HOME/cp/logs
echo "$(date -u +%FT%TZ) agent $SLURM_JOB_ID on $(hostname)" >> $HOME/cp/logs/tasks.log
for t in $Q/running/*.sh; do [ -e "$t" ] && mv "$t" $Q/; done  # a task cut by the previous agent's end runs again
while [ ! -e $Q/STOP ]; do
    t=$(ls $Q/*.sh 2>/dev/null | sort | head -1)
    if [ -z "$t" ]; then touch $Q/IDLE; sleep 30; continue; fi
    rm -f $Q/IDLE
    n=$(basename $t .sh); mv $t $Q/running/
    pkill -u $USER -f "^$HOME/cp/venv/bin/python3 $HOME/cp/venv/bin/vllm serve"; pkill -9 -u $USER -f "^VLLM::"; sleep 3  # no server left from the last task
    echo "$(date -u +%FT%TZ) start $n" >> $HOME/cp/logs/tasks.log
    TASK=$n bash $Q/running/$n.sh > $HOME/cp/logs/$n.log 2>&1
    rc=$?
    if [ $rc = 0 ]; then mv $Q/running/$n.sh $Q/done/; else mv $Q/running/$n.sh $Q/failed/; fi
    echo "$(date -u +%FT%TZ) end $n rc=$rc" >> $HOME/cp/logs/tasks.log
done
