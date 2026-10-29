#!/bin/bash
# Starts side_543_kdcbx (3 GPUs: with the agent job 7, one always free) once the cbextra side job has ended and both
# closed-book teacher sets exist. side_submit.sh enforces the 7-GPU rule.
source /etc/profile.d/modules.sh; module load slurm
for i in $(seq 1 240); do
    if ! squeue -h -u $USER -n cp-side | grep -q . && [ -s ~/cp/exp/kdcbx/teacher/part0.npz ] \
       && [ -s ~/cp/exp/kdcb/teacher/part0.npz ] && [ -s ~/cp/exp/kdcb/teacher/part1.npz ]; then
        echo "$(date -u +%FT%TZ) launching side_543_kdcbx (3 GPUs)" >> ~/cp/logs/tasks.log
        ~/cp/bin/side_submit.sh 3 ~/cp/cb/side_543_kdcbx.sh 300 >> ~/cp/logs/tasks.log 2>&1
        exit 0
    fi
    sleep 120
done
echo "$(date -u +%FT%TZ) launch_543: gave up after 8 hours" >> ~/cp/logs/tasks.log
