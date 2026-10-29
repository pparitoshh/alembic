#!/bin/bash
# Submit one extra task on spare team GPUs, within the team rule agreed with Paritosh on Oct 28: our jobs together
# use at most 7 of the team's 8 GPUs at all times, 1 always stays free (user, Oct 28 23:55: no 8-GPU window), and no
# second extra job while one is running.
# usage: side_submit.sh GPUS TASK.sh [MINUTES]
source /etc/profile.d/modules.sh; module load slurm
want=$1; task=$2; mins=${3:-90}
ours=$(squeue -h -u $USER -t R,PD -o %b | grep -o 'gpu:[0-9]*' | cut -d: -f2 | paste -sd+ | bc); ours=${ours:-0}
if squeue -h -u $USER -n cp-side | grep -q .; then echo "refused: an extra job (cp-side) is still running"; exit 1; fi
if [ $((ours + want)) -gt 7 ]; then echo "refused: we would use $((ours + want)) GPUs; the team rule is at most 7 (1 stays free for Paritosh)"; exit 1; fi
node=$(squeue -h -n cp-agent -u $USER -o %N)
sbatch --parsable --exclude=$node --gres=gpu:$want --cpus-per-task=$((28 * want)) --time=$mins $HOME/cp/bin/side.sbatch "$task"
