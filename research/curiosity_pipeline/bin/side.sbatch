#!/bin/bash
#SBATCH --job-name=cp-side
#SBATCH --partition=hackathon
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=112
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/home/eai-jjppv/cp/logs/side-%j.out
# One extra task on team GPUs that nobody is using (team cap 8; the agent job holds 4). Submit it with
# --exclude=<agent node>, so the task's server cleanup (stop_all) runs on another node than the agent job.
# usage: use ~/cp/bin/side_submit.sh GPUS TASK.sh (keeps the team rule: at most 6 GPUs for us, 2 free)
export NGPU=4
t=$1; n=$(basename $t .sh)
# the autopilot only collects results of runs in its queued list: register a run that evaluates a model (its name is the part after _)
grep -q "eval.sh" "$t" && python3 - "${n#*_}" <<'PY'
import json, os, sys
p = os.path.expanduser('~/cp/autopilot/state.json'); s = json.load(open(p))
if sys.argv[1] not in s['queued'] and sys.argv[1] not in s['results']:
    s['queued'].append(sys.argv[1]); json.dump(s, open(p + '.new', 'w'), indent=1); os.replace(p + '.new', p)
PY
echo "$(date -u +%FT%TZ) side $SLURM_JOB_ID on $(hostname): start $n" >> $HOME/cp/logs/tasks.log
TASK=$n bash $t > $HOME/cp/logs/$n.log 2>&1
echo "$(date -u +%FT%TZ) side $SLURM_JOB_ID: end $n rc=$?" >> $HOME/cp/logs/tasks.log
