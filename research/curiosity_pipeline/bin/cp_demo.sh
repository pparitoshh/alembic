#!/bin/bash
# ClusterPilot demo on Curiosity: one B300 GPU for 2 hours, a shell on that node with Ollama running and the
# `clusterpilot` command (latest dev from ~/cp/demo_app). Run on the login node:  bash ~/cp/bin/cp_demo.sh
# Uses 1 of the team s 8 GPUs only while the shell is open; type `exit` to give it back.
source /etc/profile.d/modules.sh; module load slurm >/dev/null 2>&1
[ -x $HOME/cp/ollama/bin/ollama ] || { echo "Ollama is not installed in ~/cp/ollama yet (see the install command)"; exit 1; }
exec srun --partition=hackathon --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=02:00:00 --job-name=cp-demo \
    --pty bash --rcfile $HOME/cp/bin/cp_demo_rc.sh -i
