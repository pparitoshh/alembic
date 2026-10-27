#!/bin/bash
# Curiosity task: the newest B300 models (kd2 = ClusterPilot-4B 1.2, kd3 if trained) answer the 30 execution-eval
# tasks; graded on Leonardo with research/slurm_exec_eval/grade.py (same setup as 025_slurm_exec).
source $HOME/cp/bin/lib.sh; TASK=slurm_exec2; cd $REPO
O=$CP/runs/slurm_exec; mkdir -p $O; T=$CP/bin/slurm_exec_tasks.jsonl
stop_all
serve 0 8210 $CP/exp/kd2/model kd2 --max-model-len 8192 --gpu-memory-utilization 0.85 && python $CP/bin/answer_tasks.py $T http://127.0.0.1:8210/v1 kd2 $O/kd2.jsonl &
[ -s $CP/exp/kd3/model/model.safetensors ] && { serve 1 8211 $CP/exp/kd3/model kd3 --max-model-len 8192 --gpu-memory-utilization 0.85 && python $CP/bin/answer_tasks.py $T http://127.0.0.1:8211/v1 kd3 $O/kd3.jsonl; } &
wait
stop_all
log "DONE: execution-eval answers: $(ls $O | tr '\n' ' ')"
