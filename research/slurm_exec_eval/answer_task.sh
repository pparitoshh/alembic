#!/bin/bash
# Curiosity task: every model answers the 30 execution-eval tasks (round 27, base Qwen3-4B, the newest B300 model if
# any, gpt-oss-20b and Qwen3-235B as references). Answers go to runs/slurm_exec/<model>.jsonl for grading on Leonardo.
source $HOME/cp/bin/lib.sh; TASK=slurm_exec; cd $REPO
O=$CP/runs/slurm_exec; mkdir -p $O; T=$CP/bin/slurm_exec_tasks.jsonl
serve 0 8200 $M/r27 r27 --max-model-len 8192 --gpu-memory-utilization 0.85 && python $CP/bin/answer_tasks.py $T http://127.0.0.1:8200/v1 r27 $O/r27.jsonl &
serve 1 8201 $M/qwen3-4b qwen3-4b --max-model-len 8192 --gpu-memory-utilization 0.85 && python $CP/bin/answer_tasks.py $T http://127.0.0.1:8201/v1 qwen3-4b $O/qwen3-4b.jsonl &
serve 2 8202 $M/gptoss20 gpt-oss-20b --max-model-len 8192 --gpu-memory-utilization 0.85 && python $CP/bin/answer_tasks.py $T http://127.0.0.1:8202/v1 gpt-oss-20b $O/gpt-oss-20b.jsonl &
[ -s $CP/exp/kd1/model/model.safetensors ] && { serve 3 8203 $CP/exp/kd1/model kd1 --max-model-len 8192 --gpu-memory-utilization 0.85 && python $CP/bin/answer_tasks.py $T http://127.0.0.1:8203/v1 kd1 $O/kd1.jsonl; } &
serve 4,5,6,7 8204 $M/q235fp8 qwen3-235b --max-model-len 8192 --gpu-memory-utilization 0.90 && python $CP/bin/answer_tasks.py $T http://127.0.0.1:8204/v1 qwen3-235b $O/qwen3-235b.jsonl &
wait
log "DONE: execution-eval answers: $(ls $O | tr '\n' ' ')"
