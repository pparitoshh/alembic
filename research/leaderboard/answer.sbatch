#!/bin/bash
#SBATCH --job-name=dk-lb-answer
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Open-model leaderboard, answer step. MODELS="arm=hf/model:tp:thinking ..." on one node: models with tp 1 run side
# by side (one GPU each), a model with tp 4 takes the node. Each answers v2/v3/d1 exactly like the student arms
# (research/teacher_probe/answer_teacher.py: same retrieved excerpts, system prompt, greedy decoding) into OUT.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --output="$L/lb-answer-%j.out" --error="$L/lb-answer-%j.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src:research/rag65
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 )); gpu=0
    for spec in ${MODELS:?}; do
        arm=${spec%%=*}; rest=${spec#*=}; model=${rest%%:*}; tp=$(echo $rest | cut -d: -f2); think=$(echo $rest | cut -d: -f3)
        if [ "$tp" = 1 ]; then devs=$gpu; else devs=0,1,2,3; fi
        ( export CUDA_VISIBLE_DEVICES=$devs
          start_vllm "$model" $((P + gpu * 4)) --tensor-parallel-size $tp --max-model-len 16384 --gpu-memory-utilization 0.88 --served-model-name $arm
          "$VENV/bin/python" research/teacher_probe/answer_teacher.py $arm "http://127.0.0.1:$((P + gpu * 4))/v1" $arm "${OUT:?}" $think rag ) &
        gpu=$((gpu + 1))
    done
    wait'
