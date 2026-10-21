#!/bin/bash
#SBATCH --job-name=dk-rft-judge
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Rejection sampling against the teacher, step 2: one gpt-oss-20b judge per GPU running
# research/rft/judge_vs_teacher.py on DATA (teacher answers) and CANDS (student candidates) into OUT. Resumable.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --gpu-bind=single:1 --gpus-per-task=1 --output="$L/rft-judge-%j-%t.out" --error="$L/rft-judge-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src:research/rag65
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 + SLURM_LOCALID * 4 ))
    start_vllm openai/gpt-oss-20b $P $(judge_args openai/gpt-oss-20b)
    "$VENV/bin/python" research/rft/judge_vs_teacher.py $SLURM_PROCID $SLURM_NTASKS "http://127.0.0.1:$P/v1" "${DATA:?}" "${CANDS:?}" "${OUT:?}"'
