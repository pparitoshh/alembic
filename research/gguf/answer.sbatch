#!/bin/bash
#SBATCH --job-name=dk-gguf-answer
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Quantization cost (GOAL.md G3/G6): the release model's GGUF files (bf16, Q8_0, Q4_K_M, Q3_K_M), one per GPU on
# llama.cpp's server, answer the eval sets with retrieval exactly like every student arm (same system prompt,
# greedy, 768-token budget). The bf16 GGUF on the same runtime isolates the effect of quantization. Resumable.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --gpu-bind=single:1 --gpus-per-task=1 --output="$L/gguf-answer-%j-%t.out" --error="$L/gguf-answer-%j-%t.err" \
    bash -c 'source "$REPO/research/rag65/env.sh"; cd "$REPO"; export PYTHONPATH=src:research/rag65
        module load cuda/12.6 > /dev/null 2>&1
        B=/leonardo_work/EUHPC_D30_031/runs/a08trc09/published-hf-comparison-20261015/llama_cuda/bin
        export LD_LIBRARY_PATH=$B:${LD_LIBRARY_PATH:-}
        Q=(${QUANTS:-bf16 Q8_0 Q4_K_M Q3_K_M}); (( SLURM_PROCID < ${#Q[@]} )) || exit 0  # QUANTS: up to 4 files, one per GPU
        q=${Q[$SLURM_PROCID]}; arm=${ARM:-s43_gguf}_$(echo $q | tr -d _- | tr A-Z a-z)
        P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 + SLURM_LOCALID * 4 ))
        # 16 slots x 16K tokens: every prompt (sources + question) plus the 768-token answer fits in one slot.
        $B/llama-server -m ${GGUF_DIR:-runs/gguf}/model-$q.gguf --host 127.0.0.1 --port $P -ngl 99 -c 262144 --parallel 16 \
            -fa on --jinja --no-webui > "$REPO/../logs/gguf-server-$SLURM_JOB_ID-$q.log" 2>&1 &
        until curl -sf http://127.0.0.1:$P/health > /dev/null; do
            kill -0 $! 2>/dev/null || { tail -20 "$REPO/../logs/gguf-server-$SLURM_JOB_ID-$q.log"; exit 1; }; sleep 5; done
        "$VENV/bin/python" research/teacher_probe/answer_teacher.py $arm "http://127.0.0.1:$P/v1" $arm ${ANSWERS:-runs/gguf_eval/answers} 0 rag
        kill %1'
