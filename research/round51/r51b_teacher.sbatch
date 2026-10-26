#!/bin/bash
#SBATCH --job-name=dk-r51b-teacher
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=3-6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Round 51b step 2: Qwen3-235B (AWQ, TP4 per node, checkpoint staged in node RAM) answers the round's questions; node k
# answers questions k, k+n, ...; stops 7 minutes before the limit (the step takes up to 3 minutes to die) and appends the
# parts to answers/d1__base_closed.jsonl (the controller also merges leftover parts). Loading takes ~13 minutes.
source "${REPO:?}/research/rag65/env.sh"
export RD=${RD:?} NSHARDS=$SLURM_NNODES
timeout 1380 srun --kill-on-bad-exit=0 bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
    SNAP=$(ls -d "$HF_HOME"/hub/models--QuantTrio--Qwen3-235B-A22B-Instruct-2507-AWQ/snapshots/* | head -1)
    STAGE=/dev/shm/q235-$SLURM_JOB_ID; mkdir -p $STAGE
    for f in "$SNAP"/*; do cp -rL "$f" $STAGE/ & done; wait
    trap "rm -rf $STAGE" EXIT
    start_vllm $STAGE $P --tensor-parallel-size 4 --max-model-len 16384 --gpu-memory-utilization 0.92 --served-model-name qwen3-235b
    SHARD=$SLURM_NODEID "$VENV/bin/python" "$REPO/../ctl/r51b_teacher.py" "$RD" "http://127.0.0.1:$P/v1" qwen3-235b'
"$VENV/bin/python" - "$RD" <<'PY'
import glob, json, os, sys
rd = sys.argv[1]
dst = f'{rd}/answers/d1__base_closed.jsonl'
have = {json.loads(l)['id'] for l in open(dst) if l.strip()} if os.path.exists(dst) else set()
with open(dst, 'a') as fo:
    for part in sorted(glob.glob(f'{rd}/answers/part*_teacher.jsonl')):
        for l in open(part):
            if l.strip():
                r = json.loads(l)
                if r['id'] not in have:
                    fo.write(json.dumps(r, ensure_ascii=False) + '\n')
                    have.add(r['id'])
        os.remove(part)
print(f'MERGED: {len(have)} teacher answers')
PY
