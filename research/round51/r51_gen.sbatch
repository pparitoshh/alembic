#!/bin/bash
#SBATCH --job-name=dk-r51-gen
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=2-5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
# Round 51 step 2: round 27 answers every question with the excerpts. One task per node, 4 vLLM servers per node (one per
# GPU), node k answers questions k, k+n, ...; the parts are appended to the main files at the end (resumable).
source "${REPO:?}/research/rag65/env.sh"
export RD=${RD:?} NSHARDS=$SLURM_NNODES
timeout 1560 srun --ntasks-per-node=1 bash -c '  # stop 4 min before the limit so the parts are merged
source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"
P=$(( 24000 + (SLURM_JOB_ID % 400) * 10 )); EP=""
for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g start_vllm /leonardo_work/EUHPC_D30_031/shared_split/models/r27_merged $((P + g)) \
        --served-model-name r27 --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.88
    EP="$EP,http://127.0.0.1:$((P + g))/v1"
done
SHARD=$SLURM_NODEID "$VENV/bin/python" "$REPO/../ctl/r51_gen.py" "$RD" "${EP#,}" r27'
# append the parts to the main files (ids already there are skipped), then remove them
"$VENV/bin/python" - "$RD" <<'PY'
import glob, json, os, sys
rd = sys.argv[1]
have = set()
if os.path.exists(f'{rd}/answers/d1__rag.jsonl'):
    have = {json.loads(l)['id'] for l in open(f'{rd}/answers/d1__rag.jsonl') if l.strip()}
for part in sorted(glob.glob(f'{rd}/answers/part*_d1__rag.jsonl')):
    p = os.path.basename(part).split('_')[0] + '_'
    files = [(f'{rd}/eval/{p}d1.jsonl', f'{rd}/eval/d1.jsonl'), (f'{rd}/answers/{p}d1__rag.jsonl', f'{rd}/answers/d1__rag.jsonl'),
             (f'{rd}/answers/{p}d1__base_closed.jsonl', f'{rd}/answers/d1__base_closed.jsonl')]
    rows = [[json.loads(l) for l in open(src) if l.strip()] for src, _ in files]
    ids = [{r['id'] for r in rs} for rs in rows]
    keep = ids[0] & ids[1] & ids[2] - have
    for (src, dst), rs in zip(files, rows):
        with open(dst, 'a') as f:
            for r in rs:
                if r['id'] in keep:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        os.remove(src)
    have |= keep
print(f'MERGED: {len(have)} answers')
PY
