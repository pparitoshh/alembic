#!/bin/bash
# Round 49e (agent model, 10/10 on the agent smoke test): laptop files for its Hugging Face branch (agent-r49e).
# Same steps as release43.sh: GGUF bf16 + Q8_0, imatrix Q4_K_M, Q6_K. Log: ctl/release49e.log
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; LOG=$R/ctl/release49e.log; G=runs/gguf_r49e; M=$C/runs/r49e/merged
QUANT=/leonardo_work/EUHPC_D30_031/tools/llama.cpp/bin/llama-quantize
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run() { local j; until j=$(cd $C && sbatch --parsable "$@" 2>/dev/null | tail -1) && [[ "$j" =~ ^[0-9]+$ ]]; do sleep 120; done
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $G
[ -s $G/model-Q8_0.gguf ] || srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 01:30:00 -J dk-gguf49e bash -c "MERGED=$M OUT=$C/$G bash research/gguf/export.sh" > $R/ctl/gguf_r49e_export.out 2>&1
log "export: $(cat $G/EXPORT.json 2>/dev/null | tr -d "\n ")"
cp -n runs/gguf/calibration.txt $G/calibration.txt 2>/dev/null
for a in 1 2 3; do test -s $G/model-Q4_K_M-imat.gguf && break; log "imatrix attempt $a"; REPO=$C OUT=$G run --export=ALL,OUT=$G research/gguf/imatrix.sbatch; done
[ -s $G/model-Q6_K.gguf ] || srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 00:40:00 -J dk-q6k49e $QUANT $G/model-bf16.gguf $G/model-Q6_K.gguf Q6_K 8 >> $R/ctl/gguf_r49e_export.out 2>&1
log "DONE: files: $(ls -la $G/*.gguf | awk "{printf \"%s %.2fGB; \", \$9, \$5/1e9}")"
