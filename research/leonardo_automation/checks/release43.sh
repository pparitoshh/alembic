#!/bin/bash
# Round 43 release prep: (1) GGUF files of the soup43 merged model (bf16 -> Q8_0, Q6_K, Q4_K_M with a fresh importance
# matrix), (2) the release gate with the DEFAULT retriever: soup43 answers (runs/soup43old) judged by ctl/vs_release.sh
# against round 27, exactly as every earlier release check. Log: ctl/release43.log
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; LOG=$R/ctl/release43.log; G=runs/gguf_soup43; M=$C/runs/soup43/merged
QUANT=/leonardo_work/EUHPC_D30_031/tools/llama.cpp/bin/llama-quantize
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run() { local j; until j=$(cd $C && sbatch --parsable "$@" 2>/dev/null | tail -1) && [[ "$j" =~ ^[0-9]+$ ]]; do sleep 120; done
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $G runs/soup43old/answers $R/ctl/soup43old
gate() {  # default-retriever answers, then the standard release check
    unset RAG_VARIANT; export BASE_MODEL=$M
    cnt() { cat runs/soup43old/answers/v2__base_rag.jsonl runs/soup43old/answers/v3__base_rag.jsonl runs/soup43old/answers/d1__base_rag.jsonl 2>/dev/null | wc -l; }
    for a in 1 2 3 4; do [ "$(cnt)" -ge 372 ] && break
        REPO=$C RUN=soup43old ANSWER_MODES=rag VARIANTS=none BASE_SERVED=Qwen/Qwen3-4B-Instruct-2507 run --nodes=1 --job-name=dk-soup43old --export=ALL,STAGE=answer research/rag65/stage.sbatch; done
    [ "$(cnt)" -ge 372 ] || { echo "$(date -u +%FT%TZ) STOP: answers incomplete" >> $R/ctl/soup43old/controller.log; return 1; }
    for l in v2 v3 d1; do cp runs/soup43old/answers/${l}__base_rag.jsonl runs/soup43old/answers/${l}__s42_rag.jsonl; rm -f runs/soup43old/answers/${l}__base_rag.jsonl runs/soup43old/answers/${l}__base_closed.jsonl; done
    echo "$(date -u +%FT%TZ) DONE: answers ready (arm s42_rag, default retriever)" >> $R/ctl/soup43old/controller.log
}
gate & gp=$!
[ -s $G/model-Q8_0.gguf ] || srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 01:30:00 -J dk-gguf43 bash -c "MERGED=$M OUT=$C/$G bash research/gguf/export.sh" > $R/ctl/gguf_soup43_export.out 2>&1
log "export: $(cat $G/EXPORT.json 2>/dev/null | tr -d '\n ')"
cp -n runs/gguf/calibration.txt $G/calibration.txt 2>/dev/null
for a in 1 2 3; do test -s $G/model-Q4_K_M-imat.gguf && break; log "imatrix attempt $a"; REPO=$C OUT=$G run --export=ALL,OUT=$G research/gguf/imatrix.sbatch; done
[ -s $G/model-Q6_K.gguf ] || srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 00:40:00 -J dk-q6k43 $QUANT $G/model-bf16.gguf $G/model-Q6_K.gguf Q6_K 8 >> $R/ctl/gguf_soup43_export.out 2>&1
log "files: $(ls -la $G/*.gguf | awk '{printf "%s %.2fGB; ", $9, $5/1e9}')"
wait $gp
log "DONE: release files ready; gate answers in runs/soup43old (judged by ctl/vs_release.sh soup43old)"
