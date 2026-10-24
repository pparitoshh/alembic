#!/bin/bash
# GOAL.md "Q4_K_M vs bf16 drop <= 2 pp" follow-up: round 27 as Q5_K_M (importance matrix) and Q6_K, judged in ONE
# gpt-oss run against the full-precision vLLM answers together with Q4_K_M-imat and Q8_0, so the laptop-size /
# accuracy trade-off has five points. Quantized on the CPU partition from runs/gguf_soup27/model-bf16.gguf.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
QUANT=/leonardo_work/EUHPC_D30_031/tools/llama.cpp/bin/llama-quantize
G=runs/gguf_soup27; E=runs/gguf_q56_eval; LOG=$R/ctl/gguf_q56_eval.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run() { local j; until j=$(cd $C && sbatch --parsable "$@" 2>/dev/null | tail -1) && [[ "$j" =~ ^[0-9]+$ ]]; do sleep 120; done
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $E/answers
if [ ! -s $G/model-Q6_K.gguf ] || [ ! -s $G/model-Q5_K_M-imat.gguf ]; then
    log "quantizing Q5_K_M-imat and Q6_K (CPU partition)"
    srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 01:30:00 -J dk-q56 bash -c "
        [ -s $G/model-Q5_K_M-imat.gguf ] || $QUANT --imatrix $G/imatrix.gguf $G/model-bf16.gguf $G/model-Q5_K_M-imat.gguf Q5_K_M 8
        [ -s $G/model-Q6_K.gguf ] || $QUANT $G/model-bf16.gguf $G/model-Q6_K.gguf Q6_K 8" > $R/ctl/gguf_q56_quant.out 2>&1
fi
test -s $G/model-Q6_K.gguf && test -s $G/model-Q5_K_M-imat.gguf || { log "STOP: quantization failed"; exit 1; }
log "sizes: $(ls -la $G/model-Q6_K.gguf $G/model-Q5_K_M-imat.gguf | awk '{printf "%s %.2f GB; ", $9, $5/2^30}')"
cnt() { cat $E/answers/*__cand_gguf_*_rag.jsonl 2>/dev/null | wc -l; }
for a in 1 2 3; do [ "$(cnt)" -ge 1488 ] && break; log "gguf answer attempt $a ($(cnt)/1488)"
    REPO=$C run --export=ALL,QUANTS="Q8_0 Q6_K Q5_K_M-imat Q4_K_M-imat",ARM=cand_gguf,GGUF_DIR=$G,ANSWERS=$E/answers research/gguf/answer.sbatch; done
S=runs/vs_release_soup27/answers
cp -n $S/*__base_closed.jsonl $S/*__base_rag.jsonl $E/answers/
for l in v2 v3 d1; do cp -n $S/${l}__s42_rag.jsonl $E/answers/${l}__cand_vllm_rag.jsonl; done
left() { $VPY - <<PY | tail -1
from pathlib import Path
a = Path('$E/answers'); j = Path('$E/judged'); need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
for a in 1 2 3 4; do [ "$(left)" = 0 ] && break; log "gguf judge attempt $a ($(left) units)"
    REPO=$C RUN=gguf_q56_eval run --nodes=4 --export=ALL,STAGE=judge,RUN=gguf_q56_eval research/rag65/stage.sbatch; done
export PYTHONPATH=src:research/rag65
$VPY research/rag65/aggregate.py $E/judged $E/answers $E/RESULT.json > $E/aggregate.out 2>&1
$VPY research/rag65/paired.py $E/judged $E/PAIRED.json cand_gguf_q80_rag:cand_vllm_rag cand_gguf_q6k_rag:cand_vllm_rag cand_gguf_q5kmimat_rag:cand_vllm_rag cand_gguf_q4kmimat_rag:cand_vllm_rag > $E/paired.out 2>&1
log "DONE: $(python3 -c "
import json; a=json.load(open('$E/RESULT.json'))['arms']
print('; '.join(f\"{k} {v['win_rate']:.3f}\" for k,v in sorted(a.items())))")"
