#!/bin/bash
# Quantization check for the round 27 release: importance-matrix Q4_K_M, GGUF answers (Q8_0, Q4_K_M-imat) and one
# gpt-oss judge run against the full-precision answers (as auto_release.sh does). Plain low-priority sbatch jobs (no
# ctl/PRIORITY), so score rounds are never held up.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
G=runs/gguf_soup27; E=runs/gguf_soup27_eval; LOG=$R/ctl/gguf_soup27_eval.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run() { local j; until j=$(cd $C && sbatch --parsable --nice=800 "$@" 2>/dev/null | tail -1) && [[ "$j" =~ ^[0-9]+$ ]]; do sleep 120; done
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $E/answers
cp -n runs/gguf/calibration.txt $G/calibration.txt 2>/dev/null
for a in 1 2 3; do test -s $G/model-Q4_K_M-imat.gguf && break; log "imatrix attempt $a"; REPO=$C OUT=$G run --export=ALL,OUT=$G research/gguf/imatrix.sbatch; done
test -s $G/model-Q4_K_M-imat.gguf || { log "STOP: no imatrix quant"; exit 1; }
cnt() { cat $E/answers/*__cand_gguf_*_rag.jsonl 2>/dev/null | wc -l; }
for a in 1 2 3; do [ "$(cnt)" -ge 744 ] && break; log "gguf answer attempt $a ($(cnt)/744)"
    REPO=$C run --export=ALL,QUANTS="Q8_0 Q4_K_M-imat",ARM=cand_gguf,GGUF_DIR=$G,ANSWERS=$E/answers research/gguf/answer.sbatch; done
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
for a in 1 2 3; do [ "$(left)" = 0 ] && break; log "gguf judge attempt $a ($(left) units)"
    REPO=$C RUN=gguf_soup27_eval run --nodes=2 --export=ALL,STAGE=judge,RUN=gguf_soup27_eval research/rag65/stage.sbatch; done
export PYTHONPATH=src:research/rag65
$VPY research/rag65/aggregate.py $E/judged $E/answers $E/RESULT.json > $E/aggregate.out 2>&1
$VPY research/rag65/paired.py $E/judged $E/PAIRED.json cand_gguf_q80_rag:cand_vllm_rag cand_gguf_q4kmimat_rag:cand_vllm_rag cand_gguf_q4kmimat_rag:base_rag > $E/paired.out 2>&1
log "DONE: $(python3 -c "
import json; a=json.load(open('$E/RESULT.json'))['arms']
print('; '.join(f\"{k} {v['win_rate']:.3f}\" for k,v in sorted(a.items())))")"
