#!/bin/bash
# Laptop files of the round-11 release (logit distillation on the round-5 set, seed 44, x0.5): merge, bf16/Q8_0/
# Q4_K_M GGUF on CPU, imatrix, Q4_K_M+imatrix, then Q8_0 and Q4_K_M+imatrix answer the eval sets on llama.cpp and are
# judged in one run with the same model on vLLM (runs/rag65kd3 answers, s44_x05_rag) and base+RAG, paired.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
HERE="$(cd "$(dirname "$0")" && pwd)"
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
G=runs/gguf_r11
E=runs/gguf_r11_eval
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
cd $C
mkdir -p $E/answers $G
if [ ! -s $G/model-Q4_K_M.gguf ]; then
    log "merge + GGUF export (CPU partition)"
    srun -A EUHPC_D30_031 -p lrd_all_serial -c 4 --mem=30G -t 02:00:00 -J dk-gguf-r11 bash -c "export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1; [ -s $G/merged/config.json ] || $VPY research/dpo/merge.py runs/rag65kd3/train/seed44/adapter_x05 $G/merged && MERGED=$C/$G/merged OUT=$C/$G bash research/gguf/export.sh" > "$HERE/export.out" 2>&1
    test -s $G/model-Q4_K_M.gguf || { log "STOP: export failed: $(tail -3 "$HERE/export.out")"; exit 1; }
fi
cp -n runs/gguf/calibration.txt $G/calibration.txt 2>/dev/null
for attempt in 1 2 3; do
    test -s $G/model-Q4_K_M-imat.gguf && break
    log "imatrix attempt $attempt"
    REPO=$C OUT=$G bash $R/ctl/priority_submit.sh --export=ALL,OUT=$G research/gguf/imatrix.sbatch
done
test -s $G/model-Q4_K_M-imat.gguf || { log "STOP: no imatrix quant"; exit 1; }
cnt() { cat $E/answers/*__r11_gguf_*_rag.jsonl 2>/dev/null | wc -l; }
for attempt in 1 2 3; do
    [ "$(cnt)" -ge 744 ] && break
    log "answer attempt $attempt ($(cnt)/744)"
    REPO=$C bash $R/ctl/priority_submit.sh --export=ALL,QUANTS="Q8_0 Q4_K_M-imat",ARM=r11_gguf,GGUF_DIR=$G,ANSWERS=$E/answers research/gguf/answer.sbatch
done
[ "$(cnt)" -ge 744 ] || { log "STOP: only $(cnt)/744 answers"; exit 1; }
cp -n runs/rag65kd3/answers/*__base_closed.jsonl runs/rag65kd3/answers/*__base_rag.jsonl $E/answers/
for f in runs/rag65kd3/answers/*__s44_x05_rag.jsonl; do b=$(basename $f); cp -n $f $E/answers/${b/__s44_x05_rag/__r11_vllm_rag}; done
left() { $VPY - <<'PY' | tail -1
from pathlib import Path
a = Path('runs/gguf_r11_eval/answers'); j = Path('runs/gguf_r11_eval/judged')
need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
for attempt in 1 2 3; do
    [ "$(left)" = 0 ] && break
    log "judge attempt $attempt ($(left) units left)"
    REPO=$C RUN=gguf_r11_eval bash $R/ctl/priority_submit.sh --nodes=2 --export=ALL,STAGE=judge,RUN=gguf_r11_eval research/rag65/stage.sbatch
done
[ "$(left)" = 0 ] || { log "STOP: $(left) judge units left"; exit 1; }
export PYTHONPATH=src:research/rag65
$VPY research/rag65/aggregate.py $E/judged $E/answers $E/RESULT.json > $E/aggregate.out 2>&1
$VPY research/rag65/paired.py $E/judged $E/PAIRED.json r11_gguf_q80_rag:r11_vllm_rag r11_gguf_q4kmimat_rag:r11_vllm_rag \
    r11_gguf_q4kmimat_rag:base_rag > $E/paired.out 2>&1
log "DONE: $(python3 -c "
import json
d=json.load(open('$E/PAIRED.json'))
print('; '.join(f\"{k} {v['a']:.3f} vs {v['b']:.3f} ({v['diff']:+.3f}, p={v['p_a_not_better']:.3f})\" for k,v in d.items()))")"
