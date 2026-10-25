#!/bin/bash
# GOAL.md G6 (accuracy / size frontier, >= 2 student sizes x >= 3 quant levels): the 1.7B student (round 9, seed 42,
# adapter at half strength, its best arm) merged into Qwen3-1.7B, exported to GGUF (bf16, Q8_0, Q4_K_M, Q3_K_M) and
# judged by gpt-oss-20b in ONE run together with round 27's Q8_0 and Q4_K_M answers (from runs/gguf_q56_eval), so
# both sizes are compared on the same scale (win rate vs closed-book base Qwen3-4B).
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
M=runs/gguf_s17/merged; G=runs/gguf_s17; E=runs/gguf_s17_eval; LOG=$R/ctl/gguf_s17_eval.log
BASE17=/leonardo_work/EUHPC_D30_031/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run() { local j; until j=$(cd $C && sbatch --parsable "$@" 2>/dev/null | tail -1) && [[ "$j" =~ ^[0-9]+$ ]]; do sleep 120; done
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $G $E/answers
if [ ! -s $M/config.json ]; then
    log "merging the 1.7B adapter (CPU partition)"
    srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 01:00:00 -J dk-s17merge bash -c "export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1; $VPY - <<PY
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained('$BASE17', dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, 'runs/rag65s17/train/seed42/adapter_x05').merge_and_unload()
m.save_pretrained('$M'); AutoTokenizer.from_pretrained('$BASE17').save_pretrained('$M')
print('merged')
PY" > $R/ctl/gguf_s17_merge.out 2>&1
fi
test -s $M/config.json || { log "STOP: merge failed: $(tail -3 $R/ctl/gguf_s17_merge.out)"; exit 1; }
[ -s $G/model-Q3_K_M.gguf ] || srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 01:00:00 -J dk-s17gguf \
    bash -c "MERGED=$C/$M OUT=$C/$G bash research/gguf/export.sh" > $R/ctl/gguf_s17_export.out 2>&1
test -s $G/model-Q3_K_M.gguf || { log "STOP: export failed: $(tail -3 $R/ctl/gguf_s17_export.out)"; exit 1; }
log "exported: $(cat $G/EXPORT.json | tr -d '\n ')"
cnt() { cat $E/answers/*__s17_gguf_*_rag.jsonl 2>/dev/null | wc -l; }
for a in 1 2 3; do [ "$(cnt)" -ge 1488 ] && break; log "answer attempt $a ($(cnt)/1488)"
    REPO=$C run --export=ALL,QUANTS="bf16 Q8_0 Q4_K_M Q3_K_M",ARM=s17_gguf,GGUF_DIR=$G,ANSWERS=$E/answers research/gguf/answer.sbatch; done
S=runs/vs_release_soup27/answers
cp -n $S/*__base_closed.jsonl $S/*__base_rag.jsonl $E/answers/
until ls runs/gguf_q56_eval/answers/v3__cand_gguf_q4kmimat_rag.jsonl > /dev/null 2>&1; do sleep 120; done
for l in v2 v3 d1; do for q in q80 q4kmimat; do cp -n runs/gguf_q56_eval/answers/${l}__cand_gguf_${q}_rag.jsonl $E/answers/${l}__r27_gguf_${q}_rag.jsonl; done; done
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
for a in 1 2 3 4; do [ "$(left)" = 0 ] && break; log "judge attempt $a ($(left) units)"
    REPO=$C RUN=gguf_s17_eval run --nodes=4 --export=ALL,STAGE=judge,RUN=gguf_s17_eval research/rag65/stage.sbatch; done
export PYTHONPATH=src:research/rag65
$VPY research/rag65/aggregate.py $E/judged $E/answers $E/RESULT.json > $E/aggregate.out 2>&1
log "DONE: $(python3 -c "
import json; a=json.load(open('$E/RESULT.json'))['arms']
print('; '.join(f\"{k} {v['win_rate']:.3f}\" for k,v in sorted(a.items())))")"
