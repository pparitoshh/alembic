#!/bin/bash
# kd1 (B300 full-parameter distillation, now Hugging Face main) on Leonardo's leaderboard pipeline: equal-1,536 answers,
# variant e, 3 judges (gpt-oss-20b, Gemma 4, Qwen3-235B AWQ) + majority, compared with round 27 + e. Log: ctl/kd1_leo.log
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017; S=/leonardo_work/EUHPC_D30_031/shared_split
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
M=$S/models/kd1; LOG=$R/ctl/kd1_leo.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
if [ ! -s $M/model.safetensors ]; then
    log "downloading kd1 from Hugging Face main"
    HF_HUB_OFFLINE=0 HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache $VPY -c "
from huggingface_hub import snapshot_download
snapshot_download('VaibhavKKM/hpc-assistant-qwen3-4b', revision='323f3f8bdffc8bd860050fc10bbf18c480d272a6', local_dir='$M',
                  allow_patterns=['*.safetensors', '*.json', '*.jinja'])" >> $LOG 2>&1 || { log "STOP: download failed"; exit 1; }
    chmod -R g+rX $M
fi
log "model: $(ls $M | tr '\n' ' ')"
cd $R/code
ANSWER_MAX_TOKENS=1536 VARIANT=e bash $R/ctl/ragc.sh kd1q $M
REL_ANSWERS=runs/r27e/answers REL_ARM=base_rag bash $R/ctl/vs_newret.sh kd1q
log "DONE: $(tail -1 $R/ctl/vsb_kd1q.log | cut -c1-200)"
