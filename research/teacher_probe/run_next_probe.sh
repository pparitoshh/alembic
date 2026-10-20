#!/bin/bash
# Unattended Qwen3-Next teacher probe: wait for the download, answer the eval sets (resubmitting until all
# 372 exist), judge with base, base+RAG and the 32B teacher in one run, then write the paired comparison.
# Each GPU job goes through ctl/priority_submit.sh, which never takes a round's slot between training pieces.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=$C/runs/teacher_probe2
LOG=$R/ctl/teacher_probe2.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
M=/leonardo_work/EUHPC_D30_031/hf_cache/hub/models--cyankiwi--Qwen3-Next-80B-A3B-Instruct-AWQ-4bit
mkdir -p "$OUT/answers"
cp -n $C/runs/teacher_probe/answers/*__base_closed.jsonl $C/runs/teacher_probe/answers/*__base_rag.jsonl \
      $C/runs/teacher_probe/answers/*__t32_rag.jsonl "$OUT/answers/"
log "waiting for the download"
until S=$(ls -d $M/snapshots/* 2>/dev/null | head -1) && [ -n "$S" ] && [ -f "$S/model.safetensors.index.json" ] && \
      python3 -c "import json,os,sys;i=json.load(open('$S/model.safetensors.index.json'));sys.exit(any(not os.path.exists('$S/'+f) for f in set(i['weight_map'].values())))"; do
    sleep 120
done
log "download complete: $S"
count() { cat "$OUT"/answers/*__tnext_rag.jsonl 2>/dev/null | wc -l; }
for attempt in 1 2 3 4; do
    [ "$(count)" -ge 372 ] && break
    log "answer attempt $attempt ($(count)/372)"
    (cd $C && REPO=$C bash $R/ctl/priority_submit.sh research/teacher_probe/probe_next.sbatch)
done
[ "$(count)" -ge 372 ] || { log "STOP: only $(count)/372 answers"; exit 1; }
log "answers complete; judging"
(cd $C && REPO=$C RUN=teacher_probe2 bash $R/ctl/priority_submit.sh --nodes=2 --export=ALL,STAGE=judge,RUN=teacher_probe2 research/rag65/stage.sbatch)
cd $C && PYTHONPATH=src:research/rag65 $VPY research/rag65/paired.py "$OUT/judged" "$OUT/PAIRED.json" \
    tnext_rag:t32_rag tnext_rag:base_rag t32_rag:base_rag > "$OUT/paired.out" 2>&1
log "DONE: $(tr -d '\n' < "$OUT/paired.out" | head -c 1500)"
