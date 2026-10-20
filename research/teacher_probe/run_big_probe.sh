#!/bin/bash
# Unattended bigger-teacher probe: Qwen3-235B-A22B-2507 and gpt-oss-120b answer the eval sets with retrieval, then
# one judge run with Qwen3-Next (re-judged), base and base+RAG, paired. Switch teachers only on a clear win.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=$C/runs/teacher_probe4
LOG=$R/ctl/teacher_probe4.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
mkdir -p "$OUT/answers"
cp -n $C/runs/teacher_probe2/answers/*__base_closed.jsonl $C/runs/teacher_probe2/answers/*__base_rag.jsonl \
      $C/runs/teacher_probe2/answers/*__tnext_rag.jsonl "$OUT/answers/"
count() { cat "$OUT"/answers/*__$1_rag.jsonl 2>/dev/null | wc -l; }
for t in t235 goss120; do
    if [ $t = goss120 ]; then  # wait for the download (index + every shard present)
        M=/leonardo_work/EUHPC_D30_031/hf_cache/hub/models--openai--gpt-oss-120b
        until S=$(ls -d $M/snapshots/* 2>/dev/null | head -1) && [ -n "$S" ] && [ -f "$S/model.safetensors.index.json" ] && \
              python3 -c "import json,os,sys;i=json.load(open('$S/model.safetensors.index.json'));sys.exit(any(not os.path.exists('$S/'+f) for f in set(i['weight_map'].values())))"; do
            sleep 120
        done
    fi
    for attempt in 1 2 3; do
        [ "$(count $t)" -ge 372 ] && break
        log "$t answer attempt $attempt ($(count $t)/372)"
        (cd $C && REPO=$C TEACHER=$t bash $R/ctl/priority_submit.sh --export=ALL,TEACHER=$t research/teacher_probe/probe_big.sbatch)
    done
    log "$t: $(count $t)/372 answers"
done
log "judging"
left() { (cd $C && $VPY - <<'PY' | tail -1
from pathlib import Path
a = Path('runs/teacher_probe4/answers'); j = Path('runs/teacher_probe4/judged')
need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
); }
for attempt in 1 2 3; do
    [ "$(left)" = 0 ] && break
    (cd $C && REPO=$C RUN=teacher_probe4 bash $R/ctl/priority_submit.sh --nodes=2 --export=ALL,STAGE=judge,RUN=teacher_probe4 research/rag65/stage.sbatch)
done
cd $C && PYTHONPATH=src:research/rag65 $VPY research/rag65/paired.py "$OUT/judged" "$OUT/PAIRED.json" \
    t235_rag:tnext_rag goss120_rag:tnext_rag tnext_rag:base_rag > "$OUT/paired.out" 2>&1
log "DONE: $(tr -d '\n' < "$OUT/paired.out" | head -c 2000)"
