#!/bin/bash
# Upload check: a finished round's seeds head to head with the current Hugging Face release (12e = runs/rag65kd9
# seed 44, full strength) in ONE judge run, with both judges (gpt-oss-20b, then Gemma 4 26B), plus base+RAG.
# usage: vs_release.sh ROUND_DIR (e.g. rft1)   log: ctl/vs_release_<ROUND>.log; results runs/vs_release_<ROUND>/
set -uo pipefail
RD=$1
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=vs_release_$RD; LOG=$R/ctl/$OUT.log; log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
log "waiting for $RD to finish"
until grep -q ' DONE: \| STOP: ' $R/ctl/$RD/controller.log 2>/dev/null; do sleep 120; done
grep -q ' STOP: ' $R/ctl/$RD/controller.log && { log "STOP: $RD stopped"; exit 1; }
cd $C; A=runs/$OUT/answers; mkdir -p $A
cp -n runs/rag65kd9/answers/*__base_closed.jsonl runs/rag65kd9/answers/*__base_rag.jsonl $A/
for f in runs/rag65kd9/answers/*__s44_rag.jsonl; do b=$(basename $f); cp -n $f $A/${b/__s44_rag/__rel12e_rag}; done
for f in runs/$RD/answers/*__s4?_rag.jsonl runs/$RD/answers/*__s4?_x05_rag.jsonl; do [ -s $f ] && cp -n $f $A/; done
left() { $VPY - <<PY | tail -1
from pathlib import Path
a = Path('runs/$OUT/answers'); j = Path('runs/$OUT/$1'); need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
for attempt in 1 2 3 4; do
    [ "$(left judged)" = 0 ] && break
    log "gpt-oss judge attempt $attempt ($(left judged) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch
done
for attempt in 1 2 3 4; do
    [ "$(left judged_gemma)" = 0 ] && break
    log "gemma judge attempt $attempt ($(left judged_gemma) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=3 --export=ALL,RUN=$OUT research/teacher_probe/judge_gemma.sbatch
done
pairs=""; for f in $A/v3__s4?_rag.jsonl $A/v3__s4?_x05_rag.jsonl; do [ -s $f ] || continue; a=$(basename $f .jsonl); a=${a#v3__}; pairs="$pairs $a:rel12e_rag"; done
pairs="$pairs rel12e_rag:base_rag"
export PYTHONPATH=src:research/rag65
$VPY research/rag65/paired.py runs/$OUT/judged runs/$OUT/PAIRED.json $pairs > runs/$OUT/paired.out 2>&1
$VPY research/rag65/paired.py runs/$OUT/judged_gemma runs/$OUT/PAIRED_gemma.json $pairs > runs/$OUT/paired_gemma.out 2>&1
log "DONE: $(python3 -c "
import json
for name in ['PAIRED', 'PAIRED_gemma']:
    d = json.load(open('runs/$OUT/' + name + '.json'))
    print(name, '; '.join(f\"{k} {v['diff']:+.3f}\" for k, v in d.items() if 'rel12e_rag' in k.split('-vs-')[1]))
")"
