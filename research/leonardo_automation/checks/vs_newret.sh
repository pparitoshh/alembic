#!/bin/bash
# Head to head with round 27 answered with the SAME improved retriever (exp30 s42_rag = soup27 + RAG_VARIANT=b), for
# rounds trained on new-retriever contexts (rb31-rb33, run under shared_split/ctl). Same three judges + 2-of-3 majority
# and paired stats as ctl/vs_release.sh; arm rel_rag here = round 27 with RAG_VARIANT=b.
# usage: vs_newret.sh ROUND (e.g. rb31)   log: ctl/vsb_<ROUND>.log; results runs/vsb_<ROUND>/
set -uo pipefail
RD=$1
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=vsb_$RD; S=/leonardo_work/EUHPC_D30_031/shared_split; LOG=$R/ctl/$OUT.log; log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
log "waiting for $RD to finish"
until grep -q ' DONE: \| STOP: ' $S/ctl/$RD/controller.log 2>/dev/null; do sleep 120; done
grep -q ' STOP: ' $S/ctl/$RD/controller.log && { log "STOP: $RD stopped"; exit 1; }
cd $C; A=runs/$OUT/answers; mkdir -p $A
# The current Hugging Face release is named in ctl/RELEASE ("<answers dir> <arm>"; Oct 22: round 20 = op235 s44 x0.5).
RA=${REL_ANSWERS:-runs/exp30/answers}; RARM=${REL_ARM:-s42_rag}; BA=runs/vs_release_exp30/answers  # REL_ANSWERS/REL_ARM (opt-in): another reference, e.g. runs/tok1536/answers base_rag
cp -n $BA/*__base_closed.jsonl $BA/*__base_rag.jsonl $A/
for f in $RA/*__${RARM}.jsonl; do b=$(basename $f); cp -n $f $A/${b/__${RARM}/__rel_rag}; done
for f in $S/code/runs/$RD/answers/*__s4?_rag.jsonl $S/code/runs/$RD/answers/*__s4?_x05_rag.jsonl; do [ -s $f ] && cp -n $f $A/; done
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
# Speed (Oct 22): the three judges run in parallel.
( for attempt in 1 2 3 4; do
    [ "$(left judged)" = 0 ] && break
    log "gpt-oss judge attempt $attempt ($(left judged) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch
done ) &
( for attempt in 1 2 3 4; do
    [ "$(left judged_gemma)" = 0 ] && break
    log "gemma judge attempt $attempt ($(left judged_gemma) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT research/teacher_probe/judge_gemma.sbatch
done ) &
( for attempt in 1 2 3 4; do
    [ "$(left judged_qwen)" = 0 ] && break
    log "qwen3-235b judge attempt $attempt ($(left judged_qwen) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT research/teacher_probe/judge_qwen235.sbatch
done ) &
wait
$VPY research/rag65/majority.py runs/$OUT >> $LOG 2>&1  # 2-of-3 vote of gpt-oss-20b, Gemma 4 and Qwen3-235B (Oct 22)
pairs=""; for f in $A/v3__s4?_rag.jsonl $A/v3__s4?_x05_rag.jsonl; do [ -s $f ] || continue; a=$(basename $f .jsonl); a=${a#v3__}; pairs="$pairs $a:rel_rag"; done
pairs="$pairs rel_rag:base_rag"
export PYTHONPATH=src:research/rag65
$VPY research/rag65/paired.py runs/$OUT/judged runs/$OUT/PAIRED.json $pairs > runs/$OUT/paired.out 2>&1
$VPY research/rag65/paired.py runs/$OUT/judged_gemma runs/$OUT/PAIRED_gemma.json $pairs > runs/$OUT/paired_gemma.out 2>&1
$VPY research/rag65/paired.py runs/$OUT/judged_qwen runs/$OUT/PAIRED_qwen.json $pairs > runs/$OUT/paired_qwen.out 2>&1
$VPY research/rag65/paired.py runs/$OUT/judged_majority runs/$OUT/PAIRED_majority.json $pairs > runs/$OUT/paired_majority.out 2>&1
log "DONE: $(python3 -c "
import json
for name in ['PAIRED', 'PAIRED_gemma', 'PAIRED_qwen', 'PAIRED_majority']:
    d = json.load(open('runs/$OUT/' + name + '.json'))
    print(name, '; '.join(f\"{k} {v['diff']:+.3f}\" for k, v in d.items() if 'rel_rag' in k.split('-vs-')[1]))
")"
