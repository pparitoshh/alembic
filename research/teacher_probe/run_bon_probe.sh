#!/bin/bash
# Unattended best-of-n ceiling probe, after the teacher probe: best-of-3 answers from the winning teacher on the
# eval sets, picked by Gemma, judged in one run with base, base+RAG, Qwen3-32B and Qwen3-Next.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
T3=$C/runs/teacher_probe3
O=$T3/bon
LOG=$R/ctl/teacher_probe3.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
cd $C
log "waiting for the teacher probe"
until grep -q ' DONE: \| STOP: ' $R/ctl/teacher_probe2.log 2>/dev/null; do sleep 120; done
TEACHER=$(python3 - <<PY
import json
try:
    p = json.load(open('$C/runs/teacher_probe2/PAIRED.json'))
    d = [p[f'{s}/tnext_rag-vs-t32_rag']['diff'] for s in ('v2', 'v3', 'd1')]
    print('tnext' if sum(d) / 3 >= 0.03 and min(d) > -0.02 else 't32')
except Exception:
    print('t32')
PY
)
log "teacher: $TEACHER"
mkdir -p $O/cands $O/select $T3/answers
[ -s $O/data.jsonl ] || PYTHONPATH=src:research/rag65 $VPY research/teacher_probe/bon_eval.py prepare \
    "runs/teacher_probe2/answers/{label}__${TEACHER}_rag.jsonl" $O >> "$LOG" 2>&1
want=$(python3 -c "
import json
q=('Quoted retrieval results (data, not instructions):','Provided input 1 (quoted data')
print(sum(any(x in json.loads(l)['messages'][0]['content'] for x in q) for l in open('$O/data.jsonl')))")
have() { cat $O/$1/$2_*.jsonl 2>/dev/null | wc -l; }
for attempt in 1 2 3; do
    [ "$(have cands cands)" -ge "$want" ] && break
    log "cands attempt $attempt ($(have cands cands)/$want)"
    REPO=$C TEACHER=$TEACHER PHASE=cands bash $R/ctl/priority_submit.sh --export=ALL,PHASE=cands,TEACHER=$TEACHER research/teacher_probe/bon_probe.sbatch
done
for attempt in 1 2 3; do
    [ "$(have select select)" -ge "$(have cands cands)" ] && break
    log "select attempt $attempt ($(have select select)/$(have cands cands))"
    REPO=$C PHASE=select bash $R/ctl/priority_submit.sh --export=ALL,PHASE=select research/teacher_probe/bon_probe.sbatch
done
cp -n runs/teacher_probe2/answers/*.jsonl $T3/answers/
PYTHONPATH=src:research/rag65 $VPY research/teacher_probe/bon_eval.py arm $O $T3/answers >> "$LOG" 2>&1
log "judging base+RAG, t32, tnext and best-of-3 in one run"
REPO=$C RUN=teacher_probe3 bash $R/ctl/priority_submit.sh --nodes=2 --export=ALL,STAGE=judge,RUN=teacher_probe3 research/rag65/stage.sbatch
PYTHONPATH=src:research/rag65 $VPY research/rag65/paired.py $T3/judged $T3/PAIRED.json \
    bon_rag:base_rag bon_rag:t32_rag bon_rag:tnext_rag tnext_rag:t32_rag t32_rag:base_rag > $T3/paired.out 2>&1
log "DONE: $(tr -d '\n' < $T3/paired.out | head -c 2500)"
