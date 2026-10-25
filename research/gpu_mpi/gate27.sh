#!/bin/bash
# GPU/MPI push (Oct 25): retrieval gating. On d1's held-out-site questions (Aalto Triton) round 27 + RAG scores 0.537
# because unrelated Slurm excerpts mislead it, while closed-book answers often do better. Round 27 answers every
# question closed-book (1,536 tokens); the three judges score those answers against the same reference (closed-book
# base Qwen3-4B), so each question has a judged score with and without excerpts (runs/vsb_r27e has the variant-e ones)
# and switching rules can be compared offline, tuned on d1 and confirmed on the untouched final set.
# Log: ctl/gate27.log
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=gate27; O=$C/runs/$OUT; A=$O/answers; LOG=$R/ctl/$OUT.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
cd $C; mkdir -p $O/raw $A
cnt() { cat $O/raw/v2__base_closed.jsonl $O/raw/v3__base_closed.jsonl $O/raw/d1__base_closed.jsonl 2>/dev/null | wc -l; }
export BASE_MODEL=/leonardo_work/EUHPC_D30_031/shared_split/models/r27_merged ANSWER_MAX_TOKENS=1536
for a in 1 2 3 4; do
    [ "$(cnt)" -ge 372 ] && break
    log "closed-book answer attempt $a ($(cnt)/372)"
    REPO=$C RUN=$OUT/raw ANSWER_MODES=closed VARIANTS=none BASE_SERVED=Qwen/Qwen3-4B-Instruct-2507 \
        bash $R/ctl/priority_submit.sh --nodes=1 --job-name=dk-gate27 --export=ALL,STAGE=answer research/rag65/stage.sbatch
    # stage.sbatch writes runs/$RUN/answers
    for l in v2 v3 d1; do [ -s $O/raw/answers/${l}__base_closed.jsonl ] && cp $O/raw/answers/${l}__base_closed.jsonl $O/raw/; done
done
[ "$(cnt)" -ge 372 ] || { log "STOP: closed-book answers incomplete ($(cnt)/372)"; exit 1; }
for l in v2 v3 d1; do
    cp $O/raw/${l}__base_closed.jsonl $A/${l}__r27_closed.jsonl
    cp -n $C/runs/vs_release_exp30/answers/${l}__base_closed.jsonl $A/
done
log "answers ready; judging"
left() { $VPY - "$A" "$O/$1" <<'PY' | tail -1
import sys
from pathlib import Path
a = Path(sys.argv[1]); j = Path(sys.argv[2]); need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
( for a in 1 2 3 4; do [ "$(left judged)" = 0 ] && break; log "gpt-oss judge attempt $a"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_gemma)" = 0 ] && break; log "gemma judge attempt $a"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT research/teacher_probe/judge_gemma.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_qwen)" = 0 ] && break; log "qwen3-235b judge attempt $a"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT research/teacher_probe/judge_qwen235.sbatch; done ) &
wait
export PYTHONPATH=src:research/rag65
$VPY research/rag65/majority.py runs/$OUT >> $LOG 2>&1
log "DONE: judged closed-book answers in runs/$OUT/judged_majority"
