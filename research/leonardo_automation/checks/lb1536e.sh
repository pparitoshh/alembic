#!/bin/bash
# Equal-1536 leaderboard, variant e (Oct 25): the relevance header (retrieve.py RAG_VARIANT=e; same excerpts as b) raised
# round 27 by +0.010 mean / d1 +0.021 under the 3-judge majority (runs/vsb_r27e). So the models nearest to us answer
# with the same header before any claim: Gemma-4-26B, gpt-oss-20b, Qwen3-32B re-answer with RAG_VARIANT=e
# (non-thinking 1536 tokens, gpt-oss 6,144 with thinking, every answer then cut at 1536), round 27 = runs/r27e.
# Same reference arm (closed-book base Qwen3-4B), same three judges + 2-of-3 majority. Log: ctl/lb1536e.log
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=lb1536e; O=$C/runs/$OUT; A=$O/answers; LOG=$R/ctl/$OUT.log; NS=$C/runs/lb_newsearch
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
run() { local j; until j=$(cd $C && sbatch --parsable "$@" 2>/dev/null | tail -1) && [[ "$j" =~ ^[0-9]+$ ]]; do sleep 120; done
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $A $O/raw
complete() { for l in v2 v3 d1; do [ -s $O/raw/${l}__$1.jsonl ] || return 1; done; }
export RAG_VARIANT=e ANSWER_MAX_TOKENS=1536
log "start"
( for a in 1 2 3; do complete le_q32_rag && complete le_gptoss20_rag && break; log "answer attempt $a (gpt-oss-20b, Qwen3-32B)"
    REPO=$C run --nodes=1 --job-name=dk-lbe-answer --export=ALL,MODELS="le_gptoss20=openai/gpt-oss-20b:1:1 le_q32=Qwen/Qwen3-32B-AWQ:1:0",OUT=$O/raw research/leaderboard/answer.sbatch; done ) &
( for a in 1 2 3; do complete le_gemma4_rag && break; log "answer attempt $a (gemma4)"
    REPO=$C run --nodes=1 --job-name=dk-lbe-gemma --export=ALL,MODELS="le_gemma4=google/gemma-4-26B-A4B-it:4:0",OUT=$O/raw research/leaderboard/answer.sbatch; done ) &
wait
for m in le_q32 le_gptoss20 le_gemma4; do complete ${m}_rag || { log "STOP: answers incomplete for $m"; exit 1; }; done
cp -n $NS/answers/*__base_closed.jsonl $A/
for l in v2 v3 d1; do
    cp -n $C/runs/r27e/answers/${l}__base_rag.jsonl $A/${l}__le_r27_rag.jsonl
    cp -n $O/raw/${l}__le_*_rag.jsonl $A/
done
HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 nice -n 19 $VPY - "$A" >> $LOG 2>&1 <<'PY'
import json, glob, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
for p in glob.glob(sys.argv[1] + '/*__le_*_rag.jsonl'):
    rows = [json.loads(x) for x in open(p)]; n = 0
    for r in rows:
        ids = tok(r.get('answer') or '', add_special_tokens=False)['input_ids']
        if len(ids) > 1536: r['answer'] = tok.decode(ids[:1536]); r['truncated_to'] = 1536; n += 1
    with open(p, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('budget1536', p.split('/')[-1], n, 'cut')
PY
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
log "answers ready ($(ls $A | wc -l) files)"
( for a in 1 2 3 4; do [ "$(left judged)" = 0 ] && break; log "gpt-oss judge attempt $a ($(left judged) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_gemma)" = 0 ] && break; log "gemma judge attempt $a ($(left judged_gemma) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT research/teacher_probe/judge_gemma.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_qwen)" = 0 ] && break; log "qwen3-235b judge attempt $a ($(left judged_qwen) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT research/teacher_probe/judge_qwen235.sbatch; done ) &
wait
export PYTHONPATH=src:research/rag65
$VPY research/rag65/majority.py runs/$OUT >> $LOG 2>&1
for J in judged judged_gemma judged_qwen judged_majority; do $VPY research/rag65/aggregate.py $O/$J $A $O/RESULT_$J.json > /dev/null 2>&1; done
summary=$(python3 - "$O" <<'PY'
import json, sys
for j in ('judged', 'judged_gemma', 'judged_qwen', 'judged_majority'):
    a = json.load(open(sys.argv[1] + '/RESULT_' + j + '.json'))['arms']
    def m(arm): return [a[l + '/' + arm]['win_rate'] for l in ('v3', 'v2', 'd1')]
    print(j, ' | '.join('%s %.3f (%.3f/%.3f/%.3f)' % (arm, sum(m(arm)) / 3, *m(arm))
                        for arm in ('le_r27_rag', 'le_gemma4_rag', 'le_gptoss20_rag', 'le_q32_rag')))
PY
)
log "DONE: $summary"
