#!/bin/bash
# Fair comparison with the new search (RAG_VARIANT=b): gpt-oss-20b and Qwen3-32B answer v2/v3/d1 exactly like the
# leaderboard (research/leaderboard/answer.sbatch: same answer script, system prompt, greedy decoding, gpt-oss with
# thinking), only the retrieval differs; our round 27 answers with the same search come from round 30 (exp30 s42_rag).
# All answers get the equal 768-token budget (research/exp30/budget768.py logic), then one fresh 3-judge run + majority.
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=lb_newsearch; O=$C/runs/$OUT; A=$O/answers; LOG=$R/ctl/$OUT.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
run() { local j; until j=$(cd $C && sbatch --parsable --nice=300 "$@" 2>/dev/null | tail -1) && [[ "$j" =~ ^[0-9]+$ ]]; do sleep 120; done
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $A $O/raw
complete() { for l in v2 v3 d1; do [ -s $O/raw/${l}__$1.jsonl ] || return 1; done; }
for a in 1 2 3; do
    complete ns_gptoss20_rag && complete ns_q32_rag && break
    log "answer attempt $a"
    REPO=$C RAG_VARIANT=b run --nodes=1 --job-name=dk-lbns-answer --export=ALL,RAG_VARIANT=b,MODELS="ns_gptoss20=openai/gpt-oss-20b:1:1 ns_q32=Qwen/Qwen3-32B-AWQ:1:0",OUT=$O/raw research/leaderboard/answer.sbatch
done
complete ns_gptoss20_rag && complete ns_q32_rag || { log "STOP: answers incomplete"; exit 1; }
cp -n runs/leaderboard/answers/*__base_closed.jsonl runs/leaderboard/answers/*__base_rag.jsonl $A/
for l in v2 v3 d1; do cp -n runs/exp30/answers/${l}__s42_rag.jsonl $A/${l}__ns_r27_rag.jsonl; cp -n $O/raw/${l}__ns_*_rag.jsonl $A/; done
HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 nice -n 19 $VPY - >> $LOG 2>&1 <<PY
import json, glob
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
for p in glob.glob('$A/*__ns_*_rag.jsonl'):
    rows = [json.loads(x) for x in open(p)]; n = 0
    for r in rows:
        ids = tok(r.get('answer') or '', add_special_tokens=False)['input_ids']
        if len(ids) > 768: r['answer'] = tok.decode(ids[:768]); r['truncated_to'] = 768; n += 1
    with open(p, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('budget768', p.split('/')[-1], n, 'cut')
PY
left() { $VPY - <<PY | tail -1
from pathlib import Path
a = Path('$A'); j = Path('$O/$1'); need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
( for a in 1 2 3 4; do [ "$(left judged)" = 0 ] && break; log "gpt-oss judge attempt $a"
    REPO=$C RUN=$OUT run --nodes=2 --job-name=dk-lbns-judge --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_gemma)" = 0 ] && break; log "gemma judge attempt $a"
    REPO=$C RUN=$OUT run --nodes=2 --job-name=dk-lbns-gemma --export=ALL,RUN=$OUT research/teacher_probe/judge_gemma.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_qwen)" = 0 ] && break; log "qwen3-235b judge attempt $a"
    REPO=$C RUN=$OUT run --nodes=2 --job-name=dk-lbns-qwen --export=ALL,RUN=$OUT research/teacher_probe/judge_qwen235.sbatch; done ) &
wait
export PYTHONPATH=src:research/rag65
$VPY research/rag65/majority.py runs/$OUT >> $LOG 2>&1
for J in judged judged_gemma judged_qwen judged_majority; do $VPY research/rag65/aggregate.py $O/$J $A $O/RESULT_$J.json > /dev/null 2>&1; done
log "DONE: $(python3 -c "
import json, os
for j in ('judged', 'judged_gemma', 'judged_qwen', 'judged_majority'):
    a = json.load(open('$O/RESULT_' + j + '.json'))['arms']
    def m(arm): return [a[l + '/' + arm]['win_rate'] for l in ('v3', 'v2', 'd1')]
    print(j, ' | '.join(f'{arm} {sum(m(arm))/3:.3f} ({m(arm)[0]:.3f}/{m(arm)[1]:.3f}/{m(arm)[2]:.3f})' for arm in ('ns_r27_rag', 'ns_gptoss20_rag', 'ns_q32_rag', 'base_rag')))")"
