#!/bin/bash
# Oct 25: complete the equal-1536 variant-e leaderboard (runs/lb1536e) with the 80B+ models (Qwen3-235B, Qwen3-Next-80B,
# gpt-oss-120b; measured only at 768 so far) and round 43 (tools soup), then judge only the new units with the same three
# judges. In parallel, round 43 + e head to head against round 27 + e (vs_newret r43e, reference runs/r27e).
# Log: ctl/lbbig_e.log
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=lb1536e; O=$C/runs/$OUT; A=$O/answers; LOG=$R/ctl/lbbig_e.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
run() { local j; until j=$(cd $C && sbatch --parsable "$@" 2>/dev/null | tail -1) && [[ "$j" =~ ^[0-9]+$ ]]; do sleep 120; done
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $A $O/raw
complete() { for l in v2 v3 d1; do [ -s $O/raw/${l}__$1.jsonl ] || return 1; done; }
export RAG_VARIANT=e ANSWER_MAX_TOKENS=1536
log "start"
answer() {  # $1 = arm=model:tp:thinking
    local arm=${1%%=*}
    for a in 1 2 3; do complete ${arm}_rag && return 0; log "answer attempt $a ($arm)"
        REPO=$C run --nodes=1 --job-name=dk-lbbig-answer --export=ALL,MODELS="$1",OUT=$O/raw research/leaderboard/answer.sbatch; done
}
# round 43 + e: answers for the head to head and the leaderboard (ragc.sh writes runs/r43e/answers)
( ANSWER_MAX_TOKENS=1536 VARIANT=e bash $R/ctl/ragc.sh r43e $C/runs/soup43/merged
  REL_ANSWERS=runs/r27e/answers REL_ARM=base_rag bash $R/ctl/vs_newret.sh r43e ) &
( answer "le_t235=QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ:4:0"; answer "le_gptoss120=openai/gpt-oss-120b:4:1" ) &
( answer "le_tnext=cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit:4:0" ) &
# wait for the answers only (the head to head keeps running in the background subshell)
while ! { complete le_t235_rag && complete le_gptoss120_rag && complete le_tnext_rag && \
          [ -s runs/r43e/answers/d1__base_rag.jsonl ] && [ -s runs/r43e/answers/v3__base_rag.jsonl ] && [ -s runs/r43e/answers/v2__base_rag.jsonl ]; }; do
    sleep 120
    jobs -r | grep -q . || break
done
for m in le_t235 le_gptoss120 le_tnext; do complete ${m}_rag || log "WARN: answers incomplete for $m (left out)"; done
for l in v2 v3 d1; do
    for m in le_t235 le_gptoss120 le_tnext; do [ -s $O/raw/${l}__${m}_rag.jsonl ] && cp -n $O/raw/${l}__${m}_rag.jsonl $A/; done
    [ -s runs/r43e/answers/${l}__base_rag.jsonl ] && cp -n runs/r43e/answers/${l}__base_rag.jsonl $A/${l}__le_r43_rag.jsonl
done
HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 nice -n 19 $VPY - "$A" >> $LOG 2>&1 <<'PY'
import json, glob, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
for arm in ('le_t235', 'le_gptoss120', 'le_tnext', 'le_r43'):
    for p in glob.glob(sys.argv[1] + '/*__' + arm + '_rag.jsonl'):
        rows = [json.loads(x) for x in open(p)]; n = 0
        if any(r.get('truncated_to') for r in rows):
            continue
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
log "answers ready ($(ls $A | wc -l) files); judging the new units"
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
    arms = [x for x in ('le_t235_rag', 'le_tnext_rag', 'le_gptoss120_rag', 'le_gemma4_rag', 'le_gptoss20_rag', 'le_q32_rag',
                        'le_r43_rag', 'le_r27_rag') if all(l + '/' + x in a for l in ('v3', 'v2', 'd1'))]
    print(j, ' | '.join('%s %.3f (%.3f/%.3f/%.3f)' % (arm, sum(m(arm)) / 3, *m(arm)) for arm in arms))
PY
)
log "DONE: $summary"
