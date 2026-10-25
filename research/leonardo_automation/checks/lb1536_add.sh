#!/bin/bash
# Add one model to the equal-1536 leaderboard (runs/lb1536): its v2/v3/d1 answers become arm l15_<NAME>_rag (cut at
# 1536 tokens), only its units are judged (same three judges, 4 nodes each), then majority + aggregate again.
# usage: lb1536_add.sh NAME ANSWER_DIR ARM   (e.g. r47 shared_split/code/runs/r47/answers s42_rag)
# log: ctl/lb1536_add_<NAME>.log
set -uo pipefail
umask 002
NAME=$1; SRC=$2; ARM=$3
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=lb1536; O=$C/runs/$OUT; A=$O/answers; LOG=$R/ctl/lb1536_add_$NAME.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
cd $C
for l in v2 v3 d1; do cp $SRC/${l}__${ARM}.jsonl $A/${l}__l15_${NAME}_rag.jsonl; done
HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 nice -n 19 $VPY - "$A" "$NAME" >> $LOG 2>&1 <<'PY'
import json, glob, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
for p in glob.glob(sys.argv[1] + '/*__l15_' + sys.argv[2] + '_rag.jsonl'):
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
summary=$(python3 - "$O" "$NAME" <<'PY'
import json, sys
for j in ('judged', 'judged_gemma', 'judged_qwen', 'judged_majority'):
    a = json.load(open(sys.argv[1] + '/RESULT_' + j + '.json'))['arms']
    arms = ['l15_' + sys.argv[2] + '_rag', 'l15_r27_rag', 'l15_gptoss20_rag', 'l15_q32_rag', 'l15_gemma4_rag', 'l15_q14_rag', 'l15_base_rag']
    def m(arm): return [a[l + '/' + arm]['win_rate'] for l in ('v3', 'v2', 'd1')]
    print(j, ' | '.join('%s %.3f (%.3f/%.3f/%.3f)' % (arm, sum(m(arm)) / 3, *m(arm)) for arm in arms))
PY
)
log "DONE: $summary"
