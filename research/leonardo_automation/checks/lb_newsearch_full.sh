#!/bin/bash
# FULL improved-search leaderboard (adds every remaining model to runs/lb_newsearch, same judges): models answer v2/v3/d1 exactly like the
# leaderboard (research/leaderboard/answer.sbatch: same answer script, system prompt, greedy decoding, gpt-oss with
# thinking), only the retrieval differs; our round 27 answers with the same search come from round 30 (exp30 s42_rag).
# All answers get the equal 768-token budget (research/exp30/budget768.py logic), then one fresh 3-judge run + majority.
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=lb_newsearch; O=$C/runs/$OUT; A=$O/answers; LOG=$R/ctl/lb_newsearch_full.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
run() { local j; until j=$(cd $C && sbatch --parsable --nice=300 "$@" 2>/dev/null | tail -1) && [[ "$j" =~ ^[0-9]+$ ]]; do sleep 120; done
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $A $O/raw
complete() { for l in v2 v3 d1; do [ -s $O/raw/${l}__$1.jsonl ] || return 1; done; }
LB_GROUPS=("ns_q8=Qwen/Qwen3-8B:1:0 ns_phi4mini=microsoft/Phi-4-mini-instruct:1:0 ns_granite8b=ibm-granite/granite-3.3-8b-instruct:1:0 ns_smol3b=HuggingFaceTB/SmolLM3-3B:1:0"
        "ns_q14=Qwen/Qwen3-14B:1:0 ns_base=Qwen/Qwen3-4B-Instruct-2507:1:0"
        "ns_gemma4=google/gemma-4-26B-A4B-it:4:0"
        "ns_gptoss120=openai/gpt-oss-120b:4:1"
        "ns_tnext=cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit:4:0"
        "ns_t235=QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ:4:0")
answer_group() {  # $1 = group spec; up to 3 attempts for the models of the group that are not complete yet
    for a in 1 2 3; do
        todo=""; for spec in $1; do arm=${spec%%=*}; complete ${arm}_rag || todo="$todo $spec"; done
        [ -z "$todo" ] && return 0
        log "answer attempt $a:$todo"
        REPO=$C RAG_VARIANT=b run --nodes=1 --job-name=dk-lbnf-answer --export=ALL,RAG_VARIANT=b,MODELS="$(echo $todo)",OUT=$O/raw research/leaderboard/answer.sbatch
    done
}
# relaunch-safe: let answer jobs already in the queue finish before checking what is still missing
while squeue --me -h -n dk-lbnf-answer 2>/dev/null | grep -q .; do sleep 60; done
# two answer jobs at a time (debug QoS: 2 running jobs)
( answer_group "${LB_GROUPS[0]}"; answer_group "${LB_GROUPS[2]}"; answer_group "${LB_GROUPS[4]}" ) &
( answer_group "${LB_GROUPS[1]}"; answer_group "${LB_GROUPS[3]}"; answer_group "${LB_GROUPS[5]}" ) &
wait
for spec in ${LB_GROUPS[@]}; do arm=${spec%%=*}; complete ${arm}_rag || log "WARNING: $arm incomplete (left out of the table)"; done
cp -n runs/leaderboard/answers/*__base_closed.jsonl runs/leaderboard/answers/*__base_rag.jsonl $A/
for f in $O/raw/*__ns_*_rag.jsonl; do b=$(basename $f); [ -s $A/$b ] || cp $f $A/$b; done
HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 nice -n 19 $VPY - >> $LOG 2>&1 <<PY
import json, glob
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
for p in glob.glob('$A/*__ns_*_rag.jsonl'):
    if 'truncated_checked' in open(p).read(200): pass
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
    REPO=$C RUN=$OUT run --nodes=4 --job-name=dk-lbnf-judge --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_gemma)" = 0 ] && break; log "gemma judge attempt $a"
    REPO=$C RUN=$OUT run --nodes=4 --job-name=dk-lbnf-gemma --export=ALL,RUN=$OUT research/teacher_probe/judge_gemma.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_qwen)" = 0 ] && break; log "qwen3-235b judge attempt $a"
    REPO=$C RUN=$OUT run --nodes=4 --job-name=dk-lbnf-qwen --export=ALL,RUN=$OUT research/teacher_probe/judge_qwen235.sbatch; done ) &
wait
export PYTHONPATH=src:research/rag65
$VPY research/rag65/majority.py runs/$OUT >> $LOG 2>&1
for J in judged judged_gemma judged_qwen judged_majority; do $VPY research/rag65/aggregate.py $O/$J $A $O/RESULT_$J.json > /dev/null 2>&1; done
log "DONE: full improved-search leaderboard (see ctl/leaderboard/table_newsearch_full.txt)"
$VPY - > $R/ctl/leaderboard/table_newsearch_full.txt <<PY
import json
R = {n: json.load(open(f'$O/RESULT_{f}.json'))['arms'] for n, f in [('gpt-oss', 'judged'), ('gemma', 'judged_gemma'), ('qwen235', 'judged_qwen'), ('MAJORITY', 'judged_majority')]}
arms = sorted({k.split('/', 1)[1] for k in R['MAJORITY']})
def m(r, a):
    v = [r.get(f'{l}/{a}', {}).get('win_rate') for l in ('v3', 'v2', 'd1')]
    return v, (sum(v) / 3 if all(x is not None for x in v) else None)
print('IMPROVED RETRIEVER (RAG_VARIANT=b) + EQUAL 768-TOKEN BUDGET')
print(f'{"model":18s} ' + ' | '.join(f'{n:>8s}' for n in R) + ' | majority v3 / v2 / d1')
for a in sorted(arms, key=lambda a: -(m(R['MAJORITY'], a)[1] or 0)):
    means = [m(R[n], a)[1] for n in R]; v = m(R['MAJORITY'], a)[0]
    print(f'{a:18s} ' + ' | '.join('    -   ' if x is None else f'{x:8.3f}' for x in means) + ' | ' + ' / '.join('-' if x is None else f'{x:.3f}' for x in v))
PY
