#!/bin/bash
# Leaderboard with an EQUAL 1536-TOKEN answer budget and the improved retriever (RAG_VARIANT=b). At 768 tokens, 42% of
# round 27 answers (and many of the non-thinking competitors') were cut mid-sentence; at 1536 only 9 of 372 are, and
# round 27 beat its own 768-token answers (+0.020 mean, v3 +0.030, p=0.027; runs/vsb_tok1536). Non-thinking models
# re-answer with ANSWER_MAX_TOKENS=1536 (their old answers were capped at 768 too); gpt-oss-20b keeps its answers from
# lb_newsearch (thinking models had a 6,144-token limit); every answer is then cut at 1536 tokens. Round 27 =
# runs/tok1536. Reference arm unchanged (closed-book base Qwen3-4B), same three judges + 2-of-3 majority.
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUT=lb1536; O=$C/runs/$OUT; A=$O/answers; LOG=$R/ctl/$OUT.log; NS=$C/runs/lb_newsearch
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
run() { local j; until j=$(cd $C && sbatch --parsable "$@" 2>/dev/null | tail -1) && [[ "$j" =~ ^[0-9]+$ ]]; do sleep 120; done
        log "job $j: $*"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; }
cd $C; mkdir -p $A $O/raw
complete() { for l in v2 v3 d1; do [ -s $O/raw/${l}__$1.jsonl ] || return 1; done; }
export RAG_VARIANT=b ANSWER_MAX_TOKENS=1536
( for a in 1 2 3; do complete l15_q32_rag && complete l15_q14_rag && complete l15_base_rag && break; log "answer attempt $a (tp1 models)"
    REPO=$C run --nodes=1 --job-name=dk-lb15-answer --export=ALL,MODELS="l15_q32=Qwen/Qwen3-32B-AWQ:1:0 l15_q14=Qwen/Qwen3-14B:1:0 l15_base=Qwen/Qwen3-4B-Instruct-2507:1:0",OUT=$O/raw research/leaderboard/answer.sbatch; done ) &
( for a in 1 2 3; do complete l15_gemma4_rag && break; log "answer attempt $a (gemma4)"
    REPO=$C run --nodes=1 --job-name=dk-lb15-gemma --export=ALL,MODELS="l15_gemma4=google/gemma-4-26B-A4B-it:4:0",OUT=$O/raw research/leaderboard/answer.sbatch; done ) &
wait
for m in l15_q32 l15_q14 l15_base l15_gemma4; do complete ${m}_rag || { log "STOP: answers incomplete for $m"; exit 1; }; done
cp -n $NS/answers/*__base_closed.jsonl $A/
for l in v2 v3 d1; do
    cp -n $C/runs/tok1536/answers/${l}__base_rag.jsonl $A/${l}__l15_r27_rag.jsonl
    cp -n $NS/raw/${l}__ns_gptoss20_rag.jsonl $A/${l}__l15_gptoss20_rag.jsonl
    cp -n $O/raw/${l}__l15_*_rag.jsonl $A/
done
HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1 nice -n 19 $VPY - "$A" >> $LOG 2>&1 <<'PY'
import json, glob, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
for p in glob.glob(sys.argv[1] + '/*__l15_*_rag.jsonl'):
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
# Both accounts at once (user, Oct 25: all 16 nodes): gpt-oss + Gemma here (4 nodes each), Qwen3-235B on a08trc08 with
# 8 nodes from the shared code copy (ctl/lb1536_her.sh in shared_split waits for READY), its units copied back here.
SH=/leonardo_work/EUHPC_D30_031/shared_split/code/runs/$OUT
mkdir -p $SH/answers $SH/judged_qwen && cp $A/*.jsonl $SH/answers/ && chmod -R g+rwX $SH && touch $SH/READY
log "answers ready ($(ls $A | wc -l) files); copied to shared_split for the a08trc08 Qwen3-235B judge"
( for a in 1 2 3 4; do [ "$(left judged)" = 0 ] && break; log "gpt-oss judge attempt $a ($(left judged) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,STAGE=judge,RUN=$OUT research/rag65/stage.sbatch; done ) &
( for a in 1 2 3 4; do [ "$(left judged_gemma)" = 0 ] && break; log "gemma judge attempt $a ($(left judged_gemma) units left)"
    REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT research/teacher_probe/judge_gemma.sbatch; done ) &
( t0=$SECONDS
  until [ "$(left judged_qwen)" = 0 ]; do
    mkdir -p $O/judged_qwen; cp -n $SH/judged_qwen/*.jsonl $O/judged_qwen/ 2>/dev/null
    if (( SECONDS - t0 > 7200 )); then  # fallback: a08trc08 did not finish within 2 h, judge the rest here
      log "qwen3-235b judge fallback here ($(left judged_qwen) units left)"
      REPO=$C RUN=$OUT bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$OUT research/teacher_probe/judge_qwen235.sbatch
    else sleep 60; fi
  done; log "qwen3-235b units complete" ) &
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
                        for arm in ('l15_r27_rag', 'l15_gptoss20_rag', 'l15_q32_rag', 'l15_gemma4_rag', 'l15_q14_rag', 'l15_base_rag')))
PY
)
log "DONE: $summary"
