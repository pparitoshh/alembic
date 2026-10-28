#!/bin/bash
# Evaluate one 4B model like the Leonardo leaderboard (equal-1536, variant e, reviewed sets, 3 judges + majority) and
# compare it question by question with round 27 + e. usage (inside the agent job): eval.sh NAME MODEL_DIR
# GPUs: 0 candidate | 1 gpt-oss-20b | 2 Gemma 4 26B | 4-7 Qwen3-235B (FP8, TP4). Writes runs/NAME/RESULT.json.
NAME=$1; MODEL=$2
# env (defaults = team folder): EVAL_LIB (helper lib to source), EVAL_RUNS (output root), EVAL_LOGS, EVAL_BIN (answer.py dir)
source ${EVAL_LIB:-$HOME/cp/bin/lib.sh}; TASK=eval-$NAME; cd $REPO; wait_models || exit 1
RUNS=${EVAL_RUNS:-$CP/runs}; LOGS=${EVAL_LOGS:-$CP/logs}; BIN=${EVAL_BIN:-$CP/bin}
RUN=$RUNS/$NAME; A=$RUN/answers; mkdir -p $A
for l in v2 v3 d1; do
    cp -n $CP/data/lb1536e/answers/${l}__base_closed.jsonl $A/
    cp -n $CP/data/r27e/answers/${l}__base_rag.jsonl $A/${l}__r27e.jsonl
done
# round 27's verdicts, judged once on this cluster, are reused so only the candidate is judged
for j in judged judged_gemma judged_qwen; do mkdir -p $RUN/$j; cp -n $RUNS/ref_r27e/$j/*__r27e__*.jsonl $RUN/$j/ 2>/dev/null || cp -n $CP/runs/ref_r27e/$j/*__r27e__*.jsonl $RUN/$j/ 2>/dev/null; done
log "answering with $MODEL"
serve 0 8100 "$MODEL" cand --max-model-len 16384 --gpu-memory-utilization 0.85 || exit 1
python $BIN/answer.py http://127.0.0.1:8100/v1 cand $A cand | tee -a $LOGS/tasks.log
stop_all
log "judging"
J1=1 J2=2 JQ=4,5,6,7; [ $NGPU = 4 ] && J1=0 J2=1 JQ=2,3  # 4 GPUs: Qwen3-235B FP8 on TP2
serve $J1 8101 $M/gptoss20 openai/gpt-oss-20b --max-model-len 32768 --gpu-memory-utilization 0.85 &
serve $J2 8102 $M/gemma4 google/gemma-4-26B-A4B-it --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":0}' &
serve $JQ 8103 $M/q235fp8 qwen3-235b --max-model-len 16384 --gpu-memory-utilization 0.90 &
wait
for p in 8101 8102 8103; do curl -sf http://127.0.0.1:$p/v1/models > /dev/null || { log "STOP: judge server $p not ready"; exit 1; }; done
export EVAL_DIR=$REPO/data/eval/clean
for w in 0 1 2 3; do
    python research/rag65/judge.py $w 4 http://127.0.0.1:8101/v1 $A $RUN/judged > /dev/null 2>&1 &
    JUDGE_MODEL=google/gemma-4-26B-A4B-it python research/rag65/judge.py $w 4 http://127.0.0.1:8102/v1 $A $RUN/judged_gemma > /dev/null 2>&1 &
    JUDGE_MODEL=qwen3-235b python research/rag65/judge.py $w 4 http://127.0.0.1:8103/v1 $A $RUN/judged_qwen > /dev/null 2>&1 &
done
wait
python research/rag65/majority.py $RUN > /dev/null
for j in judged judged_gemma judged_qwen judged_majority; do
    python research/rag65/paired.py $RUN/$j $RUN/PAIRED_$j.json cand:r27e > /dev/null
done
python - $RUN <<'PY' | tee -a $LOGS/tasks.log
import json, sys
run = sys.argv[1]
res = {j: json.load(open(f'{run}/PAIRED_{j}.json')) for j in ('judged', 'judged_gemma', 'judged_qwen', 'judged_majority')}
m = res['judged_majority']
out = {'model': run.split('/')[-1],
       'majority': {l: {k: m[f'{l}/cand-vs-r27e'][k] for k in ('a', 'b', 'diff', 'paired_95')} for l in ('v2', 'v3', 'd1') if f'{l}/cand-vs-r27e' in m},
       'by_judge': {j: {l: res[j][f'{l}/cand-vs-r27e']['diff'] for l in ('v2', 'v3', 'd1') if f'{l}/cand-vs-r27e' in res[j]} for j in res}}
out['mean_cand'] = round(sum(v['a'] for v in out['majority'].values()) / 3, 4)
out['mean_r27e'] = round(sum(v['b'] for v in out['majority'].values()) / 3, 4)
json.dump(out, open(f'{run}/RESULT.json', 'w'), indent=1)
print('RESULT', json.dumps(out))
PY
# keep round 27's verdicts for the next evaluations (they then judge only the candidate)
for j in judged judged_gemma judged_qwen; do mkdir -p $RUNS/ref_r27e/$j; cp -n $RUN/$j/*__r27e__*.jsonl $RUNS/ref_r27e/$j/ 2>/dev/null; done
log "DONE: $(cat $RUN/RESULT.json | tr -d '\n ' | cut -c1-300)"
