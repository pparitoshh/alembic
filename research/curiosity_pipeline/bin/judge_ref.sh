#!/bin/bash
# Judge round 27 + e (Leonardo's answers) once on this cluster with the three judges; eval.sh reuses these verdicts.
source $HOME/cp/bin/lib.sh; TASK=ref_r27e; cd $REPO; wait_models || exit 1
RUN=$CP/runs/ref_r27e; A=$RUN/answers; mkdir -p $A
for l in v2 v3 d1; do cp -n $CP/data/lb1536e/answers/${l}__base_closed.jsonl $A/; cp -n $CP/data/r27e/answers/${l}__base_rag.jsonl $A/${l}__r27e.jsonl; done
serve 1 8101 $M/gptoss20 openai/gpt-oss-20b --max-model-len 32768 --gpu-memory-utilization 0.85 &
serve 2 8102 $M/gemma4 google/gemma-4-26B-A4B-it --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":0}' &
serve 4,5,6,7 8103 $M/q235fp8 qwen3-235b --max-model-len 16384 --gpu-memory-utilization 0.90 &
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
python - $RUN <<'PY' | tee -a $CP/logs/tasks.log
import glob, json, sys
S = {'A': 1., 'T': .5, 'B': 0.}; R = {'B': 1., 'T': .5, 'A': 0.}
run = sys.argv[1]
for j in ('judged', 'judged_gemma', 'judged_qwen', 'judged_majority'):
    m = {}
    for l in ('v2', 'v3', 'd1'):
        s = [(S[r['judge']['arm_vs_base'][0]] + R[r['judge']['arm_vs_base'][1]]) / 2 for f in glob.glob(f'{run}/{j}/{l}__r27e__*.jsonl')
             for r in map(json.loads, open(f)) if None not in r['judge']['arm_vs_base']]
        m[l] = round(sum(s) / max(1, len(s)), 3)
    print('REF r27e', j, m, 'mean', round(sum(m.values()) / 3, 3))
PY
log "DONE: round 27 + e judged on Curiosity"
