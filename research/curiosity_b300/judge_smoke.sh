#!/bin/bash
# Start the gpt-oss-20b and Gemma judge servers once (and a 4B), ask each one question, stop. Catches environment
# problems before an evaluation needs them.
source $HOME/cp/bin/lib.sh; TASK=judge_smoke; cd $REPO
serve 1 8101 $M/gptoss20 openai/gpt-oss-20b --max-model-len 32768 --gpu-memory-utilization 0.85 &
serve 2 8102 $M/gemma4 google/gemma-4-26B-A4B-it --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":0}' &
serve 0 8100 $M/r27 cand --max-model-len 16384 --gpu-memory-utilization 0.85 &
wait
ok=0
for p in 8100:cand 8101:openai/gpt-oss-20b 8102:google/gemma-4-26B-A4B-it; do
    a=$(curl -s http://127.0.0.1:${p%%:*}/v1/chat/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"${p#*:}\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"What does sbatch do? One sentence.\"}]}" | head -c 300)
    log "smoke ${p#*:}: ${a:0:200}"; grep -q '"content"' <<< "$a" || ok=1
done
exit $ok
