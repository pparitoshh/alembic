#!/bin/bash
# Step 5 of PLAN.md: training examples (tool trajectories + replay rows) -> train_kd.py npz parts in $RUN/parts_$MODE.
# env: MODE (onehot | teacher, default onehot), UPSAMPLE (default 3), N_REPLAY (default 4000)
#
# Replay rows: the 1.3 data is $KDALL = $CP_RO/exp/kdall (kdall.sh joined the kd1/kd2/kd3 teacher sets). teacher_gen.py made
# each answer from the evaluation system prompt (configs/rag65_seed42.json student.system_prompt) and the
# prompts.jsonl "user" turn. teacher/part*.jsonl keeps the answer "text" by "id". This script joins the two by id
# into chat rows {id, messages: [system, user, assistant]} without tools. The replay rows thus have the same
# tokens as the 1.3 training data. build_agent_data.py then takes N_REPLAY of them.
source $(dirname $(readlink -f $0))/lib_mt.sh; cd $REPO
MODE=${MODE:-onehot}; UPSAMPLE=${UPSAMPLE:-3}; N_REPLAY=${N_REPLAY:-4000}
[ "$SMOKE" = 1 ] && { UPSAMPLE=1; N_REPLAY=${N_REPLAY_SMOKE:-50}; }
[ -s $RUN/traj_verified.jsonl ] || { log "STOP: no verified trajectories (run collect.sh)"; exit 1; }
if [ ! -s $RUN/replay_src.jsonl ]; then
    python - $KDALL $REPO/configs/rag65_seed42.json $RUN/replay_src.jsonl.tmp <<'PY' || { log "STOP: replay join failed"; exit 1; }
import glob, json, sys
from distillkit.config import load_config
d, cfg, out = sys.argv[1:4]
system = load_config(cfg).student.system_prompt
users = {}
for l in open(f'{d}/prompts.jsonl'):
    r = json.loads(l); users[r['id']] = r['user']
n = 0
with open(out, 'w') as f:
    for p in sorted(glob.glob(f'{d}/teacher/*.jsonl')):
        for l in open(p):
            r = json.loads(l)
            if r['id'] not in users:
                continue
            f.write(json.dumps({'id': r['id'], 'messages': [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': users[r['id']]}, {'role': 'assistant', 'content': r['text']}]},
                     ensure_ascii=False) + '\n'); n += 1
print('replay rows', n)
PY
    mv $RUN/replay_src.jsonl.tmp $RUN/replay_src.jsonl
fi
log "replay source: $(wc -l < $RUN/replay_src.jsonl) rows"
if [ ! -e $RUN/examples/DONE ]; then
    rm -rf $RUN/examples
    python $AT/build_agent_data.py $RUN/examples $RUN/replay_src.jsonl $RUN/traj_verified.jsonl \
        --upsample $UPSAMPLE --n-replay $N_REPLAY >> $LOGS/$TASK.build.log 2>&1 \
        && touch $RUN/examples/DONE || { log "STOP: build_agent_data failed"; exit 1; }
fi
P=$RUN/parts_$MODE
if [ ! -e $P/DONE ]; then
    rm -rf $P
    case $MODE in
        onehot)  extra=(--onehot) ;;
        teacher) extra=(--teacher $TEACHER_DIR) ;;   # one TP 4 job scores every row, so one part (no --part/--nparts)
        *) log "STOP: unknown MODE=$MODE"; exit 1 ;;
    esac
    CUDA_VISIBLE_DEVICES=0,1,2,3 TP=4 python $MT/build_tool_kd.py $RUN/examples/data_src.jsonl $P --tokenizer $STUDENT --tool-prefix mcp__hpc__ "${extra[@]}" \
        >> $LOGS/$TASK.build.log 2>&1 && touch $P/DONE || { log "STOP: build_tool_kd failed"; exit 1; }
fi
log "parts ($MODE): $(ls $P/part*.npz | wc -l)"
