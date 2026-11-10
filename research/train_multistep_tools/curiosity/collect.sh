#!/bin/bash
# Steps 2-4 of PLAN.md (round 2 data: more tool calls in each chain): task bank -> teacher trajectories (vLLM
# with tool calls, TP 4) -> verification. Resumable: the script skips a step when its output file is complete.
# Outputs go to $RUN (lib_mt.sh). env: SAMPLES (default 1, teacher retries per task, see collect_sim.py).
source $(dirname $(readlink -f $0))/lib_mt.sh; cd $REPO
SAMPLES=${SAMPLES:-1}; VFLAGS=""; [ "$SAMPLES" -gt 1 ] && VFLAGS="--first-pass-per-task"
if [ ! -s $RUN/tasks.jsonl ]; then
    log "task bank (seed ${TASK_SEED:-4}, as tasks_v3.jsonl)"
    python $MT/task_bank.py $RUN/tasks.jsonl.tmp --seed ${TASK_SEED:-4} >> $LOGS/$TASK.collect.log 2>&1 \
        && mv $RUN/tasks.jsonl.tmp $RUN/tasks.jsonl || { log "STOP: task bank failed"; exit 1; }
fi
log "tasks: $(wc -l < $RUN/tasks.jsonl)"
if [ ! -s $RUN/traj.jsonl ]; then
    log "teacher $TEACHER ($TOOL_PARSER tool parser) on GPUs 0-3, samples $SAMPLES"
    serve 0,1,2,3 $PORT $TEACHER_DIR $TEACHER_NAME --max-model-len 32768 --gpu-memory-utilization 0.90 \
        --enable-auto-tool-choice --tool-call-parser $TOOL_PARSER || { log "STOP: teacher server not ready"; exit 1; }
    python $MT/collect_sim.py $RUN/tasks.jsonl $RUN/traj.jsonl.tmp $ENDPOINT $TEACHER_NAME \
        --concurrency ${CONCURRENCY:-32} --samples $SAMPLES ${LIMIT:+--limit $LIMIT} >> $LOGS/$TASK.collect.log 2>&1 \
        && mv $RUN/traj.jsonl.tmp $RUN/traj.jsonl || { log "STOP: collection failed"; exit 1; }
    stop_all   # the EXIT trap also stops it after a failure
fi
log "trajectories: $(wc -l < $RUN/traj.jsonl)"
if [ ! -s $RUN/traj_verified.jsonl ]; then
    python $MT/verify_traj.py $RUN/traj.jsonl $RUN/traj_verified.jsonl.tmp $VFLAGS >> $LOGS/$TASK.collect.log 2>&1 \
        && mv $RUN/traj_verified.jsonl.tmp $RUN/traj_verified.jsonl || { log "STOP: verification failed"; exit 1; }
fi
log "verified: $(grep -c '"pass": *true' $RUN/traj_verified.jsonl) of $(wc -l < $RUN/traj_verified.jsonl) pass"
