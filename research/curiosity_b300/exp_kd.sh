#!/bin/bash
# One distillation experiment: N training prompts -> Qwen3-235B answers with top-20 log-probs -> full-parameter
# distillation of STUDENT -> (unless NOEVAL=1) eval.sh against round 27.
# usage: exp_kd.sh NAME N [STUDENT_DIR]   env: ALPHA, LR, EPOCHS, TEMP, PROMPT_SEED, NOEVAL, TEACHER_REPLICAS (4),
#   TEACHER_FROM=<experiment> (reuse that experiment's prompts and teacher answers instead of generating new ones)
NAME=$1; N=$2; STUDENT=${3:-$HOME/cp/models/r27}
source $HOME/cp/bin/lib.sh; TASK=$NAME; cd $REPO; wait_models || exit 1
D=$CP/exp/$NAME; mkdir -p $D
if [ -n "${TEACHER_FROM:-}" ] && [ ! -e $D/teacher ]; then
    ln -s $CP/exp/$TEACHER_FROM/teacher $D/teacher && ln -s $CP/exp/$TEACHER_FROM/prompts.jsonl $D/prompts.jsonl
    N=$(wc -l < $D/prompts.jsonl)
fi
until [ -e $CP/data/BANKS_COMPLETE ]; do sleep 60; done  # the Leonardo bank copy has finished
if [ ! -s $D/prompts.jsonl ]; then
    log "building $N prompts"
    python $CP/bin/build_prompts.py $D/prompts.jsonl.tmp $N ${PROMPT_SEED:-1} >> $CP/logs/$NAME.build.log 2>&1 && mv $D/prompts.jsonl.tmp $D/prompts.jsonl \
        || { log "STOP: prompt build failed"; exit 1; }
fi
log "prompts: $(wc -l < $D/prompts.jsonl)"
G=${TEACHER_REPLICAS:-$(( NGPU / 2 ))}; TPG=$(( NGPU / G ))  # teacher replicas x GPUs each (4 x TP2 by default; not GROUPS, a bash builtin)
P=$(( (N + 3999) / 4000 )); P=${TEACHER_PARTS:-$(( (P + G - 1) / G * G ))}  # TEACHER_PARTS: resume a split made with other G  # ~4,000 prompts per part, a multiple of G parts
for g in $(seq 0 $((G - 1))); do
    gpus=$(seq -s, $((g * TPG)) $((g * TPG + TPG - 1)))
    ( for k in $(seq $g $G $((P - 1))); do
          CUDA_VISIBLE_DEVICES=$gpus TP=$TPG python $CP/bin/teacher_gen.py \
              $D/prompts.jsonl $D/teacher $k $P $M/q235fp8 $STUDENT >> $CP/logs/$NAME.teacher$g.log 2>&1
      done ) &
done
wait
n=$(ls $D/teacher/part*.npz 2>/dev/null | wc -l)
[ "$n" = "$P" ] || { log "STOP: teacher parts $n of $P"; exit 1; }
log "teacher: $(cat $D/teacher/part*.jsonl | wc -l) answers kept"
if [ ! -s $D/model/model.safetensors ] && ! ls $D/model/model-*.safetensors > /dev/null 2>&1; then
    ACCUM=${ACCUM:-$(( 64 / NGPU ))} torchrun --standalone --nproc_per_node $NGPU $CP/bin/train_kd.py $STUDENT $D/teacher $D/model >> $CP/logs/$NAME.train.log 2>&1 \
        || { log "STOP: training failed (logs/$NAME.train.log)"; exit 1; }
fi
log "trained: $(cat $D/model/KD_TRAIN.json | tr -d '\n ')"
[ "${NOEVAL:-0}" = 1 ] && exit 0
bash $CP/bin/eval.sh $NAME $D/model
