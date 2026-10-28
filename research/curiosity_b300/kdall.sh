#!/bin/bash
# Union round: a student distilled on the union of several teacher sets (Qwen3-235B FP8 answers with top-20 log-probs),
# no new teacher generation; then eval.sh against round 27 as usual.
# env: NAME (default kdall), SETS (experiments whose teacher sets to join, default "kd1 kd2 kd3"),
#      STUDENT (model dir; default the autopilot's current best)
source $HOME/cp/bin/lib.sh; NAME=${NAME:-kdall}; TASK=$NAME; cd $REPO; wait_models || exit 1
stop_all
D=$CP/exp/$NAME; mkdir -p $D/teacher
STUDENT=${STUDENT:-$(python -c "import json; print(json.load(open('$CP/autopilot/state.json'))['best']['model'])")}
for e in ${SETS:-kd1 kd2 kd3}; do
    for f in $CP/exp/$e/teacher/part*.npz; do [ -e "$f" ] || continue; b=$(basename $f .npz)
        ln -sf $f $D/teacher/part_${e}_${b#part}.npz; ln -sf ${f%.npz}.jsonl $D/teacher/part_${e}_${b#part}.jsonl; done
    cat $CP/exp/$e/prompts.jsonl
done > $D/prompts.jsonl
log "union teacher data (${SETS:-kd1 kd2 kd3}): $(ls $D/teacher/*.npz | wc -l) parts, $(cat $D/teacher/*.jsonl | wc -l) answers; student $STUDENT"
if [ ! -s $D/model/model.safetensors ] && ! ls $D/model/model-*.safetensors > /dev/null 2>&1; then
    ACCUM=${ACCUM:-$(( 64 / NGPU ))} torchrun --standalone --nproc_per_node $NGPU $CP/bin/train_kd.py $STUDENT $D/teacher $D/model >> $CP/logs/$NAME.train.log 2>&1 \
        || { log "STOP: training failed (logs/$NAME.train.log)"; exit 1; }
fi
log "trained: $(cat $D/model/KD_TRAIN.json | tr -d '\n ')"
bash $CP/bin/eval.sh $NAME $D/model
