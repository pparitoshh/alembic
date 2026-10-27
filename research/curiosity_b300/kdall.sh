#!/bin/bash
# kdall: the current best student (autopilot state) distilled on the union of every teacher set so far (kd1, kd2, kd3:
# Qwen3-235B FP8 answers with top-20 log-probs), no new teacher generation; then eval.sh against round 27 as usual.
source $HOME/cp/bin/lib.sh; TASK=kdall; cd $REPO; wait_models || exit 1
stop_all
NAME=kdall; D=$CP/exp/$NAME; mkdir -p $D/teacher
STUDENT=$(python -c "import json; print(json.load(open('$CP/autopilot/state.json'))['best']['model'])")
for e in kd1 kd2 kd3; do
    for f in $CP/exp/$e/teacher/part*.npz; do [ -e "$f" ] || continue; b=$(basename $f .npz)
        ln -sf $f $D/teacher/part_${e}_${b#part}.npz; ln -sf ${f%.npz}.jsonl $D/teacher/part_${e}_${b#part}.jsonl; done
done
cat $CP/exp/kd1/prompts.jsonl $CP/exp/kd2/prompts.jsonl $CP/exp/kd3/prompts.jsonl > $D/prompts.jsonl 2>/dev/null
log "union teacher data: $(ls $D/teacher/*.npz | wc -l) parts, $(cat $D/teacher/*.jsonl | wc -l) answers; student $STUDENT"
if [ ! -s $D/model/model.safetensors ] && ! ls $D/model/model-*.safetensors > /dev/null 2>&1; then
    torchrun --standalone --nproc_per_node 8 $CP/bin/train_kd.py $STUDENT $D/teacher $D/model >> $CP/logs/$NAME.train.log 2>&1 \
        || { log "STOP: training failed (logs/$NAME.train.log)"; exit 1; }
fi
log "trained: $(cat $D/model/KD_TRAIN.json | tr -d '\n ')"
bash $CP/bin/eval.sh $NAME $D/model
