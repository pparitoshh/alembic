#!/bin/bash
# Side job, 3 GPUs (with the agent job's 4: 7, one GPU always free): second, larger closed-book candidate kdcbx = kdcb's 24k + cbextra's
# 12k closed-book answers + the same 6 replay parts of 1.3's data, trained from 1.3. Its evaluation runs in the agent
# job (queue/544_eval_kdcbx.sh), because the judges need 4 GPUs.
# Not registered with the autopilot: no auto-release.
source $HOME/cp/bin/lib.sh; NAME=kdcbx; TASK=$NAME; cd $REPO
D=$CP/exp/$NAME; T=$D/train_data; mkdir -p $T
# train_kd.py reads only part*.npz: every link starts with part_ (kdcb's replay_* links were skipped that way)
for f in $CP/exp/kdcb/teacher/*.npz; do b=$(basename $f .npz); ln -sf $(readlink -f $f) $T/part_kdcb_$b.npz; ln -sf $(readlink -f ${f%.npz}.jsonl) $T/part_kdcb_$b.jsonl; done
for f in $D/teacher/part*.npz; do b=$(basename $f .npz); ln -sf $f $T/part_x_$b.npz; ln -sf ${f%.npz}.jsonl $T/part_x_$b.jsonl; done
n=$(cat $T/part_*.jsonl | wc -l); r=$(cat $T/part_kdcb_replay_*.jsonl | wc -l)
log "kdcbx data: $n answers ($r replayed)"
[ $r -gt 15000 ] && [ $n -gt 45000 ] || { log "STOP: kdcbx data incomplete ($n answers, $r replayed)"; exit 1; }
if [ ! -s $D/model/model.safetensors ] && ! ls $D/model/model-*.safetensors > /dev/null 2>&1; then
    ACCUM=21 torchrun --standalone --nproc_per_node 3 $CP/bin/train_kd.py $CP/exp/kdall/model $T $D/model >> $CP/logs/$NAME.train.log 2>&1 \
        || { log "STOP: training failed (logs/$NAME.train.log)"; exit 1; }
fi
log "kdcbx trained: $(cat $D/model/KD_TRAIN.json | tr -d '\n ')"
