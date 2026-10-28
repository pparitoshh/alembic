#!/bin/bash
# Steps 6-7 of PLAN.md: full-parameter train_kd.py (bf16 autocast) from ClusterPilot 1.3 on $RUN/parts_$MODE,
# then eval.sh (leaderboard against round 27). Recipe as kdreq13 (1.3 fine-tuned on a new set): LR 1e-5, ALPHA 0.8.
# env: SEED, MODE (onehot | teacher), LR, ALPHA, EPOCHS, NOEVAL=1 (no eval; SMOKE=1 also skips the eval)
source $(dirname $(readlink -f $0))/lib_mt.sh; cd $REPO
MODE=${MODE:-onehot}; P=$RUN/parts_$MODE
[ -e $P/DONE ] || { log "STOP: no parts in $P (run build.sh)"; exit 1; }
export LR=${LR:-1e-5} ALPHA=${ALPHA:-0.8} EPOCHS=${EPOCHS:-1} SEED
mkdir -p $OUT
if [ ! -s $OUT/model/model.safetensors ] && ! ls $OUT/model/model-*.safetensors > /dev/null 2>&1; then
    log "training seed $SEED on $P (LR $LR, ALPHA $ALPHA, $NGPU GPUs x ACCUM $ACCUM)"
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node $NGPU $CB/train_kd.py $STUDENT $P $OUT/model \
        >> $LOGS/mt_${NAME}_s$SEED.train.log 2>&1 || { log "STOP: training failed"; exit 1; }
fi
log "trained: $(cat $OUT/model/KD_TRAIN.json | tr -d '\n ')"
if [ "${NOEVAL:-0}" = 1 ] || [ "$SMOKE" = 1 ]; then exit 0; fi
bash $CB/eval.sh mt_${NAME}_s$SEED $OUT/model
