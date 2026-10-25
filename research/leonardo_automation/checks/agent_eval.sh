#!/bin/bash
# Agent round evaluation (Oct 25): when shared_split/ctl/<RD>/controller.log reports its seeds trained, soup the three
# x0.5 seed adapters into round 27 (as for round 43) and run the agent smoke test (research/agent_smoke, the 10
# held-out tasks) on the result next to round 27. usage: agent_eval.sh RD   Log: ctl/agent_eval_<RD>.log
RD=${1:?round}
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split; T=$S/code/runs/$RD/train
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
LOG=$R/ctl/agent_eval_$RD.log; OUTM=$C/runs/$RD/merged
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
log "waiting for the 3 trained seeds of $RD"
until [ "$(ls $T/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)" = 3 ]; do
    grep -q ' STOP: ' $S/ctl/$RD/controller.log 2>/dev/null && { log "STOP: $RD stopped"; exit 1; }
    sleep 60
done
cd $C; export PYTHONPATH=src:research/rag65 HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1
# run_round.sh scales the adapters itself ("scaling adapters"): wait for its complete x0.5 adapters (a parallel
# make_scaled.py here raced with it and the soup read a half-written folder, Oct 25)
# -r: rounds trained on a08trc08 write owner-only weights; run_round.sh makes them group-readable when it finishes
scaled() { for s in 42 43 44; do [ -s $T/seed$s/adapter_x05/adapter_config.json ] && [ -r $T/seed$s/adapter_x05/adapter_model.safetensors ] && [ -s $T/seed$s/adapter_x05/adapter_model.safetensors ] || return 1; done; }
until scaled; do sleep 30; done
sleep 30
BASE=$(. $S/ctl/rounds/$RD.env; echo $MERGED)  # the round's own base (round 27, or round 43 for r49d)
if [ ! -s $OUTM/model.safetensors ]; then
    mkdir -p $OUTM; log "souping 3 x0.5 seeds into $BASE"
    srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 01:00:00 -J dk-$RD-soup bash -c "$VPY research/round27/soup_full.py $BASE $OUTM $T/seed42/adapter_x05 $T/seed43/adapter_x05 $T/seed44/adapter_x05" > $R/ctl/agent_eval_$RD.soup 2>&1
    test -s $OUTM/model.safetensors || { log "STOP: soup failed: $(tail -2 $R/ctl/agent_eval_$RD.soup)"; exit 1; }
fi
O=$C/runs/smoke_$RD; mkdir -p $O
log "smoke test: $RD (merged) and round 27"
REPO=$C OUT=$O MODELS="$RD=$OUTM" bash $R/ctl/priority_submit.sh --nodes=1 --export=ALL research/agent_smoke/smoke.sbatch
until [ -s $O/SUMMARY_$RD.json ] || ! squeue --me -h -n dk-agent-smoke | grep -q .; do sleep 30; done
log "DONE: $(python3 -c "import json; d=json.load(open('$O/SUMMARY_$RD.json')); print(d['passed'], '/ 10', d['tasks'])" 2>/dev/null || echo 'no summary (see runs/smoke_'$RD')')"
