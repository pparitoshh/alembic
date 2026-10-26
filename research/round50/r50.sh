#!/bin/bash
# Round 50 (GPU/MPI push, Oct 25): "better of two" training. On d1 round 27 scores 0.700 with excerpts and 0.663
# closed-book, but picking the better of the two per question would give 0.804; neither a word-coverage rule (+0.019)
# nor the model choosing for itself (+0.007) gets close. So: round 27 answers GPU/MPI/environment training questions
# both ways, Qwen3-235B picks the better answer (with the verified reference), and round 27 is trained to produce that
# answer WITH the excerpts in the prompt (plain loss; replay rows with Qwen3-235B distillation protect the Slurm sets).
# 3 seeds x0.5, soup into round 27, answered with variant e at 1,536 tokens and judged against round 27 + e.
# Log: ctl/r50.log
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
RD=r50; D=$C/runs/$RD; LOG=$R/ctl/$RD.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
cd $C; mkdir -p $D
# 1. questions + excerpts
[ -s $D/questions.jsonl ] || { nice python3 $R/ctl/r50_build.py $D 2000 >> $LOG 2>&1 || { log "STOP: build failed"; exit 1; }; }
log "questions: $(wc -l < $D/questions.jsonl)"
# 2. round 27 answers both ways
for a in 1 2 3; do
    [ -s $D/answers/d1__base_closed.jsonl ] && break
    log "generation attempt $a"
    REPO=$C RD=$D bash $R/ctl/priority_submit.sh --export=ALL $R/ctl/r50_gen.sbatch
done
[ -s $D/answers/d1__base_closed.jsonl ] || { log "STOP: generation failed"; exit 1; }
log "answers: $(wc -l < $D/answers/d1__rag.jsonl) questions answered both ways"
# 3. Qwen3-235B picks the better answer (judge.py on the round's own question file)
n=$(wc -l < $D/eval/d1.jsonl); units=$(( (n + 33) / 34 ))
left() { local k=0; for s in $(seq 0 34 $((n - 1))); do [ -e $D/judged_qwen/d1__rag__$(printf %03d $s).jsonl ] || k=$((k + 1)); done; echo $k; }
for a in 1 2 3 4 5 6; do
    [ "$(left)" = 0 ] && break
    log "qwen3-235b judge attempt $a ($(left) of $units units left)"
    REPO=$C RUN=$RD EVAL_DIR=$D/eval bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$RD,EVAL_DIR=$D/eval research/teacher_probe/judge_qwen235.sbatch
done
[ "$(left)" = 0 ] || { log "STOP: judging incomplete ($(left) units left)"; exit 1; }
# 4. training data (the round runner reads DATA_SRC; its own WAIT_LOG pattern starts training)
mkdir -p $S/code/runs/$RD $S/ctl/$RD && chmod g+rwx $S/code/runs/$RD $S/ctl/$RD
$VPY $R/ctl/r50_train_data.py $D $S/code/runs/tl38/data_src.jsonl $S/code/runs/$RD/data_src.jsonl 6000 2>&1 | tail -1 | tee -a $LOG >> $S/ctl/$RD/prep.log
chmod g+r $S/code/runs/$RD/data_src.jsonl $S/ctl/$RD/prep.log
# 5. train (this account, with the hold) and stop the round runner's answer stage once the seeds are scaled
echo "round $RD (score work first)" > $R/ctl/PRIORITY
(setsid nohup bash $S/ctl/run_round.sh $RD >> $R/ctl/$RD.round.out 2>&1 < /dev/null &)
T=$S/code/runs/$RD/train
scaled() { for s in 42 43 44; do [ -s $T/seed$s/adapter_x05/adapter_config.json ] && [ -s $T/seed$s/adapter_x05/adapter_model.safetensors ] || return 1; done; }
until scaled; do grep -q ' STOP: ' $S/ctl/$RD/controller.log 2>/dev/null && { log "STOP: round stopped"; exit 1; }; [ -e $R/ctl/PRIORITY ] || echo "round $RD" > $R/ctl/PRIORITY; sleep 60; done
sleep 60; pkill -f "^bash $S/ctl/run_round.sh $RD$"; rm -f $S/ctl/$RD/WANT_SLOT; grep -q "round $RD" $R/ctl/PRIORITY 2>/dev/null && rm -f $R/ctl/PRIORITY
log "trained and scaled; answer stage skipped"
# 6. soup into round 27, answer with variant e at 1,536 tokens, judge against round 27 + e
OUTM=$C/runs/$RD/merged; mkdir -p $OUTM
export PYTHONPATH=src:research/rag65 HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1
[ -s $OUTM/model.safetensors ] || srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 01:00:00 -J dk-r50-soup bash -c "$VPY research/round27/soup_full.py $S/models/r27_merged $OUTM $T/seed42/adapter_x05 $T/seed43/adapter_x05 $T/seed44/adapter_x05" > $R/ctl/r50_soup.out 2>&1
test -s $OUTM/model.safetensors || { log "STOP: soup failed"; exit 1; }
log "souped; answering and judging against round 27 + e"
ANSWER_MAX_TOKENS=1536 VARIANT=e bash $R/ctl/ragc.sh r50q $OUTM
REL_ANSWERS=runs/r27e/answers REL_ARM=base_rag bash $R/ctl/vs_newret.sh r50q
log "DONE: $(tail -1 $R/ctl/vsb_r50q.log | cut -c1-200)"
