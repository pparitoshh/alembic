#!/bin/bash
# Round 51b (d1 push, Oct 26): error-focused distillation. Round 51 judged round 27 against the verified reference itself
# and the judge preferred the reference almost always (4,589/5,000 both orders), so no errors were found. Here
# Qwen3-235B answers the same 5,000 questions with the same prompt (the app's excerpts), the judge compares round 27
# with Qwen3-235B using the verified reference, and round 27 is trained on Qwen3-235B's answer where it is clearly
# better, on its own answer for as many questions where round 27 is level, plus replay rows with Qwen3-235B distillation.
# 3 seeds x0.5, soup into round 27, answered with variant e at 1,536 tokens and judged against round 27 + e.
# Log: ctl/r51b.log
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
RD=r51b; D=$C/runs/$RD; LOG=$R/ctl/$RD.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
cd $C; mkdir -p $D
# 1. round 51's questions, references and round 27 answers
S51=$C/runs/r51
mkdir -p $D/eval $D/answers
[ -s $D/questions.jsonl ] || { cp $S51/questions.jsonl $D/ && cp $S51/eval/d1.jsonl $D/eval/ && cp $S51/answers/d1__rag.jsonl $D/answers/ \
    && : > $D/eval/v2.jsonl && : > $D/eval/v3.jsonl; }
nq=$(wc -l < $D/questions.jsonl)
log "questions: $nq (round 51's, with round 27's answers)"
# 2. Qwen3-235B answers the same prompts (resumable; up to 6 jobs of 3-6 nodes)
# answers in the main file or in parts a job could not merge before its time limit
nans() { cat $D/answers/d1__base_closed.jsonl $D/answers/part*_teacher.jsonl 2>/dev/null | grep -o '"id": "[^"]*"' | sort -u | wc -l; }
for a in 1 2 3 4 5 6; do
    [ "$(nans)" -ge $(( nq * 98 / 100 )) ] && break
    log "qwen3-235b answer attempt $a ($(nans)/$nq answered)"
    REPO=$C RD=$D bash $R/ctl/priority_submit.sh --export=ALL $R/ctl/r51b_teacher.sbatch
done
[ "$(nans)" -ge $(( nq * 98 / 100 )) ] || { log "STOP: teacher answers incomplete ($(nans)/$nq)"; exit 1; }
# merge leftover parts, then the judge pairs only questions with both answers
python3 - $D <<'PY'
import glob, json, os, sys
d = sys.argv[1]
dst = f'{d}/answers/d1__base_closed.jsonl'
t = {json.loads(l)['id'] for l in open(dst) if l.strip()} if os.path.exists(dst) else set()
with open(dst, 'a') as fo:
    for part in sorted(glob.glob(f'{d}/answers/part*_teacher.jsonl')):
        for l in open(part):
            if l.strip() and json.loads(l)['id'] not in t:
                fo.write(l if l.endswith('\n') else l + '\n')
                t.add(json.loads(l)['id'])
        os.remove(part)
for f in (f'{d}/eval/d1.jsonl', f'{d}/answers/d1__rag.jsonl'):
    rows = [l for l in open(f) if json.loads(l)['id'] in t]
    open(f, 'w').writelines(rows)
PY
log "teacher answers: $(nans) of $nq"
# 3. Qwen3-235B judges round 27 against Qwen3-235B's answer, with the verified reference (judge.py on the round's own question file)
n=$(wc -l < $D/eval/d1.jsonl); units=$(( (n + 33) / 34 ))
left() { local k=0; for s in $(seq 0 34 $((n - 1))); do [ -e $D/judged_qwen/d1__rag__$(printf %03d $s).jsonl ] || k=$((k + 1)); done; echo $k; }
for a in 1 2 3 4 5 6 7 8 9 10; do
    [ "$(left)" = 0 ] && break
    log "qwen3-235b judge attempt $a ($(left) of $units units left)"
    REPO=$C RUN=$RD EVAL_DIR=$D/eval bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,RUN=$RD,EVAL_DIR=$D/eval research/teacher_probe/judge_qwen235.sbatch
done
[ "$(left)" = 0 ] || { log "STOP: judging incomplete ($(left) units left)"; exit 1; }
# 4. training data (the round runner reads DATA_SRC; its own WAIT_LOG pattern starts training)
mkdir -p $S/code/runs/$RD $S/ctl/$RD && chmod g+rwx $S/code/runs/$RD $S/ctl/$RD
$VPY $R/ctl/r51b_train_data.py $D $S/code/runs/tl38/data_src.jsonl $S/code/runs/$RD/data_src.jsonl 6000 2>&1 | tail -1 | tee -a $LOG >> $S/ctl/$RD/prep.log
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
[ -s $OUTM/model.safetensors ] || srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 01:00:00 -J dk-r51b-soup bash -c "$VPY research/round27/soup_full.py $S/models/r27_merged $OUTM $T/seed42/adapter_x05 $T/seed43/adapter_x05 $T/seed44/adapter_x05" > $R/ctl/r51b_soup.out 2>&1
test -s $OUTM/model.safetensors || { log "STOP: soup failed"; exit 1; }
log "souped; answering and judging against round 27 + e"
ANSWER_MAX_TOKENS=1536 VARIANT=e bash $R/ctl/ragc.sh r51bq $OUTM
REL_ANSWERS=runs/r27e/answers REL_ARM=base_rag bash $R/ctl/vs_newret.sh r51bq
log "DONE: $(tail -1 $R/ctl/vsb_r51bq.log | cut -c1-200)"
