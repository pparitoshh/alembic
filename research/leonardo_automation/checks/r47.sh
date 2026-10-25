#!/bin/bash
# Round 47: robustness to irrelevant excerpts, by round 20's recipe (on-policy distillation from Qwen3-235B) on round 27.
# d1 analysis (Oct 25): on the 40 Aalto Triton questions (a held-out site) round 27 scores 0.531, because the retriever
# pastes unrelated Slurm pages and the model comments on them or follows them ("the retrieval results are unrelated…",
# CUDA_VISIBLE_DEVICES for a missing toolkit) where closed-book base answers directly. Data: research/round47_build.py
# (6,000 non-Slurm questions with the excerpts retriever b really returns, 2,000 Slurm questions with relevant ones;
# held-out families and near-eval questions excluded). Round 27 answers them; Qwen3-235B scores every token; LoRA on
# round 27 (weight 1.0, r16), 3 seeds, x0.5, uniform soup into round 27, answered with retriever b at 1536 tokens and
# judged against round 27 at 1536 tokens (runs/tok1536) by the three judges. Log: ctl/r47/controller.log
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split
RD=${RD:-r47}  # RD=r48 etc. (opt-in): the same recipe on runs/$RD/data/source.jsonl
HERE=$R/ctl/$RD; mkdir -p $HERE
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
D=runs/$RD
MERGED=$S/models/r27_merged
export BASE_MODEL=$MERGED SCORE_MAX_LEN=16384 CAND_MAX_TOKENS=1536  # answers may run to 1536 tokens (760 dropped 46%)
GEN=$C/$D/gen; KD=$C/$D/teacher_top20; DATA=$C/$D/data/verified.jsonl; SRC=$C/$D/data/source.jsonl
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
adopt() { squeue --me -h -n "$1" -o %i 2>/dev/null | grep -E '^[0-9]+$' | head -1; }
wait_job() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 90; done
             log "job $1: $(sacct -n -X -j "$1" --format=State,Elapsed 2>/dev/null | head -1 | xargs)"; }
submit() { (cd $C && REPO=$C RUN=$RD CFG=$RD ALPHA=1.0 KD_DIR=$KD DATA=$DATA sbatch --parsable "$@" 2>/dev/null | tail -1); }
cd $C
test -s $MERGED/config.json && test -s $SRC || { log "STOP: merged round 27 or source data missing"; exit 1; }
mkdir -p $D/train $KD $D/data $GEN
for s in 42 43 44; do
    [ -s configs/${RD}_seed$s.json ] || sed -e "s#runs/rag65kd/train#$D/train#" -e "s#\"model\": \"Qwen/Qwen3-4B-Instruct-2507\"#\"model\": \"$MERGED\"#" configs/rag65kd_seed$s.json > configs/${RD}_seed$s.json
done
want=$(wc -l < $SRC)
have() { cat $GEN/cands_*.jsonl 2>/dev/null | wc -l; }
log "start: round 27 answers $want questions (6,000 with irrelevant excerpts, 2,000 Slurm)"
for attempt in 1 2 3 4 5; do
    [ "$(have)" -ge $(( want * 95 / 100 )) ] && break  # 376 rows never qualify (generator input filter)
    j=$(adopt dk-$RD-gen); [ -n "$j" ] || j=$(cd $C && REPO=$C MODEL=$MERGED DATA=$SRC OUT=$GEN sbatch --parsable --nodes=2 --job-name=dk-$RD-gen research/kd/onpolicy_gen.sbatch 2>/dev/null | tail -1)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "generation submission refused (slots busy); retrying"; sleep 120; continue; }
    log "student generation job $j ($(have)/$want)"; wait_job "$j"
done
log "GENERATED: $(have)/$want"
[ -s $DATA ] || $VPY - > "$HERE/data.out" <<PY
import glob, json
c = {}
for f in glob.glob('$GEN/cands_*.jsonl'):
    for l in open(f):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get('candidates'):
            c[r['id']] = r['candidates'][0]
n = 0
with open('$DATA', 'w') as f:
    for l in open('$SRC'):
        r = json.loads(l)
        if r['id'] in c and c[r['id']].strip():
            r['messages'][-1]['content'] = c[r['id']]; r['onpolicy'] = 'round27'
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); n += 1
print(n)
PY
log "DATA: $(tail -1 "$HERE/data.out") rows with round 27's own answers"
[ "$(wc -l < $DATA)" -ge 6000 ] || { log "STOP: too few student answers"; exit 1; }
for s in 42 43 44; do mkdir -p $D/train/seed$s && cp -n $DATA $D/train/seed$s/verified.jsonl; done
rows=$(wc -l < $DATA)
scored() { $VPY -c "
import glob, numpy as np
print(sum(len(np.load(f)['row_ids']) for f in glob.glob('$KD/kd_*.npz')))" 2>/dev/null || echo 0; }
for attempt in 1 2 3 4 5 6 7 8; do
    [ "$(scored)" -ge $(( rows * 97 / 100 )) ] && break
    j=$(adopt dk-$RD-score); [ -n "$j" ] || j=$(submit --nodes=${SCORE_NODES:-8} --job-name=dk-$RD-score research/kd/score235.sbatch)  # keep N fixed; 8 nodes = one slot, both halves at once
    [[ "$j" =~ ^[0-9]+$ ]] || { log "scoring submission refused (slots busy); retrying"; sleep 120; continue; }
    log "teacher scoring job $j ($(scored)/$rows)"; wait_job "$j"
done
[ "$(scored)" -ge $(( rows * 97 / 100 )) ] || { log "STOP: only $(scored)/$rows rows scored"; exit 1; }
log "SCORED: $(scored)/$rows rows"
fails=0
for round in $(seq 1 60); do
    done_n=$(ls $D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)
    [ "$done_n" = 3 ] && break
    j=$(adopt dk-$RD-train)
    if [ -z "$j" ]; then
        j=$(submit --nodes=3 --job-name=dk-$RD-train research/kd/train.sbatch)
        [[ "$j" =~ ^[0-9]+$ ]] || { log "train submission refused (slots busy); retrying"; sleep 120; continue; }
        log "train round $round: job $j ($done_n/3 seeds done)"
    fi
    wait_job "$j"
    state=$(sacct -n -X -j "$j" --format=State 2>/dev/null | head -1 | xargs)
    if [ "$state" != TIMEOUT ] && [ "$state" != NODE_FAIL ] && grep -l Traceback $R/logs/train-$j-*.err > /dev/null 2>&1 \
            && ! grep -q 'SignalException' $R/logs/train-$j-*.err; then
        grep -h -B2 -A12 Traceback $R/logs/train-$j-*.err | tail -30 >> "$HERE/controller.log"
        log "STOP: KD training raised an exception (see above)"; exit 1
    fi
    case "$state" in TIMEOUT|COMPLETED*|NODE_FAIL) fails=0 ;; *) fails=$(( fails + 1 )) ;; esac
    [ $fails -ge 3 ] && { log "STOP: 3 consecutive KD training jobs failed ($state)"; exit 1; }
done
[ "$(ls $D/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)" = 3 ] || { log "STOP: KD training unfinished"; exit 1; }
export PYTHONPATH=src:research/rag65
for s in 42 43 44; do t=$D/train/seed$s/adapter_x05; [ -s $t/adapter_model.safetensors ] || $VPY research/rag65/make_scaled.py $D/train/seed$s/adapter 0.5 & done; wait
OUTM=$C/$D/merged
if [ ! -s $OUTM/model.safetensors ]; then
    log "souping the 3 x0.5 seed adapters into round 27 (CPU partition)"
    AD=""; for s in 42 43 44; do AD="$AD $C/$D/train/seed$s/adapter_x05"; done
    srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 02:00:00 -J dk-$RD-soup bash -c "export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1; $VPY research/round27/soup_full.py $MERGED $OUTM $AD" > "$HERE/soup.out" 2>&1
    test -s $OUTM/model.safetensors || { log "STOP: soup failed: $(tail -3 "$HERE/soup.out")"; exit 1; }
fi
log "DONE: round 47 soup at $OUTM; answering with retriever b at 1536 tokens, then three judges vs round 27 at 1536"
ANSWER_MAX_TOKENS=1536 VARIANT=b bash $R/ctl/ragc.sh $RD $OUTM
REL_ANSWERS=runs/tok1536/answers REL_ARM=base_rag bash $R/ctl/vs_newret.sh $RD
