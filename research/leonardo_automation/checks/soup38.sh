#!/bin/bash
# soup38 ("soup v2"): uniform weight soup on round 20 of round 27's 12 x0.5 adapters (bot23, sh24-26) plus every x0.5
# seed adapter of the new-retriever distillation rounds trained on round 20: rs34 (weight 0.8), rs36 (weight 0.5),
# rs37 (weight 0.8 + factory v9-v13) = 21 adapters. The set was fixed on Oct 24 01:55 CEST before any of these rounds
# had a test score; plain fine-tuning (rs35) is left out because that recipe lost twice (rounds 31, 33). Uniform, not
# picked by test score. Starts when all 9 new adapters exist. Answered as a plain model with RAG_VARIANT=b (arm s42_rag
# in shared_split/code/runs/soup38/answers) for ctl/vs_newret.sh soup38, and with the default retriever
# (runs/soup38old) for ctl/vs_release.sh soup38old (the release gate). Log: shared_split/ctl/soup38.
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split
HERE=$S/ctl/soup38; mkdir -p $HERE $R/ctl/soup38old
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUTM=$C/runs/soup38/merged
BASE=$C/runs/gguf_op235/merged   # round 20, as for soup27
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
cd $C; mkdir -p runs/soup38/answers runs/soup38old/answers
OLD=""; for s in 42 43 44; do OLD="$OLD $C/runs/bot23/train/seed$s/adapter_x05"; done
for r in sh24 sh25 sh26; do for s in 42 43 44; do OLD="$OLD $S/code/runs/$r/train/seed$s/adapter_x05"; done; done
NEW=""; for r in rs34 rs36 rs37; do for s in 42 43 44; do NEW="$NEW $S/code/runs/$r/train/seed$s/adapter_x05"; done; done
log "start: waiting for the 9 new x0.5 adapters (rs34, rs36, rs37)"
for a in $OLD; do test -s $a/adapter_model.safetensors || { log "STOP: missing $a"; exit 1; }; done
until n=0; for a in $NEW; do test -s $a/adapter_model.safetensors && n=$((n + 1)); done; [ $n = 9 ]; do sleep 120; done
sleep 120  # make_scaled.py writes the file last; give it time to finish
if [ ! -s $OUTM/model.safetensors ]; then
    log "souping 21 adapters into round 20 (CPU partition)"
    srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 02:00:00 -J dk-soup38 bash -c "export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1; $VPY research/round27/soup_full.py $BASE $OUTM $OLD $NEW" > "$HERE/soup.out" 2>&1
    test -s $OUTM/model.safetensors || { log "STOP: soup failed: $(tail -3 "$HERE/soup.out")"; exit 1; }
fi
log "SOUP: $(tail -1 "$HERE/soup.out")"
export BASE_MODEL=$OUTM
answer() {  # $1 = run dir, $2 = RAG_VARIANT value ("" = default retriever)
    local RD=$1 j a
    cnt() { cat runs/$RD/answers/v2__base_rag.jsonl runs/$RD/answers/v3__base_rag.jsonl runs/$RD/answers/d1__base_rag.jsonl 2>/dev/null | wc -l; }
    for a in 1 2 3 4; do
        [ "$(cnt)" -ge 372 ] && break
        j=$(cd $C && RAG_VARIANT=$2 REPO=$C RUN=$RD ANSWER_MODES=rag VARIANTS=none BASE_SERVED=Qwen/Qwen3-4B-Instruct-2507 sbatch --parsable --nodes=1 --job-name=dk-$RD --export=ALL,STAGE=answer research/rag65/stage.sbatch 2>/dev/null | tail -1)
        [[ "$j" =~ ^[0-9]+$ ]] || { log "$RD: answer submission refused; retrying"; sleep 120; continue; }
        log "$RD: answer job $j ($(cnt)/372)"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done
    done
    [ "$(cnt)" -ge 372 ]
}
answer soup38 b & p1=$!
answer soup38old "" & p2=$!
wait $p1 || { log "STOP: soup38 answers incomplete"; exit 1; }
mkdir -p $S/code/runs/soup38/answers
for l in v2 v3 d1; do cp runs/soup38/answers/${l}__base_rag.jsonl $S/code/runs/soup38/answers/${l}__s42_rag.jsonl; done
log "DONE: soup answers ready (arm s42_rag, RAG_VARIANT=b); judged by ctl/vs_newret.sh soup38"
wait $p2 || { echo "$(date -u +%FT%TZ) STOP: default-retriever answers incomplete" >> $R/ctl/soup38old/controller.log; exit 1; }
for l in v2 v3 d1; do cp runs/soup38old/answers/${l}__base_rag.jsonl runs/soup38old/answers/${l}__s42_rag.jsonl; rm -f runs/soup38old/answers/${l}__base_rag.jsonl runs/soup38old/answers/${l}__base_closed.jsonl; done
echo "$(date -u +%FT%TZ) DONE: answers ready (arm s42_rag, default retriever); judged by ctl/vs_release.sh soup38old" >> $R/ctl/soup38old/controller.log
