#!/bin/bash
# soup32: uniform weight soup of round 32's three seed adapters (full scale; Qwen3-235B logit distillation on
# new-retriever contexts, all trained on round 27), merged into round 27 (research/round27/soup_full.py). Uniform over
# all seeds, not picked by test score (round 32's best single seed, s44, won v3 under the majority but was the best of
# 6 arms). Answered as a plain model with RAG_VARIANT=b, saved as arm s42_rag in shared_split/code/runs/soup32/answers
# so ctl/vs_newret.sh soup32 judges it against round 27 with the same retriever. Log: shared_split/ctl/soup32.
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split
HERE=$S/ctl/soup32; mkdir -p $HERE
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
RD=soup32; D=runs/$RD; OUTM=$C/$D/merged
BASE=$S/models/r27_merged
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
cd $C; mkdir -p $D/answers $D/train
AD=""; for s in 42 43 44; do AD="$AD $S/code/runs/rb32/train/seed$s/adapter"; done
for a in $AD; do test -s $a/adapter_model.safetensors || { log "STOP: missing $a"; exit 1; }; done
if [ ! -s $OUTM/model.safetensors ]; then
    log "souping round 32's 3 seed adapters into round 27 (CPU partition)"
    srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 02:00:00 -J dk-soup32 bash -c "export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1; $VPY research/round27/soup_full.py $BASE $OUTM $AD" > "$HERE/soup.out" 2>&1
    test -s $OUTM/model.safetensors || { log "STOP: soup failed: $(tail -3 "$HERE/soup.out")"; exit 1; }
fi
log "SOUP: $(tail -1 "$HERE/soup.out")"
export BASE_MODEL=$OUTM RAG_VARIANT=b
cnt() { cat $D/answers/v2__base_rag.jsonl $D/answers/v3__base_rag.jsonl $D/answers/d1__base_rag.jsonl 2>/dev/null | wc -l; }
for attempt in 1 2 3 4; do
    [ "$(cnt)" -ge 372 ] && break
    j=$(cd $C && REPO=$C RUN=$RD ANSWER_MODES=rag VARIANTS=none BASE_SERVED=Qwen/Qwen3-4B-Instruct-2507 sbatch --parsable --nodes=1 --job-name=dk-soup32 --export=ALL,STAGE=answer research/rag65/stage.sbatch 2>/dev/null | tail -1)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "answer submission refused; retrying"; sleep 120; continue; }
    log "answer job $j ($(cnt)/372)"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done
done
[ "$(cnt)" -ge 372 ] || { log "STOP: soup answers incomplete ($(cnt)/372)"; exit 1; }
mkdir -p $S/code/runs/$RD/answers
for l in v2 v3 d1; do cp $D/answers/${l}__base_rag.jsonl $S/code/runs/$RD/answers/${l}__s42_rag.jsonl; done
log "DONE: soup answers ready (arm s42_rag, RAG_VARIANT=b); judged by ctl/vs_newret.sh soup32"
