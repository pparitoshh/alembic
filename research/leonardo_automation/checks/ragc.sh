#!/bin/bash
# ragc: RAG_VARIANT=c (variant b + up to 3 excerpts from the general HPC documentation index, research/rag65/
# build_general_index.py, evaluation sources excluded). Answers a merged model with variant c, saved as arm s42_rag in
# shared_split/code/runs/<RD>/answers, so ctl/vs_newret.sh <RD> judges it against round 27 with variant b: same model
# (round 27) = a retriever-only comparison. usage: ragc.sh RD MERGED_DIR   log: shared_split/ctl/<RD>/controller.log
set -uo pipefail
umask 002
RD=$1; M=$2
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split
HERE=$S/ctl/$RD; mkdir -p $HERE
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
cd $C; D=runs/$RD; mkdir -p $D/answers
until grep -q "INDEX READY" $R/ctl/general_index.log 2>/dev/null; do sleep 120; done
test -s runs/rag65/indexes/general_docs.sqlite || { log "STOP: no general index"; exit 1; }
log "index: $(tail -1 $R/ctl/general_index.log)"
export BASE_MODEL=$M RAG_VARIANT=${VARIANT:-c}  # VARIANT=d (opt-in): variant d
cnt() { cat $D/answers/v2__base_rag.jsonl $D/answers/v3__base_rag.jsonl $D/answers/d1__base_rag.jsonl 2>/dev/null | wc -l; }
for attempt in 1 2 3 4; do
    [ "$(cnt)" -ge 372 ] && break
    j=$(REPO=$C RUN=$RD ANSWER_MODES=rag VARIANTS=none BASE_SERVED=Qwen/Qwen3-4B-Instruct-2507 sbatch --parsable --nodes=1 --job-name=dk-$RD --export=ALL,STAGE=answer research/rag65/stage.sbatch 2>/dev/null | tail -1)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "answer submission refused; retrying"; sleep 120; continue; }
    log "answer job $j ($(cnt)/372)"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done
done
[ "$(cnt)" -ge 372 ] || { log "STOP: answers incomplete ($(cnt)/372)"; exit 1; }
mkdir -p $S/code/runs/$RD/answers
for l in v2 v3 d1; do cp $D/answers/${l}__base_rag.jsonl $S/code/runs/$RD/answers/${l}__s42_rag.jsonl; done
log "DONE: answers ready (arm s42_rag, RAG_VARIANT=$RAG_VARIANT); judged by ctl/vs_newret.sh $RD"
