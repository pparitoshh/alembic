#!/bin/bash
# soup32old: the round-32 seed soup (runs/soup32/merged) answered with the DEFAULT retriever (RAG_VARIANT unset), the
# setup the public hpc-assistant wrapper uses today, so ctl/vs_release.sh soup32old compares it with the Hugging Face
# release (round 27) exactly as every earlier release check did. Release gate: not worse there.
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; HERE=$R/ctl/soup32old; mkdir -p $HERE
RD=soup32old; D=runs/$RD
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
cd $C; mkdir -p $D/answers
unset RAG_VARIANT
export BASE_MODEL=$C/runs/soup32/merged
cnt() { cat $D/answers/v2__base_rag.jsonl $D/answers/v3__base_rag.jsonl $D/answers/d1__base_rag.jsonl 2>/dev/null | wc -l; }
for attempt in 1 2 3 4; do
    [ "$(cnt)" -ge 372 ] && break
    j=$(cd $C && REPO=$C RUN=$RD ANSWER_MODES=rag VARIANTS=none BASE_SERVED=Qwen/Qwen3-4B-Instruct-2507 sbatch --parsable --nodes=1 --job-name=dk-soup32old --export=ALL,STAGE=answer research/rag65/stage.sbatch 2>/dev/null | tail -1)
    [[ "$j" =~ ^[0-9]+$ ]] || { log "answer submission refused; retrying"; sleep 120; continue; }
    log "answer job $j ($(cnt)/372)"; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done
done
[ "$(cnt)" -ge 372 ] || { log "STOP: answers incomplete ($(cnt)/372)"; exit 1; }
for l in v2 v3 d1; do cp $D/answers/${l}__base_rag.jsonl $D/answers/${l}__s42_rag.jsonl; rm -f $D/answers/${l}__base_rag.jsonl $D/answers/${l}__base_closed.jsonl; done
log "DONE: answers ready (arm s42_rag, default retriever); judged by ctl/vs_release.sh soup32old"
