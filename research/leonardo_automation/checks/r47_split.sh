#!/bin/bash
# Round 47 teacher scoring split over both debug slots (user, Oct 25: use all nodes): halves a/b of the data, one
# 4-node score235 job each, resubmitted until scored; then linked into teacher_top20 and round 47 resumes at training.
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017; C=$R/code; D=$C/runs/r47
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> $R/ctl/r47/controller.log; }
scored() { $VPY -c "
import glob, numpy as np
print(sum(len(np.load(f)[\"row_ids\"]) for f in glob.glob(\"$D/teacher_$1/kd_*.npz\")))" 2>/dev/null || echo 0; }
half() {
  local h=$1 rows; rows=$(wc -l < $D/data/half_$h.jsonl)
  for a in 1 2 3 4 5 6; do
    [ "$(scored $h)" -ge $(( rows * 97 / 100 )) ] && break
    j=$(squeue --me -h -n dk-r47-score$h -o %i | head -1)
    [ -n "$j" ] || j=$(cd $C && REPO=$C RUN=r47 CFG=r47 KD_DIR=$D/teacher_$h DATA=$D/data/half_$h.jsonl SCORE_MAX_LEN=16384 sbatch --parsable --nodes=4 --job-name=dk-r47-score$h --export=ALL research/kd/score235.sbatch 2>/dev/null | tail -1)
    [[ "$j" =~ ^[0-9]+$ ]] || { sleep 120; continue; }
    log "half $h: scoring job $j ($(scored $h)/$rows)"
    while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done
  done
  log "half $h: $(scored $h)/$rows scored"
}
half a & half b & wait
for h in a b; do for f in $D/teacher_$h/kd_*.npz; do ln -sf $f $D/teacher_top20/kd_${h}_$(basename $f | sed s/^kd_//); done; done
log "split scoring merged: $(ls $D/teacher_top20/kd_*.npz | wc -l) files; resuming round 47"
exec bash $R/ctl/r47.sh
