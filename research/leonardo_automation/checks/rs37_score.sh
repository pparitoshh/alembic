#!/bin/bash
# Round 37 teacher scores: the factory v9-v13 releases (20.7K Qwen3-235B-written Q&A over every taxonomy leaf, closed-book) scored by
# Qwen3-235B so round 37 can distil on round 31 data + all new topic data (rb32 scores cover the round 31 rows).
# One 4-node scoring job (the rows are short, closed-book). Output: shared_split/code/runs/rs37/teacher_top20 =
# round 32's scores (kd235 + rebuilt-context rows) plus the new factory scores as kd_7*, then "SCORES READY" in
# shared_split/ctl/rs37/prep.log for run_round.sh rs37 (factory row ids are new, so no row is scored twice).
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split
D=$S/code/runs/rs37; HERE=$R/ctl/rs37; PREP=$S/ctl/rs37/prep.log
OLD_KD=$S/code/runs/rb32/teacher_top20
mkdir -p $D $HERE $S/ctl/rs37
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
heartbeat() { touch "$HERE/HEARTBEAT"; }
adopt() { squeue --me -h -n "$1" -o %i 2>/dev/null | grep -E '^[0-9]+$' | head -1; }
log "start: scoring factory v9-v13 rows"
true
[ -s $D/part0.jsonl ] || python3 - $S $D <<'PY'
import json, os, sys
S, d = sys.argv[1:3]
n = 0
with open(f'{d}/part0.jsonl.tmp', 'w') as out:
    for v in (9, 10, 11, 12, 13):
        for line in open(f'{S}/code/runs/ai_factory_all_v{v}/release/verified.jsonl'):
            out.write(line); n += 1
open(f'{d}/part1.jsonl', 'w').close()
os.replace(f'{d}/part0.jsonl.tmp', f'{d}/part0.jsonl')
print(n, 'factory rows to score (one 4-node job; part1 empty)')
PY
scored() { /leonardo_work/EUHPC_D30_031/shared_split/venv-train/bin/python - $D/kd_new$1 <<'PY'
import glob, sys
import numpy as np
print(sum(len(np.load(f)['row_ids']) for f in glob.glob(sys.argv[1] + '/kd_*.npz')))
PY
}
score_half() {  # $1 = 0|1; resumable; N fixed at 4 (the row split depends on it)
    local k=$1 want j a; want=$(wc -l < $D/part$k.jsonl); mkdir -p $D/kd_new$k
    for a in 1 2 3 4 5 6 7 8; do
        [ "$(scored $k)" -ge "$want" ] && { log "half $k scored ($want rows)"; return 0; }
        j=$(adopt dk-rs37-s$k)
        if [ -z "$j" ]; then
            j=$(cd $C && REPO=$C DATA=$D/part$k.jsonl KD_DIR=$D/kd_new$k sbatch --parsable --nodes=4 --job-name=dk-rs37-s$k research/kd/score235.sbatch 2>/dev/null | tail -1)
            [[ "$j" =~ ^[0-9]+$ ]] || { log "half $k: submission refused; retrying"; sleep 120; continue; }
        fi
        log "half $k: scoring job $j ($(scored $k)/$want rows)"
        while squeue -h -j "$j" 2>/dev/null | grep -q .; do heartbeat; sleep 60; done
    done
    [ "$(scored $k)" -ge "$want" ] || { log "STOP: half $k incomplete after 8 attempts ($(scored $k)/$want)"; return 1; }
}
score_half 0 & p0=$!
score_half 1 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
mkdir -p $D/teacher_top20
shopt -s nullglob  # part1 is empty: no stray kd_71* links
for f in $OLD_KD/kd_*.npz; do ln -sf $f $D/teacher_top20/; done
for k in 0 1; do for f in $D/kd_new$k/kd_*.npz; do b=$(basename $f); ln -sf $f $D/teacher_top20/kd_7${k}${b#kd_}; done; done
chmod -R g+rwX $D
echo "$(date -u +%FT%TZ) SCORES READY (rb32 scores + factory v9-v13): $(ls $D/teacher_top20 | wc -l) files" >> $PREP
log "DONE: teacher scores ready in $D/teacher_top20 (round rs37 trains on them)"
