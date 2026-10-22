#!/bin/bash
# Automatic follow-up rounds. usage: auto_chain.sh kd11|rft1     log: ctl/auto_chain_<which>.log
#   kd11: when round 15 (rag65kd11) beats 12a on its 3-seed mean (same seeds, v3 and overall), round 17 (rag65kd12) =
#         on-policy distillation (12e's recipe) on round 15's best seed, cloned from the 12e controller.
#   rft1: when round 16 (rft1) beats 12e on its 3-seed mean (v3 and overall), round 18 (rft2) = a second round of
#         rejection sampling against the teacher on top of round 16's best seed (merged onto 12e first).
# New rounds are listed in ctl/auto_rounds.txt, which the watchdog and the factory's slot rules read.
set -uo pipefail
W=$1
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; CTL=$R/ctl; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
LOG=$CTL/auto_chain_$W.log; log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
case $W in kd11) RD=rag65kd11; REF=a4s; NEW=rag65kd12 ;; rft1) RD=rft1; REF=e9s44; NEW=rft2 ;; *) exit 1 ;; esac
log "waiting for $RD"
until grep -q ' DONE: \| STOP: ' $CTL/$RD/controller.log 2>/dev/null; do sleep 120; done
grep -q ' STOP: ' $CTL/$RD/controller.log && { log "STOP: $RD stopped; no follow-up"; exit 0; }
cd $C
decision=$(python3 - <<PY
import json
p = json.load(open('runs/$RD/PAIRED.json'))
seeds = {}
for s in (42, 43, 44):
    ref = '$REF' if '$REF' == 'e9s44' else f'$REF{s}'
    ds = {l: p.get(f'{l}/s{s}_rag-vs-{ref}_rag', {}).get('diff') for l in ('v3', 'v2', 'd1')}
    if all(v is not None for v in ds.values()):
        seeds[s] = ds
if not seeds:
    print('NOWIN no-pairs'); raise SystemExit
v3 = sum(d['v3'] for d in seeds.values()) / len(seeds)
mean = sum(sum(d.values()) / 3 for d in seeds.values()) / len(seeds)
best = max(seeds, key=lambda s: sum(seeds[s].values()))
print(f'# 3-seed v3 {v3:+.3f}, overall {mean:+.3f}, best seed {best}')
print(f'WIN {best}' if v3 > 0 and mean > 0 else 'NOWIN')
PY
)
echo "$decision" >> $LOG
seed=$(echo "$decision" | grep '^WIN ' | cut -d' ' -f2)
[ -n "$seed" ] || { log "NOWIN: no follow-up round for $RD"; exit 0; }
mkdir -p $CTL/$NEW
if [ $W = kd11 ]; then
    sed -e "s#rag65kd9#$NEW#g" -e "s#runs/r12a_merged#runs/r15_merged#g" -e "s#runs/rag65kd4/train/seed43/adapter#runs/rag65kd11/train/seed$seed/adapter#" \
        -e "s#^\# Round 12e: on-policy distillation (12c's recipe) on top of round 12a.*#\# Round 17 (auto-chained): on-policy distillation (12e's recipe) on top of round 15 seed $seed (more data, weight 0.8, r32).#" \
        -e "s#merging round 12a seed 43#merging round 15 seed $seed#" -e "s#dk-merge-12a#dk-merge-r15#" \
        -e "s#round12a-s43#round15-s$seed#" $CTL/rag65kd9/controller.sh > $CTL/$NEW/controller.sh
else
    M=runs/rft1_merged
    if [ ! -s $M/config.json ]; then
        log "merging round 16 seed $seed onto 12e"
        srun -A EUHPC_D30_031 -p lrd_all_serial -c 4 --mem=30G -t 01:00:00 -J dk-merge-rft1 bash -c "export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1; BASE=$C/runs/gguf_r12e/merged $VPY research/dpo/merge.py runs/rft1/train/seed$seed/adapter $C/$M" >> $LOG 2>&1
        test -s $M/config.json || { log "STOP: merge failed"; exit 1; }
    fi
    sed -e "s#rft1#$NEW#g" -e "s#runs/gguf_r12e/merged  \# 12e.*#runs/rft1_merged  \# round 16 seed $seed merged onto 12e#" \
        -e "s#i % 3 == 0#i % 3 == 1#" \
        -e "s#^\# Round 16 (RFT-1).*#\# Round 18 (RFT-2, auto-chained): rejection sampling against the teacher on top of round 16 seed $seed.#" \
        $CTL/rft1/controller.sh > $CTL/$NEW/controller.sh
fi
bash -n $CTL/$NEW/controller.sh || { log "STOP: generated controller has a syntax error"; exit 1; }
grep -qx $NEW $CTL/auto_rounds.txt 2>/dev/null || echo $NEW >> $CTL/auto_rounds.txt
(cd $CTL && setsid nohup bash $CTL/vs_teacher_auto.sh $NEW >> vs_teacher_$NEW.nohup 2>&1 < /dev/null &)
(cd $CTL && setsid nohup bash -c "bash $CTL/vs_release.sh $NEW; bash $CTL/auto_release.sh $NEW" >> vs_release_$NEW.nohup 2>&1 < /dev/null &)
log "LAUNCHED $NEW (listed in auto_rounds.txt; the watchdog starts it within a minute)"
