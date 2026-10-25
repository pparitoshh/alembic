#!/bin/bash
# Automation after round 47 (user, Oct 25: "automate everything"). Two branches run side by side; every decision rule
# is fixed here before any result is seen. Log: ctl/auto_next.log
#
# SCORE: when round 47's judge run (ctl/vsb_r47.log) is DONE, read the 3-judge majority vs round 27 at 1536 tokens.
#   POSITIVE = mean diff > 0 and d1 diff >= +0.02 and v3 diff >= -0.015:
#     1. confirmation: the same answers re-judged in a fresh run (r47b);
#     2. leaderboard: round 47 added to the equal-1536 leaderboard (missing units only, same judges);
#     3. ctl/RELEASE_CANDIDATE written; the Hugging Face upload is done from the laptop (token never on the cluster).
#   otherwise: round 48 = the same recipe on 12,000 fresh non-Slurm questions + 2,000 Slurm questions.
# COVERAGE: when factory batches v26 and v27 are DONE, rerun the coverage report (v3-v27); every topic with fewer than
#   100 verified Q&A gets a targeted batch (v28, a08trc08 queue); after it, one more coverage report.
set -uo pipefail
umask 002
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split; LOG=$R/ctl/auto_next.log
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
done_or_stop() { grep -q ' DONE: \| STOP: ' "$1" 2>/dev/null; }

score_branch() {
    log "score: waiting for round 47's verdict"
    until done_or_stop $R/ctl/vsb_r47.log || tail -1 $R/ctl/r47/controller.log 2>/dev/null | grep -q ' STOP: '; do sleep 120; done  # only a STOP as the latest line counts (an earlier one was fixed and restarted)
    if ! grep -q ' DONE: ' $R/ctl/vsb_r47.log 2>/dev/null; then log "score: STOP: round 47 did not finish (see ctl/r47/controller.log)"; return; fi
    verdict=$(python3 - $C/runs/vsb_r47/PAIRED_majority.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); diff = {}; a = b = 0
for k, v in d.items():
    if k.split('/')[1] == 's42_rag-vs-rel_rag':
        diff[k.split('/')[0]] = v['diff']; a += v['a']; b += v['b']
mean = (a - b) / 3
ok = mean > 0 and diff['d1'] >= 0.02 and diff['v3'] >= -0.015
print('POSITIVE' if ok else 'NEGATIVE', 'mean %+.3f v2 %+.3f v3 %+.3f d1 %+.3f' % (mean, diff['v2'], diff['v3'], diff['d1']))
PY
)
    log "score: round 47 vs round 27 at 1536 tokens (majority): $verdict"
    if [ "${verdict%% *}" = POSITIVE ]; then
        mkdir -p $S/code/runs/r47b/answers $S/ctl/r47b
        cp $S/code/runs/r47/answers/*.jsonl $S/code/runs/r47b/answers/
        echo "$(date -u +%FT%TZ) DONE: confirmation copy of round 47's answers, re-judged in a fresh run" >> $S/ctl/r47b/controller.log
        log "score: confirmation run r47b"
        REL_ANSWERS=runs/tok1536/answers REL_ARM=base_rag bash $R/ctl/vs_newret.sh r47b
        log "score: confirmation: $(tail -1 $R/ctl/vsb_r47b.log | cut -c1-200)"
        log "score: adding round 47 to the equal-1536 leaderboard"
        bash $R/ctl/lb1536_add.sh r47 $S/code/runs/r47/answers s42_rag
        log "score: leaderboard: $(grep 'DONE:' $R/ctl/lb1536_add_r47.log | tail -1 | cut -c1-400)"
        echo "round 47 merged model: $C/runs/r47/merged; verdict: $verdict; confirmation and leaderboard in ctl/auto_next.log" > $R/ctl/RELEASE_CANDIDATE
        log "score: DONE: RELEASE_CANDIDATE written (Hugging Face upload from the laptop)"
    else
        log "score: round 48: 12,000 fresh non-Slurm questions + 2,000 Slurm questions, same recipe"
        mkdir -p $C/runs/r48/data
        (cd $C && N_GEN=12000 N_SLURM=2000 SEED=48 PREFIX=r48 EXCLUDE_IDS=runs/r47/data/source.jsonl nice -n 10 \
            python3 research/round47_build.py runs/r48/data/source.jsonl) >> $LOG 2>&1
        RD=r48 bash $R/ctl/r47.sh
        log "score: round 48: $(tail -1 $R/ctl/vsb_r48.log 2>/dev/null | cut -c1-300)"
    fi
}

coverage_run() {  # $1 = output name
    local O=$S/coverage/$1 Q="" P=$S/venv-train/bin/python D=/leonardo_work/EUHPC_D30_031/runs/a08trc09/documentation
    mkdir -p $O
    for v in 3 4 5 6 7 8; do Q="$Q --qa v$v=$C/runs/ai_factory_all_v$v/release/verified.jsonl"; done
    for v in $(seq 9 40); do f=$S/code/runs/ai_factory_all_v$v/release/verified.jsonl; [ -s $f ] && Q="$Q --qa v$v=$f"; done
    log "coverage: report $1"
    (cd $O && srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 04:00:00 -J dk-coverage $P $C/research/general_hpc_docs/coverage.py \
        --archive aifactory=$D/EU_AI_Factory_Documentation_2026-10-13 --archive general=$D/General_HPC_Documentation_2026-10-19 \
        --archive gaps=$D/General_HPC_Gaps_2026-10-20 --archive gaps3=$D/General_HPC_Gaps3_2026-10-23 \
        --archive gaps4=$D/General_HPC_Gaps4_2026-10-24 --qa rag65v5=$C/runs/rag65v5/train/seed42/verified.jsonl $Q \
        --doc-urls "$C/runs/ai_factory_all_v*/licences.json" --doc-urls "$S/code/runs/ai_factory_all_v*/licences.json" \
        --doc-urls "$C/runs/ai_factory_all_v[1-4]/release/verified.jsonl" --out $O --workers 8 > $O/coverage.log 2>&1)
    python3 - $O/coverage.json <<'PY' >> $LOG
import json, sys
d = json.load(open(sys.argv[1]))
thin = [r['area'] + '/' + r['leaf'] + ' (%d)' % r['qa_verified'] for r in d['rows'] if r['qa_status'] != 'OK']
print('coverage: Q&A status', d['qa_status_counts'], '; under 100:', ', '.join(thin) or 'none')
PY
}

coverage_branch() {
    log "coverage: waiting for v26 and v27"
    until done_or_stop $S/ctl/ai_factory_all_v26/controller.log && done_or_stop $S/ctl/ai_factory_all_v27/controller.log; do sleep 300; done
    coverage_run run_2026-10-25
    thin=$(python3 -c "
import json
d = json.load(open('$S/coverage/run_2026-10-25/coverage.json'))
print(','.join(r['area'] + '/' + r['leaf'] for r in d['rows'] if r['qa_status'] != 'OK'))")
    if [ -z "$thin" ]; then log "coverage: DONE: every topic has 100+ verified Q&A"; return; fi
    log "coverage: targeted batch v28 for: $thin"
    sed -e "s/ai_factory_all_v27/ai_factory_all_v28/; s/general-hpc-27/general-hpc-28/; s/VARIANT_BASE=87 TOOL_VARIANT_BASE=259/VARIANT_BASE=93 TOOL_VARIANT_BASE=269/" \
        -e "s#\$S/code/runs/ai_factory_all_v26\$#\$S/code/runs/ai_factory_all_v26,\$S/code/runs/ai_factory_all_v27#" \
        -e "s#^export LEAVES=.*#export LEAVES=$thin  \# coverage run_2026-10-25: topics under 100 verified Q\&A#" \
        -e "s#PER_LEAF_CAP=\${PER_LEAF_CAP:-450}#PER_LEAF_CAP=\${PER_LEAF_CAP:-3000}#" \
        -e "s#\${TARGET:-16000}#\${TARGET:-8000}#" $S/ctl/v27_prepare.sh > $S/ctl/v28_prepare.sh
    chmod g+rwx $S/ctl/v28_prepare.sh
    (cd $S/ctl && bash v28_prepare.sh > v28_prepare.out 2>&1)
    if grep -q "^rc=0" $S/ctl/v28_prepare.out && test -s $S/code/runs/ai_factory_all_v28/summary.json; then
        cp $S/ctl/QUEUE $S/ctl/QUEUE.v28 && echo "factory ai_factory_all_v28" >> $S/ctl/QUEUE.v28 && mv $S/ctl/QUEUE.v28 $S/ctl/QUEUE && chmod g+rw $S/ctl/QUEUE
        log "coverage: v28 queued on a08trc08"
        until done_or_stop $S/ctl/ai_factory_all_v28/controller.log; do sleep 300; done
        log "coverage: v28: $(tail -1 $S/ctl/ai_factory_all_v28/controller.log | cut -c1-200)"
        coverage_run run_2026-10-25b
        log "coverage: DONE"
    else
        log "coverage: STOP: v28 preparation failed (ctl/v28_prepare.out in shared_split)"
    fi
}

log "start"
score_branch & coverage_branch & wait
log "DONE: both branches finished"
