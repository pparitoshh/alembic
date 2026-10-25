#!/bin/bash
# Open-model leaderboard on our HPC eval (v2 66 / v3 200 / d1 106): open models answer exactly like our student
# (same retrieved excerpts, system prompt, greedy decoding; research/leaderboard/answer.sbatch), then every arm is
# judged against closed-book base Qwen3-4B with BOTH judges (gpt-oss-20b and Gemma 4 26B) in one run each.
# References: our release 12e (rel12e), base+RAG, the teacher Qwen3-Next-80B, Qwen3-235B and Qwen3-32B.
# gpt-oss models answer with reasoning on (their answers can be empty otherwise); gpt-oss-20b is also the main judge,
# so its row is read from the Gemma judge (and Gemma 4 from gpt-oss). Starts after round 15 (rag65kd11) so score rounds keep priority.
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; HERE="$(cd "$(dirname "$0")" && pwd)"
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
D=runs/leaderboard; A=$D/answers
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/controller.log"; }
cd $C; mkdir -p $A
until grep -q '^OK Qwen/Qwen3-14B\|^FAIL Qwen/Qwen3-14B' "$HERE/download.log" 2>/dev/null; do sleep 60; done
log "downloads: $(grep -c '^OK' "$HERE/download.log") ok, $(grep -c '^FAIL' "$HERE/download.log") failed $(grep '^FAIL' "$HERE/download.log" | cut -c1-80 | tr '\n' ' ')"
until grep -q ' DONE: \| STOP: ' $R/ctl/rag65kd11/controller.log 2>/dev/null; do sleep 120; done
ok() { grep -q "^OK $1 " "$HERE/download.log" || [ "$1" = openai/gpt-oss-20b ] || [ "$1" = openai/gpt-oss-120b ] || [ "$1" = google/gemma-4-26B-A4B-it ]; }
complete() { for l in v2:66 v3:200 d1:106; do [ "$(cat $A/${l%%:*}__$1_rag.jsonl 2>/dev/null | wc -l)" -ge ${l##*:} ] || return 1; done; }
MODEL_GROUPS=("lb_q8=Qwen/Qwen3-8B:1:0 lb_phi4mini=microsoft/Phi-4-mini-instruct:1:0 lb_granite8b=ibm-granite/granite-3.3-8b-instruct:1:0 lb_smol3b=HuggingFaceTB/SmolLM3-3B:1:0"
        "lb_q14=Qwen/Qwen3-14B:1:0 lb_gptoss20=openai/gpt-oss-20b:1:1"
        "lb_gemma4=google/gemma-4-26B-A4B-it:4:0"
        "lb_gptoss120=openai/gpt-oss-120b:4:1")
for g in "${MODEL_GROUPS[@]}"; do
    for attempt in 1 2 3; do
        todo=""; for spec in $g; do arm=${spec%%=*}; m=$(echo ${spec#*=} | cut -d: -f1); ok $m && ! complete $arm && todo="$todo $spec"; done
        [ -z "$todo" ] && break
        log "answer attempt $attempt:$todo"
        REPO=$C bash $R/ctl/priority_submit.sh --export=ALL,MODELS="$(echo $todo)",OUT=$A research/leaderboard/answer.sbatch
    done
done
for f in runs/rag65kd9/answers/*__base_closed.jsonl runs/rag65kd9/answers/*__base_rag.jsonl; do cp -n $f $A/; done
for f in runs/rag65kd9/answers/*__s44_rag.jsonl; do b=$(basename $f); cp -n $f $A/${b/__s44_rag/__rel12e_rag}; done
for f in runs/teacher_probe2/answers/*__tnext_rag.jsonl runs/teacher_probe4/answers/*__t235_rag.jsonl runs/teacher_probe/answers/*__t235_rag.jsonl runs/teacher_probe/answers/*__t32_rag.jsonl; do [ -s $f ] && cp -n $f $A/; done
log "arms: $(ls $A | grep '^v3__' | sed 's/^v3__//; s/\.jsonl$//' | tr '\n' ' ')"
left() { $VPY - <<PY | tail -1
from pathlib import Path
a = Path('$A'); j = Path('$D/$1'); need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
for attempt in 1 2 3 4; do
    [ "$(left judged)" = 0 ] && break
    log "gpt-oss judge attempt $attempt ($(left judged) units left)"
    REPO=$C RUN=leaderboard bash $R/ctl/priority_submit.sh --nodes=4 --export=ALL,STAGE=judge,RUN=leaderboard research/rag65/stage.sbatch
done
for attempt in 1 2 3 4; do
    [ "$(left judged_gemma)" = 0 ] && break
    log "gemma judge attempt $attempt ($(left judged_gemma) units left)"
    REPO=$C RUN=leaderboard bash $R/ctl/priority_submit.sh --nodes=3 --export=ALL,RUN=leaderboard research/teacher_probe/judge_gemma.sbatch
done
export PYTHONPATH=src:research/rag65
$VPY research/rag65/aggregate.py $D/judged $A $D/RESULT.json > $D/aggregate.out 2>&1
$VPY research/rag65/aggregate.py $D/judged_gemma $A $D/RESULT_gemma.json > $D/aggregate_gemma.out 2>&1
$VPY - > "$HERE/table.txt" <<PY
import json
r1 = json.load(open('$D/RESULT.json'))['arms']; r2 = json.load(open('$D/RESULT_gemma.json'))['arms']
arms = sorted({k.split('/', 1)[1] for k in r1})
rows = []
for a in arms:
    g = [r1.get(f'{l}/{a}', {}).get('win_rate') for l in ('v3', 'v2', 'd1')]
    m = [r2.get(f'{l}/{a}', {}).get('win_rate') for l in ('v3', 'v2', 'd1')]
    mean = lambda xs: sum(xs) / 3 if all(x is not None for x in xs) else None
    rows.append((a, g, mean(g), m, mean(m)))
rows.sort(key=lambda r: -(r[4] or 0))
fmt = lambda x: '  -  ' if x is None else f'{x:.3f}'
print('arm                gpt-oss: v3    v2    d1   mean | gemma: v3    v2    d1   mean')
for a, g, gm, m, mm in rows:
    print(f'{a:18s} ' + ' '.join(fmt(x) for x in g) + f' {fmt(gm)} | ' + ' '.join(fmt(x) for x in m) + f' {fmt(mm)}')
PY
log "DONE: $(tr '\n' '|' < "$HERE/table.txt" | cut -c1-3000)"
