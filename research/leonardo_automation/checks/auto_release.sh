#!/bin/bash
# Automatic release decision and build after vs_release.sh RD: a seed of round RD becomes the release candidate when,
# against the current release (ctl/RELEASE), v3 improves significantly (p < 0.05) under the 2-of-3 majority vote of three judges
# and its mean over v3/v2/d1 is not worse (no single judge worse by more than 0.01); without the third judge, the
# earlier rule: v3 better under both judges, significant under one, mean not worse under either. The best such arm is merged, exported (bf16 / Q8_0 / Q4_K_M),
# given an imatrix Q4_K_M, its quantized files are evaluated against it, and runs/hf_<RD>/ is staged with READY.
# The Hugging Face push itself is done from the laptop (the token never sits on the cluster).
# usage: auto_release.sh ROUND_DIR    log: ctl/auto_release_<RD>.log
set -uo pipefail
RD=$1
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
LOG=$R/ctl/auto_release_$RD.log; log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
cd $C
V=runs/vs_release_$RD
decision=$(python3 - <<PY
import json
g = json.load(open('$V/PAIRED.json')); m = json.load(open('$V/PAIRED_gemma.json'))
import os
maj = json.load(open('$V/PAIRED_majority.json')) if os.path.exists('$V/PAIRED_majority.json') else None
arms = sorted({k.split('/', 1)[1].split('-vs-')[0] for k in g if k.endswith('-vs-rel_rag')})
best = None
for a in arms:
    d = lambda P, l: P.get(f'{l}/{a}-vs-rel_rag')
    v3g, v3m = d(g, 'v3'), d(m, 'v3')
    if not v3g or not v3m:
        continue
    mg = sum(d(g, l)['diff'] for l in ('v3', 'v2', 'd1')) / 3
    mm = sum(d(m, l)['diff'] for l in ('v3', 'v2', 'd1')) / 3
    win = v3g['diff'] > 0 and v3m['diff'] > 0 and min(v3g['p_a_not_better'], v3m['p_a_not_better']) < 0.05 and mg >= 0 and mm >= 0
    if maj and d(maj, 'v3'):  # majority vote of three judges decides when available (Paritosh, Oct 22: judges may favour length)
        v3j = d(maj, 'v3'); mj = sum(d(maj, l)['diff'] for l in ('v3', 'v2', 'd1')) / 3
        win = v3j['diff'] > 0 and v3j['p_a_not_better'] < 0.05 and mj >= 0 and mg >= -0.01 and mm >= -0.01
        print(f'#   majority: v3 {v3j["diff"]:+.3f} p {v3j["p_a_not_better"]:.3f} mean {mj:+.3f}')
    print(f'# {a}: v3 {v3g["diff"]:+.3f}/{v3m["diff"]:+.3f} p {v3g["p_a_not_better"]:.3f}/{v3m["p_a_not_better"]:.3f} mean {mg:+.3f}/{mm:+.3f} win={win}')
    if win and (best is None or mg + mm > best[1]):
        best = (a, mg + mm)
print('WIN ' + best[0] if best else 'NOWIN')
PY
)
log "decision:"; echo "$decision" >> $LOG
arm=$(echo "$decision" | grep '^WIN ' | cut -d' ' -f2)
[ -n "$arm" ] || { log "NOWIN: the release stays as in ctl/RELEASE"; exit 0; }
seed=$(echo $arm | sed -E 's/^s([0-9]+).*/\1/'); x05=$(echo $arm | grep -q '_x05_' && echo _x05 || echo '')
adapter=runs/$RD/train/seed$seed/adapter$x05
base=$(grep -m1 '^MERGED=' $R/ctl/$RD/controller.sh | sed -E 's/^MERGED=([^ ]*).*/\1/' | sed "s#\$C#$C#")
log "WIN $arm: adapter $adapter, base ${base:-Qwen/Qwen3-4B-Instruct-2507}"
G=runs/gguf_$RD; E=runs/gguf_${RD}_eval; mkdir -p $G $E/answers
if [ ! -s $G/model-Q4_K_M.gguf ]; then
    srun -A EUHPC_D30_031 -p lrd_all_serial -c 4 --mem=30G -t 02:00:00 -J dk-gguf-$RD bash -c "export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1; [ -s $G/merged/config.json ] || ${base:+BASE=$base} $VPY research/dpo/merge.py $adapter $G/merged && MERGED=$C/$G/merged OUT=$C/$G bash research/gguf/export.sh" > $R/ctl/auto_release_$RD.export.out 2>&1
    test -s $G/model-Q4_K_M.gguf || { log "STOP: export failed"; exit 1; }
fi
cp -n runs/gguf/calibration.txt $G/calibration.txt 2>/dev/null
for attempt in 1 2 3; do
    test -s $G/model-Q4_K_M-imat.gguf && break
    log "imatrix attempt $attempt"; REPO=$C OUT=$G bash $R/ctl/priority_submit.sh --export=ALL,OUT=$G research/gguf/imatrix.sbatch
done
test -s $G/model-Q4_K_M-imat.gguf || { log "STOP: no imatrix quant"; exit 1; }
cnt() { cat $E/answers/*__cand_gguf_*_rag.jsonl 2>/dev/null | wc -l; }
for attempt in 1 2 3; do
    [ "$(cnt)" -ge 744 ] && break
    log "gguf answer attempt $attempt ($(cnt)/744)"
    REPO=$C bash $R/ctl/priority_submit.sh --export=ALL,QUANTS="Q8_0 Q4_K_M-imat",ARM=cand_gguf,GGUF_DIR=$G,ANSWERS=$E/answers research/gguf/answer.sbatch
done
cp -n runs/$RD/answers/*__base_closed.jsonl runs/$RD/answers/*__base_rag.jsonl $E/answers/
for f in runs/$RD/answers/*__${arm}.jsonl; do b=$(basename $f); cp -n $f $E/answers/${b/__${arm}/__cand_vllm_rag}; done
left() { $VPY - <<PY | tail -1
from pathlib import Path
a = Path('$E/answers'); j = Path('$E/judged'); need = 0
for label, n in [('v2', 66), ('v3', 200), ('d1', 106)]:
    for p in a.glob(label + '__*.jsonl'):
        arm = p.name[len(label) + 2:-6]
        if arm != 'base_closed':
            need += sum(not (j / f'{label}__{arm}__{s:03d}.jsonl').exists() for s in range(0, n, 34))
print(need)
PY
}
for attempt in 1 2 3; do
    [ "$(left)" = 0 ] && break
    log "gguf judge attempt $attempt ($(left) units left)"
    REPO=$C RUN=gguf_${RD}_eval bash $R/ctl/priority_submit.sh --nodes=2 --export=ALL,STAGE=judge,RUN=gguf_${RD}_eval research/rag65/stage.sbatch
done
export PYTHONPATH=src:research/rag65
$VPY research/rag65/aggregate.py $E/judged $E/answers $E/RESULT.json > $E/aggregate.out 2>&1
$VPY research/rag65/paired.py $E/judged $E/PAIRED.json cand_gguf_q80_rag:cand_vllm_rag cand_gguf_q4kmimat_rag:cand_vllm_rag cand_gguf_q4kmimat_rag:base_rag > $E/paired.out 2>&1
H=runs/hf_$RD; mkdir -p $H/gguf
for f in config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja; do cp $G/merged/$f $H/ 2>/dev/null; done
ln -sf $C/$G/merged/model.safetensors $H/model.safetensors
ln -sf $C/$G/model-Q4_K_M-imat.gguf $H/gguf/hpc-assistant-qwen3-4b-Q4_K_M.gguf
ln -sf $C/$G/model-Q8_0.gguf $H/gguf/hpc-assistant-qwen3-4b-Q8_0.gguf
{ cp runs/$RD/RESULT.json $H/eval_RESULT.json 2>/dev/null || $VPY research/rag65/aggregate.py $V/judged $V/answers $H/eval_RESULT.json > /dev/null 2>&1; }; cp $V/PAIRED.json $H/eval_PAIRED_vs_previous.json; cp $V/PAIRED_gemma.json $H/eval_PAIRED_gemma_judge.json
cp $E/PAIRED.json $H/gguf/eval_PAIRED_gguf.json
cp runs/hf_r12e/gguf/Modelfile $H/gguf/Modelfile 2>/dev/null
echo "$RD $arm" > $H/READY
log "READY: $H staged for Hugging Face ($arm); README and push from the laptop"
