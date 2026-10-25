#!/bin/bash
# GGUF export of a merged student (default: round 5, seed 43, x0.2; MERGED/OUT override): bf16 -> Q8_0, Q4_K_M, Q3_K_M.
# CPU only (lrd_all_serial). Each step is skipped when its output exists. Writes runs/gguf/EXPORT.json.
set -euo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
QUANT=/leonardo_work/EUHPC_D30_031/tools/llama.cpp/bin/llama-quantize
M=${MERGED:-$C/runs/dpo/merged_r5}  # merged HF model to convert
O=${OUT:-$C/runs/gguf}
mkdir -p $O
cd $C
[ -s $O/model-bf16.gguf ] || PYTHONPATH=$R/llama.cpp/gguf-py $VPY $R/llama.cpp/convert_hf_to_gguf.py $M --outtype bf16 --outfile $O/model-bf16.gguf
for q in Q8_0 Q4_K_M Q3_K_M; do
    [ -s $O/model-$q.gguf ] || $QUANT $O/model-bf16.gguf $O/model-$q.gguf $q 4
done
python3 - <<PY
import json, os
o = '$O'
json.dump({f: round(os.path.getsize(f'{o}/{f}') / 2**30, 2) for f in sorted(os.listdir(o)) if f.endswith('.gguf')},
          open(f'{o}/EXPORT.json', 'w'), indent=2)
print(open(f'{o}/EXPORT.json').read())
PY
