#!/bin/bash
# Oct 25: 0.75 x round 27 + 0.25 x round 43 (second point of the mid2743 sweep); weighted average of two full models, round 27 (release) and round 43 (tools soup on round 27). Equal-1536 +
# variant e: round 43 0.905 on v3 but 0.693 on d1, round 27 0.886 / 0.710. Answered with variant e at 1536 tokens and
# judged head to head against round 27 + e (runs/r27e). Log: shared_split/ctl/q2743/controller.log + ctl/vsb_q2743.log
set -uo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code; S=/leonardo_work/EUHPC_D30_031/shared_split
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
OUTM=$C/runs/q2743/merged; mkdir -p $OUTM $S/ctl/q2743
log() { echo "$(date -u +%FT%TZ) $*" >> $S/ctl/q2743/controller.log; }
if [ ! -s $OUTM/model.safetensors ]; then
  log "averaging round 27 and round 43 (CPU partition)"
  srun -A EUHPC_D30_031 -p lrd_all_serial -c 8 --mem=30G -t 01:00:00 -J dk-mid-soup $VPY - $S/models/r27_merged $C/runs/soup43/merged $OUTM > $S/ctl/q2743/soup.out 2>&1 <<'PY'
import sys, shutil, json
from pathlib import Path
from safetensors.torch import load_file, save_file
a, b, out = map(Path, sys.argv[1:4])
A = load_file(str(a / 'model.safetensors')); B = load_file(str(b / 'model.safetensors'))
assert A.keys() == B.keys()
M = {k: (0.75 * A[k].float() + 0.25 * B[k].float()).to(A[k].dtype) for k in A}
save_file(M, str(out / 'model.safetensors'), metadata={'format': 'pt'})
for f in a.iterdir():
    if f.name not in ('model.safetensors', 'SOUP.json'):
        shutil.copy(f, out / f.name)
json.dump({'average_of': [str(a), str(b)], 'weights': [0.75, 0.25]}, open(out / 'SOUP.json', 'w'), indent=1)
print('ok', len(M))
PY
  test -s $OUTM/model.safetensors || { log "STOP: average failed: $(tail -3 $S/ctl/q2743/soup.out)"; exit 1; }
fi
log "merged; answering with variant e at 1536 tokens"
ANSWER_MAX_TOKENS=1536 VARIANT=e bash $R/ctl/ragc.sh q2743 $OUTM
REL_ANSWERS=runs/r27e/answers REL_ARM=base_rag bash $R/ctl/vs_newret.sh q2743
