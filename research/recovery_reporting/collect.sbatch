#!/bin/bash
#SBATCH --job-name=dk-recovery-report
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
cd "${REPO:?Set isolated REPO}"
PYTHON=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
"$PYTHON" research/ai_factory_pilot/collect_results.py runs/ai_factory_recovery_20261014
"$PYTHON" - <<'REPORT'
import hashlib,json
from pathlib import Path
root=Path('runs/ai_factory_recovery_20261014')
path=root/'MULTISEED_RESULT.json'
result=json.loads(path.read_text())
for key,source in [
    ('separate_document_assisted_baseline_2048',Path('runs/retrieval_2048_20261014/retrieval_repaired/BASE_RAG_RESULT.json')),
    ('mentor_real_pair_agreement',root/'mentor_followup/REAL_PAIR_AGREEMENT.json')]:
    result[key]={'status':'not_available'}
    if source.exists():
        result[key]={'artifact':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'result':json.loads(source.read_text())}
result['recovery_provenance']=json.loads((root/'combination_binding.json').read_text())
path.write_text(json.dumps(result,indent=2))
print('Recovery aggregate saved; no automatic release approval')
REPORT
