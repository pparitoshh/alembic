#!/bin/bash
# Dry run of the not-yet-run stages: batch-1 merge (to release_dry) and the round-4/5 data builders.
set -euo pipefail
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
cd $R/code
export PYTHONPATH=src:research/rag65 HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1
V=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
B7=/leonardo_work/EUHPC_D30_031/runs/a08trc09/student-mixed-v2-20261015T042338Z/bundle/data/b7_original.jsonl
D=$R/dryrun; rm -rf $D; mkdir -p $D
RELEASE_DIR=release_dry $V research/ai_factory_pilot/merge_factory_v1.py 200 $B7 runs/ai_factory_ccby_v1/release/verified.jsonl runs/ai_factory_ccby_v2/release/verified.jsonl | tail -c 1500
$V research/rag65/build_v4.py runs/rag65v2/RESULT.json runs/rag65v2/data/verified.jsonl runs/rag65v3/RESULT.json runs/rag65v3/data/verified.jsonl runs/ai_factory_all_v1/release_dry/verified.jsonl $D/v4 | head -c 700
# Stand-in for the top-up release: batch-1 rows under new ids.
$V -c "import json,sys
with open('$D/fake_v2.jsonl','w') as f:
    for l in open('runs/ai_factory_all_v1/release_dry/verified.jsonl'):
        r=json.loads(l); r['id']='dry/'+r['id']; f.write(json.dumps(r)+'\n')"
$V research/rag65/build_v5.py $D/v4/verified.jsonl $D/fake_v2.jsonl $D/v5 | head -c 500
echo DRYRUN_OK
