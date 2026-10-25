#!/bin/bash
# Round 48 (Oct 25, d1 push): requeue/signals/preemption/checkpoint/array questions + Slurm, on-policy distillation from Qwen3-235B on round 27 (r47 recipe), judged with variant e vs round 27 + e.
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
cd $R/code && mkdir -p runs/r48/data && nice -n 10 python3 research/round48_build.py runs/r48/data/source.jsonl >> $R/ctl/r48_build.log 2>&1
RD=r48 FINAL_VARIANT=e FINAL_REL=runs/r27e/answers bash $R/ctl/r47.sh
