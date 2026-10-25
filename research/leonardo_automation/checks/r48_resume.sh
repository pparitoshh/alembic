#!/bin/bash
# If round 48 stops at the 6,000-row floor (it has 2,668 rows), resume it with MIN_ROWS=2400 (r47.sh is resumable).
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
until grep -q " STOP: \| DONE: " $R/ctl/r48/controller.log 2>/dev/null; do sleep 60; done
if grep -q "STOP: too few student answers" $R/ctl/r48/controller.log; then
  echo "$(date -u +%FT%TZ) resuming with MIN_ROWS=2400 (targeted round: 668 topic + 2,000 Slurm rows)" >> $R/ctl/r48/controller.log
  RD=r48 MIN_ROWS=2400 FINAL_VARIANT=e FINAL_REL=runs/r27e/answers bash $R/ctl/r47.sh
fi
