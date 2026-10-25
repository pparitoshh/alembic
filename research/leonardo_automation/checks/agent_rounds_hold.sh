#!/bin/bash
# Hold factory filler on THIS account only (ctl/PRIORITY, honoured by run_factory.sh for a08trc09) while round r49e
# trains; a08trc08 keeps its filler (a global WANT_SLOT hold left its 8 nodes idle, Oct 25).
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017; S=/leonardo_work/EUHPC_D30_031/shared_split
until [ "$(ls $S/code/runs/r49e/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)" -ge 3 ]; do
    [ -e $R/ctl/PRIORITY ] || echo "agent round r49e (score work first)" > $R/ctl/PRIORITY; sleep 60
done
grep -q "agent round r49e" $R/ctl/PRIORITY 2>/dev/null && rm -f $R/ctl/PRIORITY
