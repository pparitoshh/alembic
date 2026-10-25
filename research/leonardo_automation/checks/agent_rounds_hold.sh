#!/bin/bash
# Hold all factory submissions (shared_split/ctl/agent_rounds/WANT_SLOT) while agent round r49e trains (Oct 25: its
# second training job lost its slot to filler on this account).
S=/leonardo_work/EUHPC_D30_031/shared_split
until [ "$(ls $S/code/runs/r49e/train/seed*/TRAINING_DONE.json 2>/dev/null | wc -l)" -ge 3 ]; do touch $S/ctl/agent_rounds/WANT_SLOT; sleep 60; done
rm -f $S/ctl/agent_rounds/WANT_SLOT
