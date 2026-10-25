#!/bin/bash
# Score work first (Oct 25): hold factory submissions (ctl/PRIORITY, as priority_submit.sh does) while round 48 trains,
# and resume it with MIN_ROWS=2000 if it stops again at the row floor (2,334 student answers < 2,400).
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017; L=$R/ctl/r48/controller.log
resumed=0
until grep -q ' DONE: round 47 soup\|DONE: round\| STOP: KD\| STOP: soup\| STOP: only' $L 2>/dev/null || [ -s $R/ctl/vsb_r48.log ]; do
  [ -e $R/ctl/PRIORITY ] || echo 'round 48 (score work first)' > $R/ctl/PRIORITY
  if [ $resumed = 0 ] && [ "$(grep -c 'STOP: too few student answers' $L)" -ge 2 ] && ! pgrep -f '^bash $R/ctl/r47.sh' > /dev/null; then
    echo "$(date -u +%FT%TZ) resuming with MIN_ROWS=2000 (2,334 student answers)" >> $L
    (RD=r48 MIN_ROWS=2000 FINAL_VARIANT=e FINAL_REL=runs/r27e/answers setsid nohup bash $R/ctl/r47.sh >> $R/ctl/r48.nohup 2>&1 < /dev/null &)
    resumed=1
  fi
  sleep 60
done
grep -q 'round 48' $R/ctl/PRIORITY 2>/dev/null && rm -f $R/ctl/PRIORITY
