#!/bin/bash
# Round 49e (Oct 25): round 49c scored 9/10 (only debug failed: no debug examples, the thinking-off teacher never read
# the log). When the thinking-on, debug-focused collection (agent_traj7) ends: data = traj7 x4 + traj4/6 x1, train on
# round 49c (this account, run_round.sh), then agent_eval.sh (soup into round 49c, smoke test).
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017; S=/leonardo_work/EUHPC_D30_031/shared_split
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
until [ -s $R/code/runs/agent_traj7/fixtures.json ] && ! squeue --me -h -n dk-agent-collect | grep -q .; do sleep 60; done
cd $R/code; export PYTHONPATH=src
$VPY research/agent_train/build_agent_data.py runs/r49e_new $S/code/runs/tl38/data_src.jsonl runs/agent_traj7/traj_*.jsonl --upsample 4 --n-replay 0 > $R/ctl/r49e_build.log 2>&1
$VPY research/agent_train/build_agent_data.py runs/r49e_old $S/code/runs/tl38/data_src.jsonl runs/agent_traj4/traj_*.jsonl runs/agent_traj6/traj_*.jsonl --upsample 1 --n-replay 0 >> $R/ctl/r49e_build.log 2>&1
cat runs/r49e_new/data_src.jsonl runs/r49e_old/data_src.jsonl | shuf --random-source=<(yes) > $S/code/runs/r49e/data_src.jsonl
echo "INPUT READY: $(head -1 $R/ctl/r49e_build.log | cut -c14-120) | $(wc -l < $S/code/runs/r49e/data_src.jsonl) rows" >> $S/ctl/r49e/prep.log
chmod g+r $S/code/runs/r49e/data_src.jsonl $S/ctl/r49e/prep.log
(setsid nohup bash $R/ctl/agent_eval.sh r49e > /dev/null 2>&1 < /dev/null &)
bash $S/ctl/run_round.sh r49e
