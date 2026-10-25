#!/bin/bash
# Round 49b (Oct 25): when the agent collection jobs end, build data from ALL verified trajectories (agent_traj4 +
# agent_traj6, the latter with the teacher hint), train on this account (run_round.sh), then agent_eval.sh.
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017; S=/leonardo_work/EUHPC_D30_031/shared_split
while squeue --me -h -n dk-agent-collect | grep -q .; do sleep 60; done
cd $R/code && PYTHONPATH=src /leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python research/agent_train/build_agent_data.py $S/code/runs/r49b $S/code/runs/tl38/data_src.jsonl runs/agent_traj4/traj_*.jsonl runs/agent_traj6/traj_*.jsonl --upsample 3 2>&1 | tail -1 >> $S/ctl/r49b/prep.log
chmod g+r $S/code/runs/r49b/data_src.jsonl $S/ctl/r49b/prep.log
(setsid nohup bash $R/ctl/agent_eval.sh r49b > /dev/null 2>&1 < /dev/null &)
bash $S/ctl/run_round.sh r49b
