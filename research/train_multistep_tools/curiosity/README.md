# Fix 4 pipeline on Curiosity

Shell glue for training fix 4 (multi-step tool use, see `../PLAN.md`). The jobs run as user `eai-phshk`. The pipeline uses the 4 GPUs that the autopilot agent job does not use, as its own job on another node.

The pipeline has two roots:

| Root | Default | Use |
|---|---|---|
| `CP_RO` | `/home/eai-jjppv/cp` | Read only: `venv`, `models`, `exp/kdall/model` (student 1.3), `exp/kdall/{prompts.jsonl,teacher}`, eval data in `data`. |
| `WORK` | `~/mt` | All writes: `exp/mt_<NAME>`, `exp/mt_<NAME>_s<SEED>`, `runs`, `logs`, `hf`. The code is in `REPO=~/mt/alembic`. |

| Script | Work |
|---|---|
| `lib_mt.sh` | Roots, venv, CUDA, `log`, `serve`, and `stop_all`. The `stop_all` function stops only the servers of this pipeline. |
| `collect.sh` | Starts `vllm serve` for the teacher (TP 4, tool calls), then `task_bank.py`, `collect_sim.py`, `verify_traj.py`. |
| `build.sh` | Joins the 1.3 replay rows, then `build_agent_data.py` and `build_tool_kd.py` (`MODE=onehot` or `MODE=teacher`). |
| `train.sh` | `research/curiosity_b300/train_kd.py` from the 1.3 student (LR 1e-5, ALPHA 0.8, as kdreq13), then `research/curiosity_b300/eval.sh` with output in `$WORK/runs`. |
| `pipeline.sbatch` | One 4-GPU job: collect, build, train for one `SEED`. Log: `$WORK/logs/mt_<NAME>_s<SEED>.log`. |

All steps are resumable. A step with a complete output is skipped. The seeds share the data in `$WORK/exp/mt_<NAME>`, so only the first job collects.

Environment variables: `NAME` (default `v1`), `SEED` (default 42), `TEACHER` (`q235` with the `hermes` tool parser, or `gptoss120` with the `openai` tool parser), `MODE` (`onehot` or `teacher`), `LIMIT`, `SMOKE`, `CP_RO`, `WORK`, `REPO`.

## Setup

1. Make the work folders on Curiosity.

   ```bash
   mkdir -p ~/mt/{logs,exp}
   ```

2. Copy the repo from the laptop to `~/mt/alembic`.

   ```bash
   rsync -a --exclude .venv --exclude .git ./ eai-phshk@<curiosity login>:mt/alembic/
   ```

The autopilot queue (`/home/eai-jjppv/cp/queue`) is not writable for `eai-phshk`. Thus the seeds run as a chain of own jobs.

## Submit

1. Find the node of the agent job.

   ```bash
   AGENT=$(squeue -u eai-jjppv -n cp-agent -h -o %N)
   ```

2. Check that teammates do not need the GPUs.

   ```bash
   squeue -u eai-ajzxf,eai-jjppv
   ```

3. Run the smoke test first. It uses 50 tasks, 50 replay rows and no eval, so the training has only a few steps.

   ```bash
   S=~/mt/alembic/research/train_multistep_tools/curiosity/pipeline.sbatch
   sbatch --exclude=$AGENT --time=03:00:00 --output=$HOME/mt/logs/mt-%j.out --export=ALL,SEED=42,SMOKE=1 $S
   ```

4. Read `~/mt/logs/mt_v1_smoke_s42.log`. Make sure that it shows tasks, trajectories, verified rows, parts and `trained:`.

5. Submit the 3 seeds as a chain. Seeds 43 and 44 wait for the job before them, because only 4 GPUs are free.

   ```bash
   O=--output=$HOME/mt/logs/mt-%j.out
   j42=$(sbatch --parsable $O --exclude=$AGENT --export=ALL,SEED=42 $S)
   j43=$(sbatch --parsable $O --exclude=$AGENT --dependency=afterok:$j42 --export=ALL,SEED=43 $S)
   sbatch $O --exclude=$AGENT --dependency=afterok:$j43 --export=ALL,SEED=44 $S
   ```

6. Report the 3-seed mean of `~/mt/runs/mt_v1_s<SEED>/RESULT.json`.
