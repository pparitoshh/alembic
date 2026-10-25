# Round 1 smoke: hpc-assistant-x02 vs hpc-assistant-qwen3-4b (2026-10-25)

Four prompts, one fresh `ollama run <model> "<prompt>" --think=false` call per
prompt, no interaction. Models: old `hpc-assistant-x02:latest`, new
`hf.co/VaibhavKKM/hpc-assistant-qwen3-4b:latest`. Scores: new model 3 of 4
correct with one-line answers; old model 1 of 4 with code-fence answers.

## 1. Submit job.sh, 2 nodes, 4 GPUs per node, 1 h, partition boost_usr_prod

| Model | Exact answer | Verdict |
|---|---|---|
| old | `sbatch --nodes=2 --gpus-per-node=4 --time=01:00:00 --partition=boost_usr_prod job.sh` (in a ```bash fence) | correct |
| new | `sbatch --nodes=2 --gres=gpu:4 --time=1:00:00 --partition=boost_usr_prod job.sh` | correct |

Real container check: `sbatch --nodes=2` is accepted but stays PENDING with
reason `(PartitionNodeLimit)` on the one-node cluster. No submit-time error.

## 2. One line: show only my pending jobs (job id, name, reason)

| Model | Exact answer | Verdict |
|---|---|---|
| old | `scontrol show job <your_job_id> --format="JobId,Name,Reason"` (fence) | wrong - shows one job by id, not my pending jobs; needs a placeholder |
| new | `squeue -u $USER -t PENDING -o "%.10i %.15j %.20R"` | correct |

Real container check: `squeue -u $USER -t PENDING -o '%i %j %R'` works and
lists pending jobs with name and reason.

## 3. One line: cancel all my jobs named train

| Model | Exact answer | Verdict |
|---|---|---|
| old | `scontrol cancel --jobs "Name=train"` (fence) | wrong - `scontrol cancel` does not exist |
| new | `slurm cancel -n train` | wrong - there is no `slurm` command; the correct command is `scancel -n train`. In one other run the new model gave `scancel --name=train --user=$USER`, which is correct. |

Real container check: `scancel --name=train --user=$USER` works (RC=0,
cancels all matching jobs); `scontrol cancel --jobs Name=train` fails with
`scontrol: unrecognized option '--jobs'`; `slurm cancel ...` fails with
`command not found`.

## 4. One line: interactive shell on 1 GPU node, 30 min

| Model | Exact answer | Verdict |
|---|---|---|
| old | `sbatch --time=0:30:00 --gpus-per-node=1 --ntasks=1 --cpus-per-task=1 --partition=gpu --nodes=1 --interactive` (fence) | wrong - sbatch does not give a shell |
| new | `srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash` | correct |

Real container check: `srun --partition=gpu --gres=gpu:1 --time=00:30:00
hostname` runs and prints the node name. With `--pty` but no client tty,
srun warns `Not using a pseudo-terminal, disregarding --pty option` and still
runs. With a tty it allocates /dev/pts/N and works.

## Verdict

- New model: 3 of 4 correct, clean one-line format every time.
- Old model: 1 of 4 correct, always wraps answers in markdown fences.
- Common failure: cancel by job name. Train `scancel -n train` next.
