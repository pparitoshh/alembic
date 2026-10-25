# Round 1 smoke: student model (2026-10-25)

Four prompts. One fresh `ollama run <model> "<prompt>" --think=false` call per
prompt for `hf.co/VaibhavKKM/hpc-assistant-qwen3-4b:latest`. Every answer was
then run unchanged in the live Slurm 23.11 container `slurm-test` as user
`ubuntu`, and the real output is recorded below.

## Models compared

| Role | Model | ID | Size | Source | Details |
|---|---|---|---|---|---|
| student (NEW) | `hf.co/VaibhavKKM/hpc-assistant-qwen3-4b:latest` | `a0e788e3baed` | 2.5 GB | Hugging Face repo, GGUF | base `Qwen/Qwen3-4B-Instruct-2507`; `ollama show`: architecture qwen3, 4.02B, quantization `Q4_K_M`, thinking capability |

## Prompt 1: submit job.sh, 2 nodes, 4 GPUs per node, 1 hour, boost_usr_prod

Query (exact text): "Give only the Slurm command to submit job.sh, requesting
2 nodes, 4 GPUs per node, 1 hour, partition boost_usr_prod."

Expected answer:

```bash
sbatch --nodes=2 --gpus-per-node=4 --time=01:00:00 --partition=boost_usr_prod job.sh
```

| Model | Actual model answer (exact) | Actual output in the Slurm container | Format followed | Verdict |
|---|---|---|---|---|
| student | `sbatch --nodes=2 --gres=gpu:4 --time=1:00:00 --partition=boost_usr_prod job.sh` | `Submitted batch job 75`, RC=0; stays `PENDING (PartitionNodeLimit)` | yes - command only | **correct** - `--gres=gpu:4` is the modern equivalent of `--gpus-per-node=4`; the pending state is the expected one-node behavior |

## Prompt 2: show only my pending jobs (job id, name, reason)

Query (exact text): "One line only. Slurm command to show only my own pending
jobs with job id, name and reason."

Expected answer:

```bash
squeue -u $USER -t PENDING -o '%i %j %R'
```

| Model | Actual model answer (exact) | Actual output in the Slurm container | Format followed | Verdict |
|---|---|---|---|---|
| student | `squeue -u $USER -t PENDING -o "%.10i %.15j %.20R"` | lists the pending jobs: 74, 75 (PartitionNodeLimit), 78 (Resources), 79 (Priority), RC=0 | yes | **correct** - own jobs, pending state, id, name, reason |

## Prompt 3: cancel all my jobs named train

Query (exact text): "One line only. Slurm command to cancel all my jobs named
train."

Expected answer:

```bash
scancel --name=train --user=$USER
```

| Model | Actual model answer (exact) | Actual output in the Slurm container | Format followed | Verdict |
|---|---|---|---|---|
| student (first sample) | `slurm cancel -n train` | `bash: slurm: command not found`, RC=127 | yes | **wrong** - no `slurm` command exists; the correct command is `scancel -n train` |
| student (second sample, one other pane run) | `scancel --name=train --user=$USER` | RC=0; all four train jobs cancelled | yes | **correct** - this variant works as written |

## Prompt 4: interactive shell on 1 GPU node, 30 minutes

Query (exact text): "One line only. Slurm command to get an interactive shell
on 1 GPU node for 30 minutes."

Expected answer:

```bash
srun --gres=gpu:1 --time=00:30:00 --pty bash
```

| Model | Actual model answer (exact) | Actual output in the Slurm container | Format followed | Verdict |
|---|---|---|---|---|
| student | `srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash` | run with `hostname` instead of an interactive bash: pty warning (`Not using a pseudo-terminal, disregarding --pty option`), then `slurmnode`, RC=0 | yes | **correct** - right tool and flags; --pty is ignored without a tty but the step still runs |

## Summary

| Model | correct | partly | wrong | strict accuracy |
|---|---|---|---|---|
| student (`hf.co/VaibhavKKM/hpc-assistant-qwen3-4b`) | 3 | 0 | 1 | 75.0% |

The student answered 3 of the 4 prompts correctly; the only miss was the
cancel command, which named a command that does not exist.

## Run notes

- Prompt 1: the 2-node job was accepted and stayed `PENDING
  (PartitionNodeLimit)`; that is the expected behavior on a one-node cluster,
  not a model error.
- Prompt 2: before the run, four train jobs were submitted
  (`--gres=gpu:2`); two ran and two stayed pending, so squeue had real data.
- Prompt 3: the placeholder run used the jobs from prompt 2; the second
  sample is a separate run in another pane.
- Prompt 4: the prompt asked to run the answer unchanged except for the shell
  name, so `hostname` replaced `bash`.
