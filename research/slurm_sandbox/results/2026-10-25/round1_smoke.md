# Round 1 smoke: OLD vs NEW student (2026-10-25)

Four prompts. One fresh `ollama run <model> "<prompt>" --think=false` call per
prompt. Every answer was then run unchanged in the live Slurm 23.11 container
`slurm-test` as user `ubuntu`, and the real output is recorded below.

## Models compared

| Role | Model | ID | Size | Source | Details |
|---|---|---|---|---|---|
| OLD | `hpc-assistant-x02:latest` | `6dcc7319e329` | 2.5 GB | local Ollama model | `ollama show`: architecture qwen3, 4.0B, quantization `Q4_K_M`, context 262144. No explicit base-model field is shown by `ollama show`. |
| NEW | `hf.co/VaibhavKKM/hpc-assistant-qwen3-4b:latest` | `a0e788e3baed` | 2.5 GB | Hugging Face repo, GGUF | base `Qwen/Qwen3-4B-Instruct-2507`; `ollama show`: architecture qwen3, 4.02B, quantization `Q4_K_M`, thinking capability |

## Prompt 1: submit job.sh, 2 nodes, 4 GPUs per node, 1 hour, boost_usr_prod

Query (exact text): "Give only the Slurm command to submit job.sh, requesting
2 nodes, 4 GPUs per node, 1 hour, partition boost_usr_prod."

Expected answer:

```bash
sbatch --nodes=2 --gpus-per-node=4 --time=01:00:00 --partition=boost_usr_prod job.sh
```

| Model | Actual model answer (exact) | Actual output in the Slurm container | Format followed | Verdict |
|---|---|---|---|---|
| OLD | `sbatch --nodes=2 --gpus-per-node=4 --time=01:00:00 --partition=boost_usr_prod job.sh` (in a code fence) | `Submitted batch job 74`, RC=0; stays `PENDING (PartitionNodeLimit)` on the one-node cluster | no - wrapped in a bash fence | **correct** - all required flags; the pending state is the expected one-node behavior |
| NEW | `sbatch --nodes=2 --gres=gpu:4 --time=1:00:00 --partition=boost_usr_prod job.sh` | `Submitted batch job 75`, RC=0; stays `PENDING (PartitionNodeLimit)` | yes - command only | **correct** - `--gres=gpu:4` is the modern equivalent |

## Prompt 2: show only my pending jobs (job id, name, reason)

Query (exact text): "One line only. Slurm command to show only my own pending
jobs with job id, name and reason."

Expected answer:

```bash
squeue -u $USER -t PENDING -o '%i %j %R'
```

| Model | Actual model answer (exact) | Actual output in the Slurm container | Format followed | Verdict |
|---|---|---|---|---|
| OLD | `scontrol show job <your_job_id> --format="JobId,Name,Reason"` | run with a real id (78) instead of the placeholder: `scontrol: unrecognized option '--format=JobId,Name,Reason'`, RC=1 | partly - one line, but inside a fence | **wrong** - scontrol has no --format option, and the form shows one job, not my pending jobs |
| NEW | `squeue -u $USER -t PENDING -o "%.10i %.15j %.20R"` | lists the pending jobs: 74, 75 (PartitionNodeLimit), 78 (Resources), 79 (Priority), RC=0 | yes | **correct** - own jobs, pending state, id, name, reason |

## Prompt 3: cancel all my jobs named train

Query (exact text): "One line only. Slurm command to cancel all my jobs named
train."

Expected answer:

```bash
scancel --name=train --user=$USER
```

| Model | Actual model answer (exact) | Actual output in the Slurm container | Format followed | Verdict |
|---|---|---|---|---|
| OLD | `scontrol cancel --jobs "Name=train"` | `scontrol: unrecognized option '--jobs'`, RC=1 | partly - one line, but inside a fence | **wrong** - scontrol has no cancel command and no --jobs option |
| NEW (first sample) | `slurm cancel -n train` | `bash: slurm: command not found`, RC=127 | yes | **wrong** - no `slurm` command exists; the correct command is `scancel -n train` |
| NEW (second sample, one other pane run) | `scancel --name=train --user=$USER` | RC=0; all four train jobs cancelled | yes | **correct** - this variant works as written |

## Prompt 4: interactive shell on 1 GPU node, 30 minutes

Query (exact text): "One line only. Slurm command to get an interactive shell
on 1 GPU node for 30 minutes."

Expected answer:

```bash
srun --gres=gpu:1 --time=00:30:00 --pty bash
```

| Model | Actual model answer (exact) | Actual output in the Slurm container | Format followed | Verdict |
|---|---|---|---|---|
| OLD | `sbatch --time=0:30:00 --gpus-per-node=1 --ntasks=1 --cpus-per-task=1 --partition=gpu --nodes=1 --interactive` | `sbatch: unrecognized option '--interactive'`, RC=255 | partly - one line, but inside a fence | **wrong** - sbatch has no --interactive option; it cannot hand out a shell, and no script file was given |
| NEW | `srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash` | run with `hostname` instead of an interactive bash: pty warning (`Not using a pseudo-terminal, disregarding --pty option`), then `slurmnode`, RC=0 | yes | **correct** - right tool and flags; --pty is ignored without a tty but the step still runs |

## Summary

| Model | correct | partly | wrong | strict accuracy |
|---|---|---|---|---|
| NEW (`hf.co/VaibhavKKM/hpc-assistant-qwen3-4b`) | 3 | 0 | 1 | 75.0% |
| OLD (`hpc-assistant-x02`) | 1 | 0 | 3 | 25.0% |

The new model picks the right command family and answers in clean one-line
form; the old model often names a different tool or a non-existent option and
wraps answers in code fences. The one shared weakness is cancel by job name:
train `scancel -n train` next.

## Run notes

- P1: both 2-node jobs were accepted and stayed `PENDING
  (PartitionNodeLimit)`; that is the expected behavior on a one-node cluster,
  not a model error.
- P2 old: the placeholder `<your_job_id>` was replaced with a real pending id
  (78) before the run; the error comes from the `--format` option itself.
- P4 new: the prompt asked to run the answer unchanged except for the shell
  name, so `hostname` replaced `bash`.
- Before prompt 2, four train jobs were submitted (`--gres=gpu:2`); two ran
  and two stayed pending, so both squeue and scancel had real data.
