# Agent live coverage run — results

## Run state

- One run on 2026-10-27 on Leonardo. The jobs ran in reservation `s_tra_ncc1` on partition `boost_usr_prod`.
- Model: `hpc-assistant-vkkm-live` (VaibhavKKM qwen3-4b weights with the x02 Ollama template).
- Temperature: 0. The runner sets this value in every chat request.
- Tasks: 37. Scored run file: `data/agent_live/runs/final.jsonl`.
- Mean time per task: 18.8 s.

## Task table

`f` = first call has the expected tool. `a` = the expected tool appears at least once. `r` = the arguments of the expected tool call are correct. `z` = no tool call at all.

| id | f | a | r | z | expected tool | calls | made call names | seconds |
|---|:-:|:-:|:-:|:-:|---|---:|---|---:|
| js-01 | x | x | x |  | job_status | 1 | job_status | 17.0 |
| js-02 | x | x | x |  | job_status | 1 | job_status | 8.5 |
| lq-01 | x | x | x |  | list_queue | 1 | list_queue | 12.7 |
| lq-02 | x | x | x |  | list_queue | 1 | list_queue | 9.2 |
| rjl-01 | x | x | x |  | read_job_log | 1 | read_job_log | 21.2 |
| rjl-02 | x | x | x |  | read_job_log | 1 | read_job_log | 19.9 |
| ja-01 | x | x | x |  | job_accounting | 1 | job_accounting | 15.0 |
| ja-02 | x | x | x |  | job_accounting | 1 | job_accounting | 5.4 |
| ga-01 | x | x | x |  | gpu_availability | 1 | gpu_availability | 5.8 |
| ga-02 | x | x | x |  | gpu_availability | 1 | gpu_availability | 11.8 |
| pi-01 | x | x | x |  | partition_info | 1 | partition_info | 11.8 |
| pi-02 | x | x | x |  | partition_info | 1 | partition_info | 35.7 |
| sj-01 | x | x | x |  | submit_job | 1 | submit_job | 16.8 |
| sj-02 | x | x | x |  | submit_job | 1 | submit_job | 17.4 |
| sj-03 | x | x | x |  | submit_job | 1 | submit_job | 40.3 |
| cj-01 | x | x | x |  | cancel_job | 1 | cancel_job | 4.7 |
| cj-02 | x | x | x |  | cancel_job | 1 | cancel_job | 4.9 |
| jsz-01 | x | x | x |  | job_script | 1 | job_script | 9.2 |
| jsz-02 | x | x | x |  | job_script | 1 | job_script | 26.3 |
| ju-01 | x | x | x |  | job_usage | 4 | job_usage, run_readonly_command, run_readonly_command, job_script | 59.8 |
| ju-02 | x | x | x |  | job_usage | 1 | job_usage | 10.4 |
| jp-01 | x | x | x |  | job_priority | 1 | job_priority | 20.4 |
| jp-02 |  | x | x |  | job_priority | 2 | job_status, job_priority | 65.2 |
| uj-01 |  |  |  |  | update_job | 1 | job_status | 16.3 |
| uj-02 |  |  |  |  | update_job | 1 | job_status | 13.6 |
| hj-01 | x | x | x |  | hold_job | 1 | hold_job | 7.0 |
| hj-02 | x | x | x |  | hold_job | 1 | hold_job | 4.9 |
| rj-01 | x | x | x |  | release_job | 1 | release_job | 4.5 |
| rj-02 |  | x | x |  | release_job | 2 | job_status, release_job | 15.0 |
| rq-01 |  |  |  | x | requeue_job | 0 | none | 3.7 |
| rq-02 |  |  |  | x | requeue_job | 0 | none | 7.1 |
| rr-01 |  |  |  | x | run_readonly_command | 0 | none | 20.1 |
| rr-02 | x | x | x |  | run_readonly_command | 1 | run_readonly_command | 34.8 |
| rr-03 | x | x | x |  | run_readonly_command | 1 | run_readonly_command | 12.4 |
| rr-04 |  |  |  | x | run_readonly_command | 0 | none | 31.3 |
| sr-01 |  |  |  | x | search_docs | 0 | none | 11.7 |
| sr-02 |  |  |  |  | search_docs | 1 | run_readonly_command | 64.8 |
| total | 27/37 | 29/37 | 29/37 | 5/37 | all tools | 37 | mean 18.8 s per task | 696.6 total |

## Per-tool table

Percentages are shares of the tasks in the row.

| expected tool | tasks | first-tool % | any-tool % | args % |
|---|---:|---:|---:|---:|
| job_status | 2 | 100% | 100% | 100% |
| list_queue | 2 | 100% | 100% | 100% |
| read_job_log | 2 | 100% | 100% | 100% |
| job_accounting | 2 | 100% | 100% | 100% |
| gpu_availability | 2 | 100% | 100% | 100% |
| partition_info | 2 | 100% | 100% | 100% |
| submit_job | 3 | 100% | 100% | 100% |
| cancel_job | 2 | 100% | 100% | 100% |
| job_script | 2 | 100% | 100% | 100% |
| job_usage | 2 | 100% | 100% | 100% |
| job_priority | 2 | 50% | 100% | 100% |
| update_job | 2 | 0% | 0% | 0% |
| hold_job | 2 | 100% | 100% | 100% |
| release_job | 2 | 50% | 100% | 100% |
| requeue_job | 2 | 0% | 0% | 0% |
| run_readonly_command | 4 | 50% | 50% | 50% |
| search_docs | 2 | 0% | 0% | 0% |
| total | 37 | 73% | 78% | 78% |

## Zero-call tasks

- `rq-01`: the model made no tool call. The expected tool was `requeue_job`.
- `rq-02`: the model made no tool call. The expected tool was `requeue_job`.
- `rr-01`: the model made no tool call. The expected tool was `run_readonly_command`.
- `rr-04`: the model made no tool call. The expected tool was `run_readonly_command`.
- `sr-01`: the model made no tool call. The expected tool was `search_docs`.

## Misses

A miss has no first-tool match, or the arguments of the expected tool call do not match.

| id | expected tool | actual calls | problem |
|---|---|---|---|
| jp-02 | job_priority | job_status, job_priority | first call was `job_status` |
| uj-01 | update_job | job_status | first call was `job_status`; the model never called `update_job` |
| uj-02 | update_job | job_status | first call was `job_status`; the model never called `update_job` |
| rj-02 | release_job | job_status, release_job | first call was `job_status` |
| rq-01 | requeue_job | none | the model made no tool call |
| rq-02 | requeue_job | none | the model made no tool call |
| rr-01 | run_readonly_command | none | the model made no tool call |
| rr-04 | run_readonly_command | none | the model made no tool call |
| sr-01 | search_docs | none | the model made no tool call |
| sr-02 | search_docs | run_readonly_command | first call was `run_readonly_command`; the model never called `search_docs` |

## Safety

- The model made 6 live action calls (cancel 2, hold 2, release 2). The runner passes `confirm=False` to SlurmLive, so every live action call was declined. Nothing changed on the cluster.
- All 3 submit_job calls had `test_only: true` (3 of 3 confirmed by the scorer). They ran only `sbatch --test-only` and created no job.
