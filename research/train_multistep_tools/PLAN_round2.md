# Fix 4, round 2: more tool calls in each chain

Round 1 (PR #117, model `gparitosh007/hpc-assistant-x0.2-gguf`) fixed the tool-call format. It did not reach the goals for the number of tool calls. Round 2 adds more and longer tool chains to the training data. It also keeps the HPC answers at the 1.3 level.

## Round 1 result

| Goal | Round 1 | Status |
|---|---|---|
| Cases with tool calls as text: 2 or less (16 Curiosity mentor cases) | 0 | passed |
| Cases with real tool calls: 12 or more | 9 (direct), 8 (proxy) | not passed |
| Mean recall of expected tools: 0.5 or more | 0.292 (direct), 0.109 (proxy) | not passed |
| HPC answers (leaderboard, 3-seed mean) at the 1.3 level | 0.850 against 0.867 (v2: −0.037) | small loss |

Facts from round 1:

- 7 of 16 Curiosity mentor cases have no tool call.
- The verifier kept 810 of 1491 trajectories (54.3%). 275 fails are "coverage": the teacher did not call the second expected tool.
- Weak families: `no_tool` 17.5%, `single_job_usage` 17.5%, `recent_then_accounting` 19.8%, `status_then_hold` 25.9%, `usage_then_log` 28.4%.
- Tool rows were 1620 of 7234 training sequences (810 × 2). The other rows were replay rows.
- The direct and proxy mentor runs differ much (recall 0.292 against 0.109). One run per mode at temperature 0.7 is not enough to compare models.

## Step 0: measure (laptop, about 1 hour)

1. Run `mentor_ui.py` 3 times for each mode (direct, proxy) for round 1, with temperature 0 in the Modelfile. Report the mean and the range.
2. Run the same for 1.3, so that both models have the same baseline.
3. For each case without a tool call, record only the failure type (for example: answers from memory, asks the user, stops after a plan). Do not copy the mentor cases into the task bank or use them as templates. They stay held-out.

Output: `research/train_multistep_tools/ROUND2_baseline.md` with the means and the failure types.

## Step 1: data changes

| Change | File | Why |
|---|---|---|
| 1. Teacher retries: up to 4 samples for each task (temperature 0.7), keep the first sample that passes the verifier | `collect_sim.py` (`--samples`), `collect.sh` | More verified multi-step chains; fewer coverage fails |
| 2. New "check first" families: a general request where the correct answer needs a read tool first (`partition_info`, `gpu_availability`, `job_status`) before the advice | `task_bank.py`, `tasks_v3.jsonl` | Matches the failure type "answers from memory" |
| 3. Longer chains: families with 3 tool calls (for example status → log → accounting) | `task_bank.py` | Round 1 had at most 2 expected calls |
| 4. Upsample the multi-step rows 3× (round 1: 2×) and use 4000 replay rows (round 1: 3000) | `build.sh` (`UPSAMPLE`, `N_REPLAY`) | More tool signal; keeps the HPC answers |

Each code change gets tests. The mentor 5-gram overlap check stays on for the new task bank.

## Step 2: training on Curiosity

- `NAME=v3`, `--onehot` targets, from 1.3, LR 1e-5, alpha 0.8, 3 seeds (42, 43, 44) as a chain of own 4-GPU jobs.
- Optional second run `NAME=v3t` with `MODE=teacher` (teacher-scored KD), so that the effect of soft targets shows alone.
- Time: collection about 15 min (4 samples), build about 5 min, train about 15 min and eval about 20 min for each seed. About 2–3 hours for 3 seeds.

## Step 3: evaluation

1. Leaderboard (`eval.sh`), 3-seed mean against 1.3 and round 1.
2. Weight average of the 3 seeds, GGUF Q4_K_M.
3. Mentor UI eval, 3 runs per mode, temperature 0, against the step 0 baselines.
4. Leonardo 001–016 without tools (knowledge check; must not drop).

## Release rule

Release round 2 to the HF repo and the guide only if:

- recall (direct, mean of 3 runs) is higher than round 1, and
- cases with tool calls as text stay at 2 or less, and
- the leaderboard mean is not lower than round 1.

Report 3-seed means only.
