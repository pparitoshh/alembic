# Head-to-head check: x0.2 against the base model (20 questions)

Date: 13 October 2026.

## Method

- Student: x0.2, seed 42 (`W = base + 0.2 × (B7 − base)`). This is the best student model at this date
  (v2 0.481, v3 0.528, means of 3 seeds).
- Base model: Qwen3-4B-Instruct-2507, not trained.
- Questions: 10 from v3 and 10 from v2, chosen at random (`random.Random(20261013)`).
- Answers: the answers that the judge scored (`wt-eval-v3/runs/judge_b7s42_t0.2_<v2|v3>/eval_outputs.jsonl`).
  We did not generate new answers. The data is in `research/head_to_head/sample_x0.2_s42.json`.
- Check: one reviewer (Claude) compared each answer with the reference answer and the Slurm documentation.
  - **Correct**: the main command or fact is correct, and the answer contains no incorrect fact that changes the result.
  - **Partly**: the main idea is correct, but the answer has an incorrect option, an incorrect fact or a missing part.
  - **Wrong**: the main command or fact is incorrect.
- Judge: the paired-order judge result (gpt-oss-20b). Score 1 = student wins both orders, 0 = base wins both orders.

Some answers stop before the end because of the token limit (questions 4, 5, 6, 9, 11, 15 and 17). This applies
to both models.

## Summary

| Result | Base model | x0.2 |
|---|---|---|
| Correct | 6 | 5 |
| Partly correct | 9 | 10 |
| Wrong | 5 | 5 |
| Better answer (manual check) | 7 | 3 |
| Same quality (manual check) | 10 | 10 |
| Win rate of x0.2, manual check | – | 0.40 |
| Win rate of x0.2, judge (these 20 questions) | – | 0.43 |
| Manual check and judge agree on the better model (or a tie) | – | 12 of 20 |

## Results for each question

| # | Set | Id | Question (short) | Base model | x0.2 | Better (manual) | Judge score | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | v3 | v3-056 | `sacct`: completed jobs with ID, state, exit code | Partly | Partly | Base | 0 | Base main command correct; invents `--exit-code`. x0.2 first command mixes `-o` with an invalid `--format="%J,%s,%e"`. |
| 2 | v3 | v3-172 | `#SBATCH` for 8 tasks, 2 CPUs, 4 GB, deadline tomorrow 18:00 | Partly | Partly | Same | 0.25 | Same directives. Both use `--deadline=18:00` (today), not `tomorrow18:00`. |
| 3 | v3 | v3-155 | Cancel only my pending jobs | Wrong | Partly | x0.2 | 1 | Base says `scancel --user $USER` cancels only pending jobs (false: it cancels all). x0.2 says the same, then gives the correct `--state PENDING`. |
| 4 | v3 | v3-004 | Dependency + parsable job ID | Partly | Partly | Same | 0.75 | Both give `afterok` correctly. Neither gives `--parsable`. x0.2 invents `afterok:latest`. |
| 5 | v3 | v3-182 | Check `InactiveLimit`, fix the script | Partly | Partly | Base | 0 | Base: correct `scontrol show config`; fix with `srun` (job step) is near the reference; invents `scontrol update ... InactiveLimit`. x0.2: `grep slurm.conf`; fix `sleep 1` without `srun` does not help. |
| 6 | v3 | v3-055 | `ExitCode` vs `DerivedExitCode` | Partly | Partly | x0.2 | 1 | Base invents a list of `DerivedExitCode` values (0–4) and says 143 = SIGKILL (it is SIGTERM). x0.2 is vague but has no invented values. Neither says "highest step exit code". |
| 7 | v3 | v3-133 | 8 tasks × 4 CPUs × 1 GPU on one node | Correct | Correct | Same | 1 | Both correct. Base adds a redundant `--gpus=8`. |
| 8 | v3 | v3-048 | `sacct` COMPLETED and FAILED with exit code and user | Partly | Correct | x0.2 | 1 | Base omits the `State` field, so the user cannot see which job failed. |
| 9 | v3 | v3-068 | `--time-min` shortened the time limit | Partly | Wrong | Base | 0 | Base: the limit can become 1:30 (mechanism not exact). x0.2: says `--time-min` does not shorten the limit (false). Both give the correct fix. |
| 10 | v3 | v3-109 | Mail at start and end, do not overwrite output | Correct | Correct | Same | 0 | Both correct. Base adds `>> job_%j.out` in the script (`%j` is not expanded by the shell). The judge preferred base. |
| 11 | v2 | req-cand-003 | `--time-min` vs `scontrol update TimeLimit` | Wrong | Wrong | Same | 1 | Both describe `--time-min` incorrectly (a guaranteed minimum or a preemption guard). It lets Slurm shorten the limit for backfill. |
| 12 | v2 | arr-006 | `afterok` vs `afterany` | Partly | Partly | Same | 0.25 | Both correct on success vs any state. Both say `afterany` starts after any one of the jobs (false: it waits for all). |
| 13 | v2 | req-cand-011 | `--no-requeue` and missing `--open-mode=append` | Correct | Correct | Same | 0.25 | Both find the 2 conflicts. |
| 14 | v2 | req-010 | State TIMEOUT | Correct | Partly | Base | 0 | x0.2 gives incorrect causes (resource exhaustion, missing `sleep`) and does not say "increase `--time`". |
| 15 | v2 | arr-cand-009 | `afterany` report, then diagnose only if the report fails | Wrong | Wrong | Base | 0.5 | Both invent `afterany:6501:exitcode!=0`. Neither uses `--parsable` or `afternotok`. x0.2's script also points the diagnosis at job 6500, which is the error the user wants to prevent. |
| 16 | v2 | req-cand-018 | Requeue job 4711 and query `sacct` | Correct | Correct | Same | 0.5 | Almost the same answer. |
| 17 | v2 | req-cand-014 | Old log text and same job ID after requeue | Correct | Partly | Base | 0 | Base: correct (same ID, append keeps text). x0.2: contradicts itself in the heading and suggests a check that does not test the requeue. |
| 18 | v2 | req-cand-009 | Time-limit kill vs memory kill | Partly | Partly | Same | 0.5 | Both use `scontrol show job` (`sacct` with TIMEOUT / OUT_OF_MEMORY is correct). Both give the correct condition for the time-limit change. |
| 19 | v2 | arr-cand-018 | Resubmit array tasks 1 and 7, one at a time | Wrong | Wrong | Base | 0 | Neither cancels the old tasks or uses `%1`. Base invents `--array-start`. x0.2 also submits an extra array of 8 tasks. |
| 20 | v2 | arr-cand-019 | prepare → analyze → recover chain | Wrong | Wrong | Same | 0.5 | Same answer: dependencies on script names, and an invented `afterfail`. Correct: `--parsable`, `afterok`, `afternotok`. |

## Findings

1. x0.2 and the base model make the same type of errors. 10 of 20 answers have the same quality.
2. Neither model knows `--parsable` or `afternotok` (questions 4, 15 and 20). On v2 this is expected: these topics are not in the training data.
3. x0.2 wins when the base model adds incorrect details (questions 3, 6 and 8).
4. x0.2 loses when it gives an incorrect explanation of a mechanism (questions 9, 14 and 17).
5. The judge and the manual check agree on 12 of 20 questions. Most differences are questions that the manual check rates as the same quality. On these 20 questions, both give x0.2 less than 0.5.
   The 3-seed v3 result (0.528) is close to 0.5. A sample of 20 cannot confirm or reject a difference this small.
