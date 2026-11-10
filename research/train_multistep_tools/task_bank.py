"""Training fix 4, step 2 (research/train_multistep_tools/PLAN.md): a task bank for teacher collection.

Writes a JSONL file of randomized user requests for the 19 tools of distillkit.slurm_live, on Curiosity and
on Leonardo. One JSON object per line:

    {"id", "cluster": "curiosity"|"leonardo", "family", "user_request", "expected_tools": [ordered tool names],
     "needs_tool": bool, "multi_step": bool,
     "fixture": null|"running"|"pending"|"completed"|"failed_oom"|"failed_time"|"failed_error"}

Rules this bank follows (PLAN.md step 2 and Rules):
- Job IDs are never invented. A request that speaks about one job carries the literal placeholder {JOB_ID} and a
  "fixture" kind that the collector (step 3, based on research/agent_train/collect.py) must create on the cluster
  first and substitute for the placeholder:
      running        a job that is RUNNING (a sleeper)
      pending        a job held in PENDING (start time in the future)
      completed      a short job that finished successfully
      failed_oom     a job killed by its memory limit (OUT_OF_MEMORY)
      failed_time    a job stopped at its time limit
      failed_error   a job that died on a command error (non-zero exit, e.g. a Python traceback)
  A request without {JOB_ID} has fixture null and needs no job.
- Cluster facts in requests come only from: Curiosity (partition hackathon, 8x B300 per node, QOS eai-team04,
  enroot+pyxis, no debug QOS) and Leonardo (partitions boost_usr_prod, dcgp_usr_prod, lrd_all_serial; 4x A100
  64 GB per boost node). dcgp_usr_prod and lrd_all_serial take no GPU requests on Leonardo: GPU tasks
  (gpu_availability, GPU counts) name boost_usr_prod only. Limits are never hard-coded in a request; the model
  must get them from tools.
- Action tools (submit_job, cancel_job, update_job, hold_job, release_job, requeue_job) appear only when the
  request asks for the action in plain words.
- Round 2 (PLAN_round2.md): two three-call families (three expected tools in the listed order), and three
  "check first" families: a general request (is_implicit, no tool words) whose correct answer needs one read
  tool first, because the advice depends on what the tool reports.
- Multi-step requests come in two registers: at least 70% state only the goal (no step words such as
  first/then/afterwards, no tool-like words such as check/list/pull; see is_implicit), at most 30% spell the
  steps out. The mix is fixed per family by a quota, and the implicit pool is self-checked with is_implicit.
- Every task is checked against the mentor use cases (data/mentor_usecases/, held-out test data) for word
  5-gram overlap; a task whose maximum overlap over the mentor requests is above 0.3 is dropped, and identical
  requests are deduplicated (a duplicate retry draws another wording instead).

usage: .venv/bin/python3 research/train_multistep_tools/task_bank.py OUT.jsonl --seed 4 [--n-per-family N]

--n-per-family N: rows per multi-step family; the single-tool, check-first, no_tool and ask_job_id families
get max(1, N//2) rows each, so multi-step tasks stay the majority (they must be at least 50%). The default
N=81 gives ~1800 rows.
"""
import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from distillkit.slurm_live import openai_tools

TOOL_NAMES = [t["function"]["name"] for t in openai_tools()]
ACTION_TOOLS = {"submit_job", "cancel_job", "update_job", "hold_job", "release_job", "requeue_job"}
READ_ONLY_TOOLS = [t for t in TOOL_NAMES if t not in ACTION_TOOLS]
FIXTURE_KINDS = (None, "running", "pending", "completed", "failed_oom", "failed_time", "failed_error")
FIX_ANY = ["running", "pending", "completed", "failed_oom", "failed_time", "failed_error"]
FIX_DONE = ["completed", "failed_oom", "failed_time", "failed_error"]
FIX_FAIL = ["failed_oom", "failed_time", "failed_error"]
ROW_KEYS = {"id", "cluster", "family", "user_request", "expected_tools", "needs_tool", "multi_step", "fixture"}

# Cluster facts that templates may use, and nothing else. "gpu_short" is the plural-friendly GPU name.
# dcgp_usr_prod and lrd_all_serial are CPU partitions on Leonardo: gpu_partitions drives every GPU request.
CLUSTERS = {
    "curiosity": {"partitions": ["hackathon"], "gpu_partitions": ["hackathon"], "gpu": "B300",
                  "gpu_short": "B300", "gpus_per_node": "8x B300", "qos": "eai-team04"},
    "leonardo": {"partitions": ["boost_usr_prod", "dcgp_usr_prod", "lrd_all_serial"],
                 "gpu_partitions": ["boost_usr_prod"], "gpu": "A100 64GB",
                 "gpu_short": "A100", "gpus_per_node": "4x A100 64GB", "qos": None},
}

WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper",
         "kepler", "lumen", "marble", "nimbus", "onyx", "prism", "quartz", "raven", "sierra", "tundra"]
MODELS = ["Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "Llama-3.1-8B", "Llama-3.3-70B", "Mistral-Small-24B"]
GPU_COUNTS = [1, 2, 4]
HOURS = [2, 4, 6, 8, 12, 24]
TAILS = [20, 50, 100]
DAYS = [3, 7, 14]
WHEN = ["last night", "yesterday", "this morning", "two days ago", "over the weekend", "late on Friday"]
UNTIL = ["this evening", "tomorrow", "the weekend", "Friday", "Monday morning"]


def jname(rng):
    return f"{rng.choice(WORDS)}-{rng.randint(10, 99)}"


def pick(rng, templates, **slots):
    """One random wording, with slot values substituted. {JOB_ID} is never a slot: it stays literal."""
    req = rng.choice(templates)
    for key, value in slots.items():
        req = req.replace("{" + key + "}", str(value))
    return req


# A multi-step request states only the goal when it names no step ("first", "then", "afterwards", ...) and no
# tool-like word ("check", "list", "pull", ...): it says what the user wants, not how to find it.
IMPLICIT_SHARE = 0.7
STEP_WORDS = re.compile(r"\b(first|then|after that|afterwards|next|once you|also|two things)\b", re.I)
TOOL_WORDS = re.compile(
    r"\b(check|look( up)?|pull|fetch|list|show|read|query|confirm|verify|inspect|examine|diagnose|monitor"
    r"|grep|get me|give me)\b", re.I)


def is_implicit(request):
    """True when the request states only the goal: no step words, no tool-like words."""
    return not STEP_WORDS.search(request) and not TOOL_WORDS.search(request)


# ---- multi-step families: the user needs two tool calls, in the listed order. The implicit pool states only
# the goal (is_implicit must hold), the explicit pool spells the steps out; build_tasks mixes them 70/30.
def f_gpu_then_partition(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "Check how many {gpu}s are idle on {part} at the moment, and also get me that partition's limits "
            "before I write my job script.",
            "First tell me whether GPUs are free on {part}, then pull the partition's time and resource "
            "limits for my {n}-GPU run.",
            "Two things about {part} for my {m} evaluation as a {n}-GPU job: are GPUs free right now, and "
            "what does the partition allow?",
        ], part=rng.choice(cl["gpu_partitions"]), gpu=cl["gpu_short"], n=rng.choice(GPU_COUNTS),
            m=rng.choice(MODELS)), None
    return pick(rng, [
        "Is there room for a {n}-GPU {h}-hour run on {part} today?",
        "Would a {n}-GPU {h}-hour {m} job fit on {part} right now?",
        "Can {part} take a {n}-GPU {m} run this week?",
        "Do {n} free {gpu}s on {part} sound realistic for a {h}-hour job tomorrow?",
    ], part=rng.choice(cl["gpu_partitions"]), gpu=cl["gpu_short"], n=rng.choice(GPU_COUNTS),
        h=rng.choice(HOURS), m=rng.choice(MODELS)), None


def f_status_then_log(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "Can you check the state of job {JOB_ID} and then show me the last {t} lines of its output log?",
            "First tell me the current state of my {jn} job {JOB_ID}, then read its log and summarize where it stands.",
            "Please check my {m} job {JOB_ID}'s state, then pull the last {t} lines of its output for me.",
        ], t=rng.choice(TAILS), jn=jname(rng)), rng.choice(["failed_error", "completed", "running"])
    return pick(rng, [
        "Did job {JOB_ID} run cleanly? I never saw any output from it.",
        "What did my {jn} job {JOB_ID} produce in the end?",
        "Is something wrong with job {JOB_ID}? Its results have not appeared.",
        "Did my {m} job {JOB_ID} get to the end without trouble?",
    ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(["failed_error", "completed", "running"])


def f_why_failed(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "Job {JOB_ID} did not succeed. Look at its final state and its accounting record, then tell me "
            "why it died.",
            "My {jn} job {JOB_ID} is gone from the queue. Check its final state and pull its accounting "
            "record so we know what killed it.",
            "Check what state job {JOB_ID} ended in, then use its accounting record to explain the failure "
            "to me.",
        ], jn=jname(rng)), rng.choice(FIX_FAIL)
    return pick(rng, [
        "Why did job {JOB_ID} fail?",
        "My {m} job {JOB_ID} died {when} — what happened there?",
        "What was the reason my {jn} job {JOB_ID} never made it to the end?",
        "How come my {m} run {JOB_ID} is no longer among my jobs?",
    ], jn=jname(rng), m=rng.choice(MODELS), when=rng.choice(WHEN)), rng.choice(FIX_FAIL)


def f_why_pending(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "Job {JOB_ID} has been waiting for {h} hours. Confirm that it is still pending, then use its "
            "priority data to explain why it is not starting.",
            "My {jn} job {JOB_ID} will not start. First check its state, then look at its priority and tell "
            "me what makes it wait.",
            "Check job {JOB_ID}: is it still pending? If yes, get its priority breakdown and explain the wait.",
        ], h=rng.choice([1, 2, 3, 5, 8]), jn=jname(rng)), "pending"
    return pick(rng, [
        "Why does job {JOB_ID} still wait?",
        "Why has my {jn} job {JOB_ID} not started for {h} hours?",
        "My {m} run {JOB_ID} has been in the queue since {when} — what is the holdup?",
    ], h=rng.choice([1, 2, 3, 5, 8]), jn=jname(rng), m=rng.choice(MODELS), when=rng.choice(WHEN)), "pending"


def f_queue_then_status(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "I have a few jobs in the queue right now, my {jn} job {JOB_ID} among them. Show me my whole queue and "
            "then check the exact state of {JOB_ID}.",
            "List my queued jobs, then give me the detailed state of {JOB_ID} (my {m} run) in particular.",
            "What do I have in the queue at the moment? Afterwards, look up the exact state of job {JOB_ID} "
            "(the {jn} run) for me.",
        ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(["running", "pending"])
    return pick(rng, [
        "Where does my {jn} job {JOB_ID} stand in the queue right now?",
        "Will my queued {m} job {JOB_ID} start soon?",
        "Is my {jn} job {JOB_ID} still far from its turn in the queue?",
        "How is the position of my {m} job {JOB_ID} in the queue looking?",
    ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(["running", "pending"])


def f_fix_failed_job(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "My {jn} job {JOB_ID} failed. Get its batch script and read its log, find the mistake and tell me "
            "exactly what to change.",
            "Fetch the script of my failed {m} job {JOB_ID} and look at its log as well, then work out why it broke.",
            "Please pull up both the script and the output log of job {JOB_ID} (the {jn} run) and tell me "
            "what in the script went wrong.",
        ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(FIX_FAIL)
    return pick(rng, [
        "What in my {jn} job {JOB_ID} is broken?",
        "My {m} job {JOB_ID} has failed twice in a row now — what is wrong with it?",
        "Why does the script of job {JOB_ID} keep dying?",
        "What is wrong with job {JOB_ID}? It should have been done hours ago.",
    ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(FIX_FAIL)


def f_recent_then_accounting(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "In the last {d} days I ran job {JOB_ID} and a few others. List my recent jobs, then give me the "
            "accounting record (state, elapsed time, memory) for {JOB_ID}.",
            "Look up my jobs from the past {d} days, then pull the accounting details for my {m} job "
            "{JOB_ID}: I want its elapsed time and memory use.",
            "Show my recent jobs (about {d} days back) and then the accounting record of the {jn} run "
            "{JOB_ID} in particular.",
        ], d=rng.choice(DAYS), jn=jname(rng)), rng.choice(["completed", "failed_error", "failed_time"])
    return pick(rng, [
        "How expensive was my {m} run {JOB_ID} from the last {d} days, in time and memory?",
        "What are the time and memory numbers of my {jn} job {JOB_ID} from the past {d} days?",
        "Did my {d}-day-old {m} job {JOB_ID} leave a reasonable footprint?",
    ], d=rng.choice(DAYS), jn=jname(rng), m=rng.choice(MODELS)), rng.choice(
        ["completed", "failed_error", "failed_time"])


def f_partition_then_reservation(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "Get me the details and limits of partition {part}, then check whether any reservation will "
            "take nodes away from it soon.",
            "First pull the properties of partition {part}, then list the current reservations on this "
            "cluster for me.",
            "I need two things for my {h}-hour plan on {part}: the partition's limits, and any reservations "
            "that could block nodes.",
        ], part=rng.choice(cl["partitions"]), h=rng.choice(HOURS)), None
    return pick(rng, [
        "Can I count on {part} staying usable for a {h}-hour job this week?",
        "Is {part} a safe bet for my {h}-hour run tomorrow?",
        "Will maintenance or reservations get in the way of a {h}-hour job on {part}?",
        "Does {part} have what my {h}-hour job needs, and will reservations stay out of the way this week?",
    ], part=rng.choice(cl["partitions"]), h=rng.choice(HOURS)), None


def f_usage_then_log(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "Check how much CPU and memory my {jn} job {JOB_ID} actually used, then show me the last {t} lines "
            "of its log so I can see where the time went.",
            "Pull the resource usage of my {m} job {JOB_ID} and then the tail of its output ({t} lines).",
            "Get the usage numbers of job {JOB_ID} (the {jn} run), then read the last {t} lines of its log.",
        ], t=rng.choice(TAILS), jn=jname(rng)), rng.choice(["completed", "failed_error"])
    return pick(rng, [
        "Did my {jn} job {JOB_ID} use its resources sensibly?",
        "Was job {JOB_ID} oversized for what it actually did?",
        "Is my {m} job {JOB_ID} wasting CPU or memory, judging by its output?",
    ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(["completed", "failed_error"])


def f_status_then_hold(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "Please check the state of my {jn} job {JOB_ID}; if it has not started yet, put it on hold for me.",
            "First confirm that job {JOB_ID} is still pending, then hold it for me until {until}.",
            "Check my {m} job {JOB_ID}'s state and, if it is still waiting, hold the job for me now.",
        ], until=rng.choice(UNTIL), jn=jname(rng)), "pending"
    return pick(rng, [
        "Keep job {JOB_ID} from starting until {until}, please.",
        "Job {JOB_ID} should not start until {until} — please make that happen.",
        "I need my {jn} job {JOB_ID} to stay unstarted until {until}.",
    ], until=rng.choice(UNTIL), jn=jname(rng)), "pending"


def f_status_then_update(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "Check the state of my {jn} job {JOB_ID} for me. It will need longer than planned: please raise its "
            "time limit to {h} hours.",
            "Look at my {m} job {JOB_ID}; if it is running or waiting, change its wall time limit to {h} hours for me.",
            "First tell me the state of job {JOB_ID} (the {jn} run), then update its time limit to {h} hours.",
        ], h=rng.choice(HOURS), jn=jname(rng)), rng.choice(["pending", "running"])
    return pick(rng, [
        "My {m} job {JOB_ID} will not fit into its current time budget — it needs {h} hours.",
        "Job {JOB_ID} deserves {h} hours instead of what it has now.",
        "Raise job {JOB_ID} to {h} hours so it can finish.",
        "My {jn} job {JOB_ID} needs {h} hours rather than its current limit.",
    ], h=rng.choice(HOURS), jn=jname(rng), m=rng.choice(MODELS)), rng.choice(["pending", "running"])


# ---- three-call families: three read calls, in the listed order (round 2: longer chains) ----------------------------
def f_status_then_log_then_accounting(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "Check job {JOB_ID}: tell me how it ended, then read its last {t} log lines, then pull its "
            "accounting record and explain the failure with all three.",
            "First the end state of job {JOB_ID}, then the tail of its log ({t} lines), then its accounting "
            "record — I want state, error and resource numbers in one explanation.",
            "Look at the state, the log tail ({t} lines) and the accounting data of my {jn} run {JOB_ID} and "
            "tell me what went wrong and what it cost.",
        ], t=rng.choice(TAILS), jn=jname(rng)), rng.choice(FIX_FAIL)
    return pick(rng, [
        "Job {JOB_ID} failed {when}. What went wrong, and how much wall time and memory did it waste?",
        "Why did my {m} job {JOB_ID} die, and what should I change in the script for the retry?",
        "My {jn} job {JOB_ID} failed: explain the error, and judge from the accounting whether the run was "
        "sized sensibly at all.",
        "What killed job {JOB_ID}, and was the failed run badly oversized in CPU or memory?",
    ], jn=jname(rng), m=rng.choice(MODELS), when=rng.choice(WHEN)), rng.choice(FIX_FAIL)


def f_gpu_then_partition_then_reservation(rng, cl, explicit):
    if explicit:
        return pick(rng, [
            "Check the free {gpu}s on {part}, then pull the partition's limits and list the reservations "
            "that could take nodes away from it, for my {n}-GPU {h}-hour {m} plan.",
            "First show me how many {gpu}s are idle on {part}, then its time and resource limits, then the "
            "upcoming reservations — I plan an {h}-hour {n}-GPU {m} run.",
            "Three things about {part} before I submit my {n}-GPU {m} job: idle {gpu}s now, the partition's "
            "limits, and reservations this week.",
        ], part=rng.choice(cl["gpu_partitions"]), gpu=cl["gpu_short"], n=rng.choice(GPU_COUNTS),
            m=rng.choice(MODELS)), None
    return pick(rng, [
        "Can I still fit a {n}-GPU {h}-hour {m} run on {part} this week, or should I plan around maintenance "
        "and limits?",
        "Is {part} workable for my {n}-GPU {m} run tomorrow and for the rest of the week as well?",
        "What does a {n}-GPU {h}-hour {m} run on {part} face today, tomorrow and this week?",
        "Do the GPUs, the settings and the plans of {part} allow my {n}-GPU {m} job before {until}?",
    ], part=rng.choice(cl["gpu_partitions"]), gpu=cl["gpu_short"], m=rng.choice(MODELS),
        n=rng.choice(GPU_COUNTS), h=rng.choice(HOURS), until=rng.choice(UNTIL)), None


# ---- "check first" families: a general request (is_implicit: no tool words) whose correct answer needs one 
# read tool first, because the advice depends on what the tool reports (round 2) --------------------------------------
def f_check_first_partition_info(rng, cl):
    return pick(rng, [
        "What wall time should I give my {m} training job on {part}?",
        "How should I set --mem for a {m} run on {part} so it is allowed but not oversized?",
        "How many CPUs will one task of my {m} job get on {part}, so I can pick --cpus-per-task?",
        "Is a {h}-hour wall time realistic for a {m} run on {part}, or does the partition stop it?",
    ], part=rng.choice(cl["partitions"]), m=rng.choice(MODELS), h=rng.choice(HOURS)), None


def f_check_first_gpu_availability(rng, cl):
    return pick(rng, [
        "Should I send my {n}-GPU {m} job to {part} this evening, or is all of it busy for now?",
        "Is {part} a good target for {n} {gpu}s right now, or should I wait until tomorrow?",
        "Can my {n}-GPU {m} run really start on {part} tonight?",
        "Is there room on {part} for a quick {n}-GPU test of {m} right now?",
    ], part=rng.choice(cl["gpu_partitions"]), gpu=cl["gpu_short"], m=rng.choice(MODELS),
        n=rng.choice(GPU_COUNTS)), None


def f_check_first_job_status(rng, cl):
    return pick(rng, [
        "Should I wait for my {jn} job {JOB_ID}, or can I book my {m} evaluation in the meantime?",
        "Can I already start the postprocessing, or is my {jn} run {JOB_ID} still far from its results?",
        "My {m} job {JOB_ID} has no results yet: do I still have a reason to keep waiting?",
        "Is my {jn} job {JOB_ID} close enough to the end that I can plan my {m} follow-up for today?",
    ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(["running", "pending", "completed"])


# ---- single-tool families: one per read-only tool ------------------------------------------------------------------
def f_single_job_status(rng, cl):
    return pick(rng, [
        "What state is job {JOB_ID} in right now?",
        "Is job {JOB_ID} still running?",
        "Check the current state of job {JOB_ID} for me.",
        "Tell me the exact state of job {JOB_ID}, please.",
        "My {jn} job {JOB_ID}: what state is it in?",
        "Did my {m} job {JOB_ID} get to run, and where does it stand now?",
        "Look up the state of job {JOB_ID} (the {jn} run) for me.",
    ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(FIX_ANY)


def f_single_list_queue(rng, cl):
    return pick(rng, [
        "What do I have in the queue right now? Job {JOB_ID} should be in there.",
        "Show me my jobs that are queued or running at the moment; I expect {JOB_ID} among them.",
        "List my current queue for me. I am waiting for {JOB_ID} in particular.",
        "My {jn} job {JOB_ID} should be queued: show me my queue.",
        "What is on the queue from my side right now? I care most about {JOB_ID}.",
        "Show me my queued jobs, please; my {m} run {JOB_ID} should be one of them.",
    ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(["running", "pending"])


def f_single_read_job_log(rng, cl):
    return pick(rng, [
        "Show me the last {t} lines of the output log of job {JOB_ID}.",
        "What did job {JOB_ID} print at the end of its run? Show me the tail of its log.",
        "I need the last {t} lines of job {JOB_ID}'s log, please.",
        "Show me the tail of the error stream of job {JOB_ID} ({t} lines).",
        "My {jn} job {JOB_ID} just ended: show me the last {t} lines of its log.",
        "Read me the end of the log of job {JOB_ID}, last {t} lines.",
    ], t=rng.choice(TAILS), jn=jname(rng)), rng.choice(FIX_DONE)


def f_single_job_accounting(rng, cl):
    return pick(rng, [
        "How long did job {JOB_ID} run, and how much memory did it use?",
        "Pull the accounting record for job {JOB_ID}: final state, elapsed time, CPU and memory.",
        "What were the resource numbers (time, CPU, memory) of job {JOB_ID}?",
        "My {jn} job {JOB_ID}: what do the accounting data say about its time and memory?",
        "Get me the accounting numbers of job {JOB_ID}, my {m} run from earlier.",
        "From the accounting record of job {JOB_ID}: did it end cleanly, and how much did it consume?",
    ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(FIX_DONE)


def f_single_gpu_availability(rng, cl):
    return pick(rng, [
        "Are there free {gpu}s on {part} right now?",
        "How many {gpu}s are idle on {part} at the moment?",
        "I want to launch a {n}-GPU job soon; is anything free on {part} right now?",
        "Check GPU availability on {part} for me, please.",
        "How do the {gpu}s on {part} look right now, free or busy?",
        "For a short {n}-GPU test of {m}: are there idle GPUs on {part}?",
    ], part=rng.choice(cl["gpu_partitions"]), gpu=cl["gpu_short"], n=rng.choice(GPU_COUNTS),
        m=rng.choice(MODELS)), None


def f_single_partition_info(rng, cl):
    return pick(rng, [
        "What are the limits of partition {part} (max wall time, resources per job)?",
        "Tell me the properties of partition {part}: what nodes does it have and what are its limits?",
        "Before I write my script: what does partition {part} allow, and what defaults does it set?",
        "I plan to run {m} there: what are the limits and defaults of partition {part}?",
        "For a longer job, what does partition {part} allow (time, resources, defaults)?",
        "What kind of nodes and limits does partition {part} have?",
    ], part=rng.choice(cl["partitions"]), m=rng.choice(MODELS)), None


def f_single_recent_jobs(rng, cl):
    return pick(rng, [
        "Which jobs have I submitted in the last {d} days? Job {JOB_ID} should be among them.",
        "List my jobs from the past {d} days; I want to see {JOB_ID} in the list.",
        "Show my recent submissions, about {d} days back, please. I am looking for {JOB_ID}.",
        "My {m} run {JOB_ID} was a few days ago: list my recent jobs so I can see it.",
        "List the jobs I submitted in the last {d} days, including the {jn} one ({JOB_ID}).",
    ], d=rng.choice(DAYS), jn=jname(rng), m=rng.choice(MODELS)), "completed"


def f_single_job_script(rng, cl):
    return pick(rng, [
        "Show me the batch script of job {JOB_ID}.",
        "I lost my local copy; can you fetch the script of job {JOB_ID} for me?",
        "Get the script that job {JOB_ID} was started with, please.",
        "My {jn} job {JOB_ID}: show me the exact batch script it runs with.",
        "Fetch the batch script behind job {JOB_ID} so I can compare it with my {m} template.",
    ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(["completed", "failed_error"])


def f_single_job_usage(rng, cl):
    return pick(rng, [
        "How much memory and CPU did job {JOB_ID} actually use?",
        "What was the resource usage of job {JOB_ID} while it ran?",
        "Show me the usage numbers of job {JOB_ID}; I want to know if it can run leaner.",
        "My {jn} job {JOB_ID}: what did it really use in CPU and memory?",
        "Get the usage data of job {JOB_ID}, my {m} run, for me.",
    ], jn=jname(rng), m=rng.choice(MODELS)), rng.choice(["completed", "failed_error"])


def f_single_job_priority(rng, cl):
    return pick(rng, [
        "What is the current priority of my pending job {JOB_ID}?",
        "Show me the priority breakdown of job {JOB_ID}, please.",
        "Get me the priority data for job {JOB_ID}.",
        "My {jn} job {JOB_ID} has waited {h} hours: show me its priority breakdown.",
        "What does the priority of job {JOB_ID} look like right now?",
    ], jn=jname(rng), h=rng.choice([1, 2, 3, 5, 8])), "pending"


READONLY_CMDS = ["sinfo -o %P %a %l %D", "squeue -o %.10i %.9P %.20j %.2t %.10M", "sinfo -t idle",
                 "sacctmgr show qos", "scontrol show nodes", "scontrol show partition {part}", "sinfo -Ne",
                 "squeue -u $USER", "scontrol show reserv", "sinfo -O partition,cpus,memory,features"]


def f_single_account_info(rng, cl):
    return pick(rng, [
        "Which QOS on this cluster can my account use?",
        "Tell me my account and QOS situation here, with the limits of each QOS.",
        "Show me my accounts, the QOS I may use in them and my fairshare on this cluster.",
        "What does my QOS allow (GPUs, CPUs, wall time), and what is my fairshare in it?",
        "My {m} job has waited since {when}: how strong is my fairshare in the QOS of my account?",
    ], m=rng.choice(MODELS), when=rng.choice(WHEN)), None


def f_single_run_readonly_command(rng, cl):
    return pick(rng, [
        "Run `{cmd}` for me and show me the raw output.",
        "What does `{cmd}` return? Just run it in read-only mode and paste the output.",
        "Give me the output of the read-only command `{cmd}`.",
        "Run `{cmd}` (read-only) and show me exactly what it prints.",
        "Could you execute `{cmd}` for me? Read-only, and show the raw output.",
    ], cmd=rng.choice(READONLY_CMDS).replace("{part}", rng.choice(cl["partitions"]))), None


def f_single_reservation_info(rng, cl):
    return pick(rng, [
        "Are there any reservations on the cluster at the moment that could block {part}?",
        "List the current reservations and who owns them, please.",
        "Show me upcoming reservations; I want to know when nodes might be reserved away from {part}.",
        "Before I plan my {h}-hour run: are there reservations that touch {part}?",
        "Is a reservation going to take nodes away from {part} this week?",
        "Do any reservations overlap with {part} right now?",
        "List the active and future reservations on this cluster, with their owners.",
    ], part=rng.choice(cl["partitions"]), h=rng.choice(HOURS), n=rng.choice(GPU_COUNTS)), None


# ---- no-tool and ask-for-the-job-id families -----------------------------------------------------------------------
def f_no_tool(rng, cl):
    return pick(rng, [
        "How do I request a {n}-GPU allocation for {h} hours in a Slurm batch script? Which #SBATCH lines do I need?",
        "What is the difference between --ntasks and --ntasks-per-node in sbatch? I always mix them up.",
        "How do I write a job array in sbatch that runs at most 2 tasks at the same time?",
        "How do I chain two Slurm jobs so that the second starts only after the first finished successfully?",
        "What is the difference between srun and sbatch, and when do I use which?",
        "How do I make my conda environment visible inside a Slurm batch job?",
        "How do I compile a CUDA extension with the right -arch flag for a modern GPU?",
        "In an MPI job, how do I bind one MPI rank per GPU in the batch script?",
        "How do I profile a short CUDA kernel from inside a Slurm job without an interactive session?",
        "What does --dependency=afterok:JOBID do in sbatch, exactly?",
        "How do I estimate the memory that a {m} fine-tune in bf16 needs, so I can pick a --mem value?",
        "How do I pass an environment variable from a Slurm batch script into an MPI rank?",
        "How do I check the CUDA compute capability that my compiled binary was built for?",
        "What is the cleanest way to limit OpenMP threads per Slurm task?",
        "For a {h}-hour {m} evaluation as a {n}-GPU job, which sbatch lines do I need at minimum?",
        "How do I make a Slurm job wait for a file to appear before it starts the real command?",
    ], n=rng.choice(GPU_COUNTS), h=rng.choice(HOURS), m=rng.choice(MODELS)), None


def f_ask_job_id_no_tool(rng, cl):
    return pick(rng, [
        "My training job from {when} failed; can you tell me why?",
        "Something is off with my job from {when}. Can you look into it?",
        "One of my jobs is stuck since {when}. Can you check what is going on?",
        "My long run from {when} should be done by now, but I have a feeling it is not. Can you find out?",
        "I think one of my jobs died {when}. Can you tell me which one and why?",
        "My job from {when} never wrote its results. Can you help me understand what happened?",
        "My {m} run from {when}: I am not sure it finished. Can you look into that?",
        "A job of mine from {when} behaved strangely; can you help me understand it?",
    ], when=rng.choice(WHEN), m=rng.choice(MODELS)), None


def f_ask_job_id_recent_jobs(rng, cl):
    return pick(rng, [
        "Can you check how my training run from {when} ended? Look through my recent jobs to find it.",
        "Did my long job from {when} make it to the end? Check my recent jobs and tell me.",
        "I submitted something {when} but forgot to write down the id. Can you find it in my recent jobs "
        "and tell me its state?",
        "Please look at my recent jobs, find the big run from {when}, and tell me how it ended.",
        "My {m} run from {when}: find it in my recent jobs and tell me how it ended.",
        "Look through my recent jobs for the {m} job from {when} and tell me its final state.",
    ], when=rng.choice(WHEN), m=rng.choice(MODELS)), None


MULTI_STEP_FAMILIES = {
    "gpu_then_partition": (["gpu_availability", "partition_info"], f_gpu_then_partition),
    "status_then_log": (["job_status", "read_job_log"], f_status_then_log),
    "why_failed": (["job_status", "job_accounting"], f_why_failed),
    "why_pending": (["job_status", "job_priority"], f_why_pending),
    "queue_then_status": (["list_queue", "job_status"], f_queue_then_status),
    "fix_failed_job": (["job_script", "read_job_log"], f_fix_failed_job),
    "recent_then_accounting": (["recent_jobs", "job_accounting"], f_recent_then_accounting),
    "partition_then_reservation": (["partition_info", "reservation_info"], f_partition_then_reservation),
    "usage_then_log": (["job_usage", "read_job_log"], f_usage_then_log),
    "status_then_hold": (["job_status", "hold_job"], f_status_then_hold),
    "status_then_update": (["job_status", "update_job"], f_status_then_update),
    "status_then_log_then_accounting": (["job_status", "read_job_log", "job_accounting"],
                                        f_status_then_log_then_accounting),
    "gpu_then_partition_then_reservation": (["gpu_availability", "partition_info", "reservation_info"],
                                            f_gpu_then_partition_then_reservation),
}
CHECK_FIRST_FAMILIES = {
    f"check_first_{t}": ([t], globals()[f"f_check_first_{t}"])
    for t in ("partition_info", "gpu_availability", "job_status")
}
SINGLE_TOOL_FAMILIES = {f"single_{t}": ([t], globals()[f"f_single_{t}"]) for t in READ_ONLY_TOOLS}
NO_TOOL_FAMILIES = {
    "no_tool": ([], f_no_tool),
    "ask_job_id_no_tool": ([], f_ask_job_id_no_tool),
    "ask_job_id_recent_jobs": (["recent_jobs"], f_ask_job_id_recent_jobs),
}
FAMILIES = {**MULTI_STEP_FAMILIES, **CHECK_FIRST_FAMILIES, **SINGLE_TOOL_FAMILIES, **NO_TOOL_FAMILIES}

for _name, (_expected, _gen) in FAMILIES.items():  # validate against the real tool list at import time
    assert all(t in TOOL_NAMES for t in _expected), _name
    if _name.startswith("single_"):
        assert _expected == [_name[len("single_"):]] and _name[len("single_"):] in READ_ONLY_TOOLS, _name
    if _name.startswith("check_first_"):
        assert _expected == [_name[len("check_first_"):]] and _name[len("check_first_"):] in READ_ONLY_TOOLS, _name


# ---- overlap check against the held-out mentor use cases -----------------------------------------------------------
def word_ngrams(text, n=5):
    words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def max_overlap(request, mentor_requests, n=5):
    """|task n-grams and mentor n-grams| / |task n-grams|, maximum over the mentor requests."""
    task_grams = word_ngrams(request, n)
    if not task_grams:
        return 0.0
    best = 0.0
    for mentor in mentor_requests:
        grams = word_ngrams(mentor, n)
        if grams:
            best = max(best, len(task_grams & grams) / len(task_grams))
    return best


def load_mentor_requests(mentor_dir=Path("data/mentor_usecases")):
    """The held-out mentor user requests, for the overlap check only (never as templates)."""
    requests = []
    for path in sorted(Path(mentor_dir).glob("*/usecases.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                requests.append(json.loads(line)["user_request"])
    return requests


# ---- build and validate --------------------------------------------------------------------------------------------
def validate_row(row):
    assert set(row) == ROW_KEYS, row["id"]
    assert re.fullmatch(r"tb-\d+-\d{5}", row["id"]), row["id"]
    assert row["cluster"] in CLUSTERS
    assert row["family"] in FAMILIES
    assert isinstance(row["expected_tools"], list)
    assert all(t in TOOL_NAMES for t in row["expected_tools"]), row["id"]
    assert row["needs_tool"] is bool(row["expected_tools"])
    assert row["multi_step"] is (len(row["expected_tools"]) >= 2)
    assert row["fixture"] in FIXTURE_KINDS
    assert ("{JOB_ID}" in row["user_request"]) == (row["fixture"] is not None), row["id"]
    left = set(re.findall(r"\{([A-Z_]+)\}", row["user_request"]))
    assert left <= {"JOB_ID"}, row["id"]


def build_tasks(seed, n_per_family, mentor_requests):
    """The task bank: rows per family (multi-step families get n_per_family, the rest max(1, n_per_family//2)),
    mixed order, checked for mentor overlap and duplicates. Deterministic for a given seed. Multi-step families
    place IMPLICIT_SHARE of their rows as goal-only requests (is_implicit) and the rest as step-naming ones."""
    rng = random.Random(seed)
    tasks, seen = [], set()
    dropped_overlap = duplicate_retries = 0
    for family, (expected, gen) in FAMILIES.items():
        multi = len(expected) >= 2
        target = n_per_family if multi else max(1, n_per_family // 2)
        impl_left = math.ceil(IMPLICIT_SHARE * target) if multi else 0
        placed, attempts = 0, 0
        while placed < target and attempts < 60 * target:
            attempts += 1
            cluster = rng.choice(sorted(CLUSTERS))
            explicit = multi and impl_left <= 0  # implicit rows first, then the explicit quota
            request, fixture = (gen(rng, CLUSTERS[cluster], explicit) if multi
                                else gen(rng, CLUSTERS[cluster]))
            if multi and not explicit:
                assert is_implicit(request), request  # the implicit pool must stay free of step/tool words
            if request in seen:  # identical user_request: draw another wording instead
                duplicate_retries += 1
                continue
            if max_overlap(request, mentor_requests) > 0.3:  # too close to a held-out mentor request: drop
                dropped_overlap += 1
                continue
            seen.add(request)
            if multi:
                impl_left -= 1  # goes negative for explicit rows, never below the explicit tail
            tasks.append({"id": "", "cluster": cluster, "family": family, "user_request": request,
                          "expected_tools": list(expected), "needs_tool": bool(expected),
                          "multi_step": multi, "fixture": fixture})
            placed += 1
    rng.shuffle(tasks)  # mix families and clusters for training order
    for i, task in enumerate(tasks):
        task["id"] = f"tb-{seed}-{i:05d}"
        validate_row(task)
    return tasks, {"dropped_overlap": dropped_overlap, "duplicate_retries": duplicate_retries}


def print_summary(tasks, stats):
    per_family = Counter(t["family"] for t in tasks)
    print("=== task bank summary ===")
    print(f"{'family':<30} {'rows':>5}  expected chain")
    for family, (expected, _) in FAMILIES.items():
        print(f"{family:<30} {per_family[family]:>5}  {' -> '.join(expected) or '(no tool)'}")
    print(f"{'total':<30} {len(tasks):>5}")
    print(f"\n{'cluster':<12} {'rows':>5} {'multi_step':>12} {'no_tool':>9}")
    for cluster in sorted(CLUSTERS):
        rows = [t for t in tasks if t["cluster"] == cluster]
        multi = sum(t["multi_step"] for t in rows) / len(rows) if rows else 0.0
        notool = sum(not t["needs_tool"] for t in rows) / len(rows) if rows else 0.0
        print(f"{cluster:<12} {len(rows):>5} {multi:>11.1%} {notool:>9.1%}")
    multi = sum(t["multi_step"] for t in tasks) / len(tasks) if tasks else 0.0
    notool = sum(not t["needs_tool"] for t in tasks) / len(tasks) if tasks else 0.0
    multi_rows = [t for t in tasks if t["multi_step"]]
    implicit = sum(is_implicit(t["user_request"]) for t in multi_rows) / len(multi_rows) if multi_rows else 0.0
    print(f"\nmulti_step share: {multi:.1%}   no_tool share: {notool:.1%}"
          f"   implicit multi-step share: {implicit:.1%}")
    print(f"dropped: overlap={stats['dropped_overlap']}  duplicate_retries={stats['duplicate_retries']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="output JSONL path")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--n-per-family", type=int, default=81,
                        help="rows per multi-step family; other families get max(1, N//2) (default: 81, ~1500 rows)")
    parser.add_argument("--mentor-dir", type=Path, default=Path("data/mentor_usecases"),
                        help="held-out mentor use cases for the overlap check (never used as templates)")
    args = parser.parse_args(argv)
    mentor_requests = load_mentor_requests(args.mentor_dir)
    tasks, stats = build_tasks(args.seed, args.n_per_family, mentor_requests)
    with args.out.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    print_summary(tasks, stats)
    print(f"\nwrote {len(tasks)} rows to {args.out}")


if __name__ == "__main__":
    main()
