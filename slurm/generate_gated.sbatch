#!/bin/bash
#SBATCH --job-name=dk-gen-gated
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
# Quality-gated teacher data on one GPU (docs/plan/HPC10K_DATA.md). Each gate runs only if the previous
# one succeeded:
#   1. generate  teacher ($TEACHER_MODEL) writes questions + answers from $SEEDS_DIR
#   2. review    an independent reviewer ($REVIEW_MODEL) checks every answer against its source chunk
#   3. verify    deterministic gates (flags, bash, tool replay/grounding, duplicates) + the review verdicts
#   4. decontam  drops records overlapping any held-out question under data/eval/
# Output in $RUN_DIR: generated, grounding_reviews, verified/rejected/pending_review, clean (= train on this).
# Resumable: generate and review skip finished rows, so resubmit after a wall-time kill.
# Fetch the seeds first, on the login node (network):
#   $VENV/bin/python -m distillkit.fetch_sources data/sources/hpc10k_sources.json $RUN_DIR/seeds [--limit N]
#   CONFIG=configs/gen_hpc10k.yaml RUN_DIR=runs/hpc10k sbatch slurm/generate_gated.sbatch
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"

SEEDS_DIR="${SEEDS_DIR:-$RUN_DIR/seeds}"
REVIEW_MODEL="${REVIEW_MODEL:-$CROSS_JUDGE_MODEL}"
[ -f "$SEEDS_DIR/sources_manifest.json" ] || { echo "[job] no $SEEDS_DIR/sources_manifest.json: run distillkit.fetch_sources first"; exit 1; }
SET=(--set run_dir="$RUN_DIR" --set seeds.dir="$SEEDS_DIR" --set verify.grounding_reviews="$RUN_DIR/grounding_reviews.jsonl")
# The few-shot gold examples (data/gold) must find their train-split source docs in seeds.dir
for doc in $("$VENV/bin/python" -c 'import json; p = json.load(open("data/gold/provenance.json"))
print(" ".join(s["doc_id"] for e in p["examples"].values() for s in e["sources"] if s["split"] == "train"))'); do
    cp -n "data/seeds/$doc.md" "$SEEDS_DIR/" || exit 1
done

PORT=$(job_port)
start_vllm "$TEACHER_MODEL" "$PORT" $(teacher_args)
distillkit generate -c "$CONFIG" "${SET[@]}" --set teacher.base_url="http://127.0.0.1:$PORT/v1" \
    --set teacher.model="$TEACHER_MODEL" || exit 1
stop_vllm

PORT=$(job_port 1)
start_vllm "$REVIEW_MODEL" "$PORT" $(judge_args "$REVIEW_MODEL")
distillkit review -c "$CONFIG" "${SET[@]}" --set judge.base_url="http://127.0.0.1:$PORT/v1" \
    --set judge.model="$REVIEW_MODEL" || exit 1
stop_vllm

distillkit verify -c "$CONFIG" "${SET[@]}" || exit 1
distillkit decontam -c "$CONFIG" "${SET[@]}"
