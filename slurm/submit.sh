#!/bin/bash
# Submit the pipeline with dependencies:  generate -> train (seeds) -> evaluate (per seed) -> export
#   bash slurm/submit.sh                 # everything
#   bash slurm/submit.sh train           # start from training (data already generated)
#   bash slurm/submit.sh evaluate        # start from evaluation
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"
mkdir -p slurm/logs   # sbatch --output needs the directory to exist
if [[ "$SBATCH_ACCOUNT" == TODO* ]]; then echo "set SBATCH_ACCOUNT in slurm/env.sh first"; exit 1; fi

START="${1:-generate}"
N=$(( $(wc -w <<< "$SEEDS") - 1 ))
dep=()
submit() { sbatch --parsable --export=ALL "${dep[@]}" "$@"; }

if [ "$START" = generate ]; then
    gen=$(submit slurm/generate.sbatch); echo "generate: $gen"; dep=(--dependency=afterok:$gen)
fi
if [ "$START" = generate ] || [ "$START" = train ]; then
    tr=$(submit --array=0-$N slurm/train.sbatch); echo "train:    $tr (seeds: $SEEDS)"; dep=(--dependency=afterok:$tr)
fi
ev_dep=("${dep[@]}")
[ ${#dep[@]} -gt 0 ] && ev_dep=(--dependency=aftercorr:${tr})   # seed i is evaluated once seed i is trained
ev=$(sbatch --parsable --export=ALL "${ev_dep[@]}" --array=0-$N slurm/evaluate.sbatch); echo "evaluate: $ev"
ex=$(submit slurm/export.sbatch); echo "export:   $ex"
echo "watch: squeue --me    logs: slurm/logs/"
