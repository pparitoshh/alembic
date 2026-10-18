#!/bin/bash
#SBATCH --job-name=dk-ai-factory-report
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
cd "${REPO:?Set isolated REPO}"
/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python research/ai_factory_pilot/collect_results.py runs/ai_factory_pilot_20261013
