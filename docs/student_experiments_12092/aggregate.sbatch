#!/bin/bash
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --chdir=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/student-experiments-v1
#SBATCH --output=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/student-experiments-v1/logs/%x-%j.out
#SBATCH --export=NONE
set -euo pipefail
BASE=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z
exec "$BASE/runtime/train-cu129/bin/python" -B "$BASE/student-experiments-v1/aggregate.py"
