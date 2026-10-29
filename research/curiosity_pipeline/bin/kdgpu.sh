#!/bin/bash
# GPU/MPI round: the best student at run time distilled from Qwen3-235B on 60,000 fresh prompts weighted to GPU, MPI and
# the software environment (the weakest test set, d1). env: NAME (default kdgpu), PROMPT_SEED (default 11),
# PROMPT_MIX (default 0.2,0.65,0.15: Slurm, GPU/MPI/env, other)
B=$(python3 -c "import json; print(json.load(open('$HOME/cp/autopilot/state.json'))['best']['model'])")
PROMPT_MIX=${PROMPT_MIX:-0.2,0.65,0.15} PROMPT_SEED=${PROMPT_SEED:-11} exec bash $HOME/cp/bin/exp_kd.sh ${NAME:-kdgpu} 60000 $B
