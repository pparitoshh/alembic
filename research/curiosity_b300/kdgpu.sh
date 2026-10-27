#!/bin/bash
# kdgpu: the best student at run time distilled from Qwen3-235B on 60,000 fresh prompts weighted to GPU, MPI and the
# software environment (the weakest test set, d1): 20% Slurm, 65% GPU/MPI/env, 15% other.
B=$(python3 -c "import json; print(json.load(open('$HOME/cp/autopilot/state.json'))['best']['model'])")
PROMPT_MIX=0.2,0.65,0.15 PROMPT_SEED=11 exec bash $HOME/cp/bin/exp_kd.sh kdgpu 60000 $B
