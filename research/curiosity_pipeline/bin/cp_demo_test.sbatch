#!/bin/bash
#SBATCH --job-name=cp-demo-test
#SBATCH --partition=hackathon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home/eai-jjppv/cp/logs/cp_demo_test-%j.out
# End-to-end check of the cp_demo.sh setup without an interactive shell: Ollama, --setup, a question, an agent
# question, then the web page on port 8765 for 25 minutes (for the ssh tunnel test).
set -x
source $HOME/cp/bin/cp_demo_rc.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
time clusterpilot --setup
time clusterpilot "How do I run 100 tasks as a job array but only 10 at a time?" | head -25
time clusterpilot --agent --yes "Show my jobs in the queue" | head -20
clusterpilot --web --agent --port 8765 &
sleep 1500
