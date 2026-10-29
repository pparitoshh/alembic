# Shell setup inside the cp_demo.sh GPU job: start Ollama on this node, define `clusterpilot`, print how to open
# the web page from the laptop.
[ -f /etc/bashrc ] && source /etc/bashrc
source /etc/profile.d/modules.sh 2>/dev/null; module load slurm >/dev/null 2>&1
export PATH=$HOME/cp/ollama/bin:$PATH OLLAMA_MODELS=$HOME/cp/ollama_models OLLAMA_HOST=127.0.0.1:11434
mkdir -p $OLLAMA_MODELS $HOME/cp/logs
if ! ollama list >/dev/null 2>&1; then
    nohup ollama serve > $HOME/cp/logs/ollama_demo.log 2>&1 &
    for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done
fi
clusterpilot() { PYTHONPATH=$HOME/cp/demo_app/src $HOME/cp/venv/bin/python -c "import sys; sys.argv[0] = \"clusterpilot\"; from distillkit.assistant import main; sys.exit(main())" "$@"; }
export -f clusterpilot
PS1="(clusterpilot on $(hostname -s)) \w\$ "
RELAY=$(( 20000 + ${SLURM_JOB_ID:-0} % 10000 ))  # port on the login node, one per job: no clash with other users
cat <<MSG

  ClusterPilot on Curiosity, node $(hostname -s), 1 B300 GPU, Ollama $(ollama --version 2>/dev/null | grep -o "[0-9.]*$")
    clusterpilot --setup                          (first time: model 4.3 GB + docs index)
    clusterpilot "How do I run 100 tasks as a job array but only 10 at a time?"
    clusterpilot --agent "Show my jobs in the queue"          (live Slurm on Curiosity)
    clusterpilot --web --agent --port 8765        (web page; then on the LAPTOP run:)
        ssh -N -L 8765:127.0.0.1:$RELAY curiosity "ssh -N -L $RELAY:127.0.0.1:8765 $(hostname -s)"
      then open http://127.0.0.1:8765 in the laptop browser (Ctrl+C in that laptop window closes the tunnel)
  Type exit to give the GPU back.

MSG
