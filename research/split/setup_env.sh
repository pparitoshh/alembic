#!/bin/bash
# Self-contained Python runtime for the second account: copies of the uv CPython and both virtual environments in the
# shared folder, rewired to point only inside it (the originals point into a08trc09's private home).
set -euo pipefail
S=/leonardo_work/EUHPC_D30_031/shared_split
PY=$S/python/cpython-3.12.15-linux-x86_64-gnu
mkdir -p $S/python
[ -x $PY/bin/python3.12 ] || cp -a ~/.local/share/uv/python/cpython-3.12.15-linux-x86_64-gnu $S/python/
fix() {  # $1 = source venv, $2 = destination
    local src=$1 dst=$2
    [ -d $dst ] || cp -a $src $dst
    sed -i "s#^home = .*#home = $PY/bin#" $dst/pyvenv.cfg
    for l in python python3 python3.12; do [ -L $dst/bin/$l ] && ln -sfn $PY/bin/python3.12 $dst/bin/$l; done
    # console scripts carry the old venv path in their shebang
    grep -rlI "^#!$src/bin/python" $dst/bin | xargs -r sed -i "1s|^#!$src/bin/python|#!$dst/bin/python|"
}
fix /leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129 $S/venv-train
fix /leonardo_work/EUHPC_D30_031/venvs/vllm $S/venv-vllm
sed -i -e "s#^export VENV=.*#export VENV=$S/venv-train#" -e "s#^export VLLM_VENV=.*#export VLLM_VENV=$S/venv-vllm#" $S/code/research/rag65/env.sh
chgrp -R EUHPC_D30_031 $S/python $S/venv-train $S/venv-vllm
chmod -R g+rX $S/python $S/venv-train $S/venv-vllm
$S/venv-train/bin/python -c "import sys, torch, transformers, numpy; print('train venv ok', sys.version.split()[0], torch.__version__)"
$S/venv-vllm/bin/python -c "import vllm; print('vllm venv ok', vllm.__version__)"
head -1 $S/venv-vllm/bin/vllm
grep -n 'VENV=' $S/code/research/rag65/env.sh
echo SETUP_DONE
