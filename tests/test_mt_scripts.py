"""Hermetic checks for the fix 4 Curiosity shell glue (research/train_multistep_tools/curiosity)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

D = Path(__file__).resolve().parents[1] / 'research' / 'train_multistep_tools' / 'curiosity'
SCRIPTS = sorted([*D.glob('*.sh'), *D.glob('*.sbatch')])
ALLOWED = {'task_bank.py', 'collect_sim.py', 'verify_traj.py', 'build_agent_data.py', 'build_tool_kd.py',
           'train_kd.py'}


def test_scripts_exist():
    names = {p.name for p in SCRIPTS}
    assert {'lib_mt.sh', 'collect.sh', 'build.sh', 'train.sh', 'pipeline.sbatch'} <= names


@pytest.mark.skipif(shutil.which('bash') is None, reason='needs bash')
@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(script):
    r = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.name)
def test_only_known_python_clis(script):
    code = '\n'.join(re.sub(r'\s#\s.*$', '', l) for l in script.read_text().splitlines()
                     if not l.lstrip().startswith('#'))
    used = set(re.findall(r'([\w-]+\.py)\b', code))
    assert used <= ALLOWED, used - ALLOWED


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.name)
def test_no_team_folder_writes(script):
    text = script.read_text()
    assert '$HOME/cp' not in text and '~/cp' not in text
    assert 'bin/lib.sh' not in text
    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            continue
        # no redirect, mkdir, mv, touch, rm or tee into the read-only root
        assert not re.search(r'(>>?|mkdir -p|\bmv\b|\btouch\b|\brm -rf|\btee -a)\s+\$\{?(CP_RO|CP|KDALL|M)\b', line), line
        assert not re.search(r'\b(RUN|OUT|LOGS|PIDFILE)=\$\{?(CP_RO|CP)\b', line), line


def test_lib_roots():
    lib = (D / 'lib_mt.sh').read_text()
    assert 'CP_RO=${CP_RO:-/home/eai-jjppv/cp}' in lib and 'WORK=${WORK:-$HOME/mt}' in lib
    assert 'RUN=$WORK/exp/' in lib and 'OUT=$WORK/exp/' in lib and 'EVAL_RUNS=$WORK/runs' in lib
    assert 'pkill' not in lib


def test_tool_parsers_and_prefix():
    lib = (D / 'lib_mt.sh').read_text()
    assert 'TOOL_PARSER=hermes' in lib and 'TOOL_PARSER=openai' in lib
    assert '--tool-prefix mcp__hpc__' in (D / 'build.sh').read_text()
    assert '--enable-auto-tool-choice' in (D / 'collect.sh').read_text()


def test_build_calls_build_tool_kd():
    text = (D / 'build.sh').read_text()
    # data_src.jsonl is the file build_agent_data.py writes; $STUDENT is the tokenizer, not a positional
    assert '$MT/build_tool_kd.py $RUN/examples/data_src.jsonl $P --tokenizer $STUDENT --tool-prefix mcp__hpc__' in text


def test_build_round2_defaults():
    text = (D / 'build.sh').read_text()
    assert 'UPSAMPLE=${UPSAMPLE:-3}' in text and 'N_REPLAY=${N_REPLAY:-4000}' in text


def test_collect_teacher_retries_round2():
    text = (D / 'collect.sh').read_text()
    assert '--samples $SAMPLES' in text and '${SAMPLES:-1}' in text  # env SAMPLES (default 1) -> collect_sim
    assert '[ "$SAMPLES" -gt 1 ] && VFLAGS="--first-pass-per-task"' in text  # verification keeps the first pass
    assert 'verify_traj.py $RUN/traj.jsonl $RUN/traj_verified.jsonl.tmp $VFLAGS' in text


def test_collect_seed_reproduces_tasks_v3():
    first = json.loads((D.parent / 'tasks_v3.jsonl').read_text(encoding='utf-8').splitlines()[0])
    seed = int(first['id'].split('-')[1])  # the file's own seed decides the collect default
    assert f'task_bank.py $RUN/tasks.jsonl.tmp --seed ${{TASK_SEED:-{seed}}}' in (D / 'collect.sh').read_text()
