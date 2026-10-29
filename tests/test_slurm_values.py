import json
from pathlib import Path
import pytest
from distillkit.checks import check_answer, load_flags
from distillkit.slurm_values import check_distribution_values, distribution_issue


@pytest.mark.parametrize('value', ['fcyclic', 'fcyclic,NoPack', 'block:Pack', 'block:NoPack',
                                  'block:block:Pack', 'block,fcyclic', 'block:arbitrary'])
def test_known_invalid_values_rejected(value):
    answer = f'```bash\n#SBATCH --distribution={value}\n```'
    result = check_answer(answer, load_flags(), check_option_values=True)
    assert not result['passed'] and result['bad_flags'] == []
    assert result['bad_option_values'][0]['value'] == value


def test_against_captured_native_parser():
    fixture = Path(__file__).parent / 'fixtures/slurm_distribution_native.json'
    probe = json.loads(fixture.read_text())
    caught = 0
    for row in probe['results']:
        issue = distribution_issue(row['value'])
        if issue:
            assert row['returncode'] != 0 and 'Invalid --distribution' in row['stderr']
            caught += 1
        if row['returncode'] == 0:
            assert issue is None, row
    assert caught == 7  # narrow guard, not a complete parser


@pytest.mark.parametrize('command', [
    'srun --distribution=fcyclic ./app', 'salloc --distribution fcyclic ./app',
    'sbatch -mfcyclic job.sh', '/usr/bin/srun --exclusive -n 4 -m fcyclic ./app',
    'srun --ntasks=4 --distribution="block:Pack" ./app',
    'srun \\\n  --distribution=fcyclic ./app',
])
def test_direct_commands(command):
    assert check_distribution_values(f'```bash\n{command}\n```')


@pytest.mark.parametrize('command', [
    'srun python train.py --distribution=fcyclic',
    'srun --exclusive python train.py --distribution=fcyclic',
    'srun -- python train.py --distribution=fcyclic',
    'echo sbatch --distribution=fcyclic',
    'srun bash -c "echo --distribution=fcyclic"',
    'srun --site-specific-switch python --distribution=fcyclic',
    '#SBATCH --wrap="echo --distribution=fcyclic"',
    '#SBATCH --distribution=$DIST', '#SBATCH --distribution=<method>',
    '#SBATCH --distribution=block,Pack # not --distribution=fcyclic',
])
def test_application_arguments_templates_and_comments_are_not_slurm_values(command):
    assert check_distribution_values(f'```bash\n{command}\n```') == []


def test_legacy_evaluation_contract_is_unchanged():
    answer = '```bash\n#SBATCH --distribution=fcyclic\n```'
    old = check_answer(answer, load_flags())
    assert old['passed'] and 'option_value_check_protocol' not in old
    new = check_answer(answer, load_flags(), check_option_values=True)
    assert not new['passed'] and new['option_value_check_protocol']


def test_verify_cannot_ignore_failed_value_check(tmp_path):
    from distillkit import verify
    from distillkit.config import Config
    from distillkit.io import write_jsonl, read_jsonl
    from test_pipeline import _cfg_dict
    cfg = Config.model_validate(_cfg_dict(tmp_path))
    cfg.run_dir.mkdir()
    write_jsonl(cfg.run_dir / 'generated.jsonl', [dict(id='invalid-distribution',
        question='How should these simulation tasks be placed?',
        answer='```bash\n#SBATCH --distribution=block:Pack\n```')])
    verify.run(cfg)
    assert read_jsonl(cfg.run_dir / 'verified.jsonl') == []
    rejected = read_jsonl(cfg.run_dir / 'rejected.jsonl')
    assert rejected[0]['reject_reason'] == 'bad_option_value'
