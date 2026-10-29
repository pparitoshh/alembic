import pytest
from distillkit.checks import check_answer, load_flags
from distillkit.slurm_time import check_time_comments, duration_minutes


@pytest.mark.parametrize('value,minutes', [
    ('5', 5), ('5:00', 5), ('00:05', 1), ('00:05:00', 5), ('00:01:01', 2),
    ('2-3', 3060), ('2-3:04', 3064), ('2-3:04:05', 3065), ('0-00:05', 5),
    ('0', None), ('00:00', None), ('0-0:0:0', None),
])
def test_documented_formats_and_minute_rounding(value, minutes):
    assert duration_minutes(value) == minutes


@pytest.mark.parametrize('value', ['tomorrow', '$TIME', '-1', '1:2:3:4', '1--2'])
def test_nonliteral_times_not_interpreted(value):
    with pytest.raises(ValueError):
        duration_minutes(value)


@pytest.mark.parametrize('directive', [
    '#SBATCH --time=00:05 # 5 minutes time limit',
    '#SBATCH --time=00:04 # timeout in HH:MM',
    '#SBATCH --time=00:05 # 5 minutes',
    '#SBATCH --time "00:05" # time limit: 5 minutes',
    "#SBATCH -t'00:05' # 5 minutes",
    '#SBATCH --time=0 # 5 minutes',
    '#SBATCH --time=00:02:00 # 1 hour and 30 minutes',
])
def test_explicit_contradictions_rejected(directive):
    result = check_answer(f'```bash\n{directive}\n```', load_flags(), check_option_values=True)
    assert result['bad_flags'] == [] and not result['passed']
    assert result['bad_option_values'][0]['option'] == 'time'
    assert result['option_value_check_protocol'] == 'slurm-known-invalid-values-v2'


@pytest.mark.parametrize('directive', [
    '#SBATCH --time=5 # 5 minutes', '#SBATCH -t 5:00 # 5 minutes',
    '#SBATCH --time=00:05:00 # 5 minutes', '#SBATCH --time=1-0:05 # 1 day 5 minutes',
    '#SBATCH --time=01:30:00 # 1 hour and 30 minutes',
    '#SBATCH --time=00:01 # 1 minute', '#SBATCH --time=0 # 0 minutes',
    '#SBATCH --time=00:01:01 # 1.5 minutes',
    '#SBATCH --time=23:59:00 # 1 minute less than the max time',
    '#SBATCH --time=02:00:00 # 10 minutes buffer',
    '#SBATCH --time=00:04 # use 5 minutes for a larger workload',
    '#SBATCH --time=00:04 # NOT HH:MM',
    '#SBATCH --time=00:04:00 # HH:MM:SS',
    '#SBATCH --begin=00:05 # 5 minutes', '#SBATCH --time-min=00:05 # 5 minutes',
    '#SBATCH --time=$TIME # 5 minutes',
    '#SBATCH --wrap="echo --time=00:05 # 5 minutes"',
    'echo "#SBATCH --time=00:05 # 5 minutes"',
    '#SBATCH --comment="--time=00:05" # 5 minutes',
    '#SBATCH --time=00:05 --nodes=1 # 5 minutes',
    'srun application --time=00:05 # 5 minutes',
])
def test_valid_relative_and_unhandled_comments_are_preserved(directive):
    assert check_time_comments(directive) == []


def test_legacy_benchmark_checks_remain_unchanged():
    result = check_answer('```bash\n#SBATCH --time=00:05 # 5 minutes\n```', load_flags())
    assert result['passed'] and 'bad_option_values' not in result


def test_verification_rejects_duration_contradiction(tmp_path):
    from distillkit import verify
    from distillkit.config import Config
    from distillkit.io import write_jsonl, read_jsonl
    from test_pipeline import _cfg_dict
    cfg = Config.model_validate(_cfg_dict(tmp_path))
    cfg.run_dir.mkdir()
    write_jsonl(cfg.run_dir / 'generated.jsonl', [dict(id='wrong-duration',
        question='Request a five-minute limit for this preprocessing task.',
        answer='```bash\n#SBATCH --time=00:05 # 5 minutes\n```')])
    verify.run(cfg)
    assert read_jsonl(cfg.run_dir / 'verified.jsonl') == []
    assert read_jsonl(cfg.run_dir / 'rejected.jsonl')[0]['reject_reason'] == 'bad_option_value'
