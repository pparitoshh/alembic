import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]

def fixture(tmp):
    cfg=json.loads((ROOT/'configs/ai_factory_pilot.json').read_text());cfg['run_dir']=str(tmp/'run')
    path=tmp/'config.json';path.write_text(json.dumps(cfg))
    for seed in [42,43,44]:
        for label,n in [('v2',66),('v3',200)]:
            folder=Path(cfg['run_dir'])/f'seed_{seed}/evaluation_{label}';folder.mkdir(parents=True)
            rows=[{'id':f'{label}-{i}','question':'How do I inspect a Slurm job?',
              'reference':'Use squeue.','answer_base':'#SBATCH --invented-impossible-flag',
              'answer_student':'Use squeue.', 'judge':{'student_vs_base':['A','B']}} for i in range(n)]
            (folder/'eval_outputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    return path,Path(cfg['run_dir'])

def invoke(script,args):
    with patch.object(sys,'argv',[script,*map(str,args)]),contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(ROOT/'research/mentor_followup'/script),run_name='__main__')

def test_real_pairs_are_separate_and_resume_without_rejudging(tmp_path):
    cfg,root=fixture(tmp_path)
    def opposite(judge,rows,x,y):
        for row in rows:row['judge']['student_vs_base']=['B','A']
        return 0.,0
    with patch('distillkit.teacher.Teacher',return_value=object()),patch('distillkit.evaluate.win_rate',side_effect=opposite):
        invoke('real_pair_review.py',[cfg,'http://test/v1'])
    report=json.loads((root/'mentor_followup/REAL_PAIR_AGREEMENT.json').read_text())
    assert len(report['comparisons'])==6
    assert all(r['ordered_verdict_agreement']==0 and r['gemma_win_rate']==0 for r in report['comparisons'].values())
    original=json.loads((root/'seed_42/evaluation_v2/eval_outputs.jsonl').read_text().splitlines()[0])
    assert original['judge']['student_vs_base']==['A','B']
    packet=[json.loads(x) for x in (root/'mentor_followup/human_review_packet_30.jsonl').read_text().splitlines()]
    assert len(packet)==30 and all(r['human_verdict'] is None and 'answer_A_model' not in r for r in packet)
    with patch('distillkit.teacher.Teacher',return_value=object()),patch('distillkit.evaluate.win_rate',side_effect=AssertionError('Cache was not reused')):
        invoke('real_pair_review.py',[cfg,'http://changed-port/v1'])

def test_invalid_verdicts_do_not_become_agreement(tmp_path):
    import pytest
    cfg,root=fixture(tmp_path)
    def invalid(judge,rows,x,y):
        for row in rows:row['judge']['student_vs_base']=[None,'B']
        return None,len(rows)
    with patch('distillkit.teacher.Teacher',return_value=object()),patch('distillkit.evaluate.win_rate',side_effect=invalid),pytest.raises(SystemExit):
        invoke('real_pair_review.py',[cfg,'http://test/v1'])
    report=json.loads((root/'mentor_followup/REAL_PAIR_AGREEMENT.json').read_text())
    assert all(r['ordered_verdict_agreement'] is None and r['gemma_win_rate'] is None for r in report['comparisons'].values())

def test_no_scheduler_candidates_are_not_a_perfect_pass_rate(tmp_path):
    cfg,root=fixture(tmp_path)
    with patch('subprocess.run',side_effect=AssertionError('No scheduler invocation expected')):
        invoke('absolute_metrics.py',[cfg])
    report=json.loads((root/'mentor_followup/ABSOLUTE_METRICS.json').read_text())
    assert report['scheduler_tested_scripts']==0 and report['scheduler_pass_rate'] is None
    metrics=report['slices']['42/v3']
    assert metrics['base']['share_answers_without_detected_invalid_flags']==0
    assert metrics['student']['share_answers_without_detected_invalid_flags']==1
    assert metrics['student']['share_flag_containing_answers_without_detected_invalid_flags'] is None
