"""Dataset regression checks, real production scorer, isolated mocks, no inference."""
from pathlib import Path
import importlib.util,json,shutil
import pytest
import yaml
from distillkit.config import Config, apply_overrides
from distillkit.evaluate import _eval_rows
from distillkit.toolcheck import score_tool_item
from distillkit.tools import _SESSION

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'data/eval/dataset_v1'
spec=importlib.util.spec_from_file_location('dataset_release',HERE/'validate_release.py')
release=importlib.util.module_from_spec(spec);spec.loader.exec_module(release)
ITEMS={r['id']:r for r in release.rows(HERE/'development_tools.jsonl')}
FIXTURES=json.loads((HERE/'scorer_fixtures.json').read_text())['cases']

def test_complete_release_bindings_and_original_slices():
    result=release.validate()
    assert result['total_ready']==151 and result['inherited_82_unchanged']
    assert result['new_training_records']==0 and result['calibration_pairs_separate']==24

@pytest.mark.parametrize('slice_name,expected',[('development',(106,23)),('final_normal_only',(22,0))])
def test_documented_overrides_select_actual_production_slice(monkeypatch,slice_name,expected):
    monkeypatch.chdir(ROOT)
    overrides=json.loads((HERE/'config_overrides.json').read_text())[slice_name]
    raw=yaml.safe_load((ROOT/'configs/qwen3_4b_qdora.yaml').read_text())
    cfg=Config.model_validate(apply_overrides(raw,[f'{key}={json.dumps(value)}' for key,value in overrides.items()]))
    normal,tools=_eval_rows(cfg)
    assert (len(normal),len(tools))==expected

@pytest.mark.parametrize('case',FIXTURES,ids=[f"{x['id']}:{i}" for i,x in enumerate(FIXTURES)])
def test_new_tool_expectations_with_production_first_response_scorer(case):
    assert _SESSION.get() is None
    result=score_tool_item(case['synthetic_response'],ITEMS[case['id']])
    assert result['first_response_pass'] is case['expected_first_response_pass']
    assert result['response_quality_assessed'] is False and result['task_completion_assessed'] is False
    assert _SESSION.get() is None

@pytest.mark.parametrize('fault',['duplicate','empty_reference','extra_field','changed_evidence'])
def test_release_rejects_corrupted_records_or_evidence(tmp_path,fault):
    here=tmp_path/'release';shutil.copytree(HERE,here)
    path=here/'development_normal.jsonl';rows=release.rows(path)
    if fault=='duplicate':rows[-1]['id']=rows[-2]['id']
    elif fault=='empty_reference':rows[-1]['reference']=' '
    elif fault=='extra_field':rows[-1]['training_eligible']=True
    else:
        m=json.loads((here/'item_metadata.json').read_text())
        e=next(x for r in m['items'] for x in r['evidence'] if 'quote' in x)
        e['quote']+=' invented support'
        (here/'item_metadata.json').write_text(json.dumps(m))
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(AssertionError):release.validate(here,ROOT,check_manifest=False)
