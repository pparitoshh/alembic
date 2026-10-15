import copy
import json
from types import SimpleNamespace

import pytest

from distillkit.training_quality import command_target_findings, require_training_target_quality
from distillkit.train import admitted_training_rows, split_by_document, validation_groups


def row(answer, question='Show standard output from task 7.'):
    return {'id': 'synthetic-training-only', 'doc_id': 'synthetic-source',
            'question': question, 'answer': answer,
            'grounding_checks': {'status': 'supported'}}


def test_stream_filter_matches_requested_stream():
    bad = '```bash\nsattach --error-filter=7 200.0\n```'
    assert command_target_findings('Show stdout.', bad) == ['stdout_requirement_missing_output_filter']
    assert not command_target_findings('Show stderr.', bad)
    assert not command_target_findings('Show stdout.', bad.replace('--error-filter', '--output-filter'))
    assert not command_target_findings('Show stdout and stderr.',
                                      '```sh\nsattach --output-filter=7 --error-filter=7 200.0\n```')


@pytest.mark.parametrize('value', ['=plane,3', ' plane,3'])
def test_invalid_distribution_value(value):
    assert command_target_findings('Launch tasks.', f'```bash\nsrun --distribution{value} ./a\n```') == [
        'invalid_plane_distribution_separator']


@pytest.mark.parametrize('value', ['=plane=3', ' plane=3', '=block,Pack', '=cyclic'])
def test_valid_distribution_is_not_rejected(value):
    assert not command_target_findings('Launch tasks.', f'```bash\nsrun --distribution{value} ./a\n```')


def test_comment_and_plain_prose_are_not_commands():
    assert not command_target_findings('Explain.', 'Do not use --distribution=plane,3.')
    assert not command_target_findings('Explain.', '```bash\n# srun --distribution=plane,3 is invalid\ntrue\n```')


def test_multiline_command_and_transcript_target():
    r = row('unused')
    r['messages'] = [{'role': 'user', 'content': 'Launch tasks.'},
                     {'role': 'assistant', 'content': '```bash\nsrun \\\n  --distribution=plane,3 ./a\n```'}]
    with pytest.raises(ValueError, match='command/intent'):
        require_training_target_quality([r])


def test_historical_support_cannot_bypass_gate_and_evidence_does_not_modify_rows(tmp_path):
    rows = [row('```bash\nsattach --error-filter=7 200.0\n```')]
    before = copy.deepcopy(rows)
    with pytest.raises(ValueError, match='command/intent'):
        require_training_target_quality(rows, tmp_path)
    assert rows == before
    report = json.loads((tmp_path / 'training_target_quality.json').read_text())
    assert not report['passed'] and not report['answers_modified']


def test_training_entrypoint_rejects_before_gpu_import(tmp_path):
    (tmp_path / 'verified.jsonl').write_text(json.dumps(row('```bash\nsattach --error-filter=7 200.0\n```')) + '\n')
    cfg = SimpleNamespace(train=SimpleNamespace(input_manifest=None),
                          seeds=SimpleNamespace(registry=None), run_dir=tmp_path)
    with pytest.raises(ValueError, match='command/intent'):
        admitted_training_rows(cfg, evidence_dir=tmp_path)


def test_passing_bounded_checks_is_not_semantic_certification():
    report = require_training_target_quality([row('```bash\nsattach --output-filter=7 200.0\n```')])
    assert report['passed'] and not report['semantic_certification']


def test_connected_source_families_never_cross_validation():
    rows = [{'id': str(i), 'doc_id': f'chunk{i}', 'source_family': f'family{i // 2}'} for i in range(8)]
    rows.append({'id': 'bridge', 'doc_id': 'bridge', 'source_families': ['family0', 'family1']})
    groups = validation_groups(rows)
    assert groups[0] == groups[1] == groups[2] == groups[3] == groups[-1]
    train, val = split_by_document(rows, rows, .25, 42)
    groups_by_id = dict(zip((r['id'] for r in rows), groups))
    assert not ({groups_by_id[r['id']] for r in train} & {groups_by_id[r['id']] for r in val})
    assert len(train) + len(val) == len(rows)
    assert dict(zip((r['id'] for r in rows), groups)) == dict(zip(
        (r['id'] for r in reversed(rows)), validation_groups(list(reversed(rows)))))


def test_legacy_document_groups_unchanged_and_metadata_is_validated():
    rows = [{'id': 'a', 'doc_id': 'doc1'}, {'id': 'b', 'doc_id': 'doc1'}, {'id': 'c', 'doc_id': 'doc2'}]
    assert validation_groups(rows) == ['doc1', 'doc1', 'doc2']
    with pytest.raises(ValueError, match='source_families'):
        validation_groups([{'id': 'x', 'source_families': 'not-a-list'}])
    with pytest.raises(ValueError, match='align'):
        split_by_document(rows, [], .1, 42)


def test_one_connected_family_cannot_leak_to_make_a_split():
    rows = [{'id': 'a', 'source_family': 'same'}, {'id': 'b', 'source_family': 'same'}]
    with pytest.raises(ValueError, match='no training examples'):
        split_by_document(rows, rows, .2, 42)
