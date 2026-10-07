"""Real support/verify paths with fake reviewers; no model inference or training.

Captured conversations and reviews are exposed development fixtures. V3 responses
below are explicitly synthetic and cannot establish a real reviewer's accuracy.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from distillkit import support, support_v2 as v2, support_v3 as v3, verify
from distillkit.io import read_jsonl, write_jsonl
from distillkit.teacher import Completion
from test_job_status_workflow import call, fixture
from test_support_review import cfg, record, SOURCE, FakeReviewer, validate_run
from test_support_review_v2 import ROWS, SOURCE_CAPTURED, REVIEWS, payload as v2_payload, request as v2_request

GEMMA = Path(__file__).parent / 'fixtures/support_gemma_v2'
GEMMA_REVIEWS = read_jsonl(GEMMA / 'reviews_v2.jsonl')
SYNTHETIC_SOURCE = 'The synthetic scanner reports a count. A count alone does not establish a failure cause.'


def prose_record(cfg, answer, source=SYNTHETIC_SOURCE, question='Does the synthetic count establish a cause?'):
    (cfg.seeds.dir / 'unit_fixture.md').write_text(source)
    return {'id':'synthetic-unit-fixture', 'doc_id':'unit_fixture', 'source_kind':'full',
            'source_sha256':hashlib.sha256(source.encode()).hexdigest(), 'mode':'prose',
            'question':question, 'messages':[{'role':'user', 'content':question}, {'role':'assistant', 'content':answer}],
            'purpose':'diagnostic_only', 'training_eligible':False}


def eid(packet, origin):
    return next(k for k, v in packet['registry'].items() if v['origin'] == origin)


def payload(row, source=SYNTHETIC_SOURCE):
    packet = v3.evidence_packet(row, source)
    return {'units':[{'unit_id':u['unit_id'], 'verdict':'supported', 'evidence_ids':[eid(packet, 'source')],
                      'explanation':'Synthetic fixture: the source supports this stated limitation.'}
                     for u in packet['units']], 'uncertainty':{'present':False, 'reason':None}}


def request(row, review, source=SYNTHETIC_SOURCE):
    reviewer = FakeReviewer(review)
    report = support.request_review(reviewer, row, source, protocol=v3.PROTOCOL)
    assert len(reviewer.requests) == 1
    messages, kwargs = reviewer.requests[0]
    assert kwargs['temperature'] == 0
    assert kwargs['response_format']['json_schema']['name'] == 'SupportReviewV3'
    assert json.loads(messages[1]['content']) == v3.evidence_packet(row, source)
    return report


def run_v3(cfg, row, review, source=SYNTHETIC_SOURCE):
    cfg.verify.grounding_review_protocol = v3.PROTOCOL
    (cfg.seeds.dir / (row['doc_id'] + '.md')).write_text(source)
    return validate_run(cfg, row, [request(row, review, source)])


@pytest.mark.parametrize('text', [
    '  First observation. Second question? Third statement!\n',
    'Dr. Q measures 3.14 units, e.g. 2.71 here. Then the result is recorded.',
    'An approx. value is 1.25, i.e. a decimal. Another sentence follows.',
    'Use `printf "one. two"` once. Then ask a question?',
    'Use ``a `quoted. item` here`` once. Another sentence.',
    'Prefix:\n```python\nvalue = 3.14\nprint("One. Two")\n```\nA final sentence.',
    'A note:\n~~~text\nOne. Two\n~~~\nAnother note.',
    'Before:\n```sh\nunclosed code. still protected\n',
    'Overview:\n- First item\n- Second item.\n\n1. Numbered item\n2. Next item\n',
    '  αβγ. Another statement.  \r\n',
    'Same. Same. ',
])
def test_units_preserve_every_character_code_and_context(cfg, text):
    row = prose_record(cfg, text)
    packet = v3.evidence_packet(row, SYNTHETIC_SOURCE)
    units = packet['units']
    assert ''.join(u['text'] for u in units) == text
    assert ''.join(u['text'] for u in units).encode() == text.encode()
    assert units[0]['start'] == 0 and units[-1]['end'] == len(text)
    assert all(a['end'] == b['start'] for a, b in zip(units, units[1:]))
    assert all(text[u['start']:u['end']] == u['text'] for u in units)
    assert all(u['sha256'] == hashlib.sha256(u['text'].encode()).hexdigest() for u in units)
    assert len({u['unit_id'] for u in units}) == len(units)
    assert packet['turns'][0]['text'] == text
    assert packet == v3.evidence_packet(row, SYNTHETIC_SOURCE)
    protected, _ = v3._code_ranges(text)
    for start, end in protected:
        assert any(u['start'] <= start and u['end'] >= end for u in units)
    if 'Dr.' in text or 'approx.' in text:
        assert len(units) == 2


def test_unit_ids_are_content_bound_and_do_not_encode_hidden_labels(cfg):
    row = prose_record(cfg, 'The count is an observation. It does not establish the cause.')
    packet = v3.evidence_packet(row, SYNTHETIC_SOURCE)
    changed = copy.deepcopy(row)
    changed.update(id='arbitrary-case-name', expected_support='unsupported', reference='HIDDEN_EXPECTED_ANSWER', gold_verdict='reject')
    assert v3.evidence_packet(changed, SYNTHETIC_SOURCE) == packet
    changed['messages'][-1]['content'] = 'The count is an observation. It also guarantees success.'
    new = v3.evidence_packet(changed, SYNTHETIC_SOURCE)
    assert new['units'][0]['unit_id'] == packet['units'][0]['unit_id']
    assert new['units'][1]['unit_id'] != packet['units'][1]['unit_id']
    assert new['units_sha256'] != packet['units_sha256']
    assert all('HIDDEN_EXPECTED_ANSWER' not in e['content'] for e in packet['registry'].values())


def test_positive_prose_real_dispatch_verify_and_explicit_configuration(cfg):
    row = prose_record(cfg, 'The count by itself cannot establish the reason for failure.')
    kept, rejected, pending = run_v3(cfg, row, payload(row))
    assert len(kept) == 1 and not rejected and not pending
    check = kept[0]['grounding_checks']
    assert check['unit_version'] == v3.UNIT_VERSION
    assert 'fallible' in check['limitation']
    # Diagnostic fixtures stay outside training even when mechanically verified.
    from distillkit.records import training_example
    with pytest.raises(ValueError, match='diagnostic'):
        training_example(kept[0], 'Synthetic system prompt.')


def test_source_rule_application_does_not_require_verbatim_answer(cfg):
    source = 'The synthetic lookup requires an identifier. If it is absent, ask the user for that identifier.'
    row = prose_record(cfg, 'Please provide the identifier so I can perform the lookup.', source,
                       question='Please perform the lookup for my item.')
    data = payload(row, source)
    packet = v3.evidence_packet(row, source)
    data['units'][0]['evidence_ids'].append(eid(packet, 'user:0'))
    data['units'][0]['explanation'] = 'Synthetic judgment: applying the stated required-identifier rule to the user request.'
    assert len(run_v3(cfg, row, data, source)[0]) == 1


def test_future_intent_is_distinct_from_an_executed_observation(cfg):
    jid = fixture('FAILED')
    row = record(cfg, '', responses=[Completion('I will inspect its status.', [call(job_id=jid)]),
                                    Completion('The returned state is FAILED.')])
    packet = v3.evidence_packet(row, SOURCE)
    data = payload(row, SOURCE)
    data['units'][0].update(verdict='nonfactual', evidence_ids=[], explanation='Pure future intent, not an executed-action claim.')
    data['units'][1]['evidence_ids'] = [eid(packet, 'tool:2'), eid(packet, 'executed_call:2')]
    assert len(run_v3(cfg, row, data, SOURCE)[0]) == 1
    assert not any(e.startswith('tool:') for e in packet['turns'][0]['available_evidence_ids'])


@pytest.mark.parametrize('text', ['Which identifier should I inspect?', 'I will inspect the count.'])
def test_pure_question_or_intent_uses_one_nonfactual_verdict(cfg, text):
    row = prose_record(cfg, text)
    data = payload(row)
    data['units'][0].update(verdict='nonfactual', evidence_ids=[], explanation='Synthetic fixture: no factual technical assertion.')
    assert len(run_v3(cfg, row, data)[0]) == 1
    assert 'kind' not in v3.UnitReview.model_json_schema()['properties']
    assert 'support' not in v3.UnitReview.model_json_schema()['properties']


@pytest.mark.parametrize('text', [
    'Since the count proves hardware damage, should I reset the scanner?',
    'Because the count proves hardware damage, I will inspect it.',
    'I already inspected the scanner and confirmed hardware damage.',
])
def test_question_plan_and_past_action_premises_need_support(cfg, text):
    row = prose_record(cfg, text)
    data = payload(row)
    data['units'][0].update(verdict='unsupported', evidence_ids=[],
                           explanation='Synthetic judgment: the source establishes neither damage nor a completed inspection.')
    kept, rejected, pending = run_v3(cfg, row, data)
    assert not kept and not pending and rejected[0]['reject_reason'] == 'unsupported_claim'


@pytest.mark.parametrize('fault', ['omit', 'duplicate', 'unknown', 'old_kind_support', 'empty_explanation',
                                  'missing_citation', 'altered_citation', 'answer_as_evidence',
                                  'uncertain_unit', 'true_uncertainty', 'string_uncertainty', 'inconsistent_uncertainty'])
def test_incomplete_or_invalid_unit_reviews_remain_pending(cfg, fault):
    row = prose_record(cfg, 'A count is an observation. It does not establish a cause.')
    data = payload(row)
    if fault == 'omit': data['units'].pop()
    elif fault == 'duplicate': data['units'].append(copy.deepcopy(data['units'][0]))
    elif fault == 'unknown': data['units'][0]['unit_id'] = 'unit:unknown'
    elif fault == 'old_kind_support': data['units'][0].update(kind='assertion', support='supported')
    elif fault == 'empty_explanation': data['units'][0]['explanation'] = ' '
    elif fault == 'missing_citation': data['units'][0]['evidence_ids'] = []
    elif fault == 'altered_citation': data['units'][0]['evidence_ids'][0] += 'changed'
    elif fault == 'answer_as_evidence': data['units'][0]['evidence_ids'] = [data['units'][1]['unit_id']]
    elif fault == 'uncertain_unit': data['units'][0]['verdict'] = 'uncertain'
    elif fault == 'true_uncertainty': data['uncertainty'] = {'present':True, 'reason':'An interpretation remains unresolved.'}
    elif fault == 'string_uncertainty': data['uncertainty'] = 'none'
    elif fault == 'inconsistent_uncertainty': data['uncertainty'] = {'present':False, 'reason':'none'}
    kept, rejected, pending = run_v3(cfg, row, data)
    assert not kept and not rejected and len(pending) == 1


def test_any_uncertainty_keeps_mixed_verdict_record_pending(cfg):
    row = prose_record(cfg, 'A count is an observation. The scanner is guaranteed to be damaged.')
    data = payload(row)
    data['units'][1]['verdict'] = 'unsupported'
    data['uncertainty'] = {'present':True, 'reason':'Synthetic unresolved interpretation of the added assertion.'}
    kept, rejected, pending = run_v3(cfg, row, data)
    assert not kept and not rejected and len(pending) == 1


def test_unit_order_does_not_change_exact_id_coverage(cfg):
    row = prose_record(cfg, 'A count is an observation. It does not establish a cause.')
    data = payload(row); data['units'].reverse()
    assert len(run_v3(cfg, row, data)[0]) == 1


@pytest.mark.parametrize('fault', ['unit_version', 'units_sha256', 'registry_sha256', 'packet_sha256',
                                  'prompt_sha256', 'raw_response', 'source', 'transcript', 'reviewer', 'timestamp'])
def test_binding_changes_fail_closed(cfg, fault):
    row = prose_record(cfg, 'The count alone does not establish a cause.')
    report = request(row, payload(row)); source = SYNTHETIC_SOURCE
    if fault in ('unit_version', 'units_sha256', 'registry_sha256', 'packet_sha256', 'prompt_sha256'):
        report[fault] = 'changed'
    elif fault == 'raw_response': report['raw_response'] = '{}'
    elif fault == 'source': source += ' Changed source.'
    elif fault == 'transcript': row['messages'][-1]['content'] += ' Changed assertion.'
    elif fault == 'reviewer': report['reviewer'] = ''
    elif fault == 'timestamp': report['created_utc'] = '2026-01-01T00:00:00'
    assert support.check_support(row, source, report, protocol=v3.PROTOCOL)['status'] == 'uncertain'


def test_future_citation_is_invalid_even_when_intent_is_nonfactual(cfg):
    jid = fixture('FAILED')
    row = record(cfg, '', responses=[Completion('I will inspect its status.', [call(job_id=jid)]),
                                    Completion('The returned state is FAILED.')])
    data = payload(row, SOURCE); packet = v3.evidence_packet(row, SOURCE)
    data['units'][0].update(verdict='nonfactual', evidence_ids=[eid(packet, 'tool:2')])
    kept, rejected, pending = run_v3(cfg, row, data, SOURCE)
    assert not kept and not rejected and 'unavailable before' in pending[0]['grounding_checks']['reason']


def test_invalid_tool_evidence_fails_before_reviewer_request(cfg):
    jid = fixture('FAILED')
    row = record(cfg, '', responses=[Completion('', [call(job_id=jid)]), Completion('The state is FAILED.')])
    row['messages'][2]['content'] = '{"state":"INVENTED"}'
    fake = FakeReviewer({})
    with pytest.raises(ValueError, match='mock replay'):
        support.request_review(fake, row, SOURCE, protocol=v3.PROTOCOL)
    assert not fake.requests


def test_registry_and_chronology_are_exactly_v2_without_target_evidence(cfg):
    row = record(cfg, 'UNIQUE_ASSISTANT_ONLY_MARKER. A statement.')
    old, new = v2.evidence_packet(row, SOURCE), v3.evidence_packet(row, SOURCE)
    for field in ('registry', 'registry_sha256', 'turns', 'transcript_sha256', 'schemas_sha256', 'policy'):
        assert new[field] == old[field]
    assert all('UNIQUE_ASSISTANT_ONLY_MARKER' not in e['content'] for e in new['registry'].values())


def test_duplicate_json_keys_are_preserved_raw_and_not_repaired(cfg):
    row = prose_record(cfg, 'The count alone does not establish a cause.')
    raw = json.dumps(payload(row)).replace('"verdict": "supported"', '"verdict": "unsupported", "verdict": "supported"')
    class Reviewer:
        model = 'synthetic-duplicate-key-no-inference'
        def complete(self, *args, **kwargs): return Completion(raw)
    report = support.request_review(Reviewer(), row, SYNTHETIC_SOURCE, protocol=v3.PROTOCOL)
    assert report['raw_response'] == raw and report['review'] is None
    cfg.verify.grounding_review_protocol = v3.PROTOCOL
    kept, rejected, pending = validate_run(cfg, row, [report])
    assert not kept and not rejected and pending[0]['grounding_checks']['reason'] == 'malformed review'


@pytest.mark.parametrize('cid', ['js4-supported-hypothesis', 'js4-suspicion-not-evidence'])
def test_exposed_unsupported_addenda_have_separate_units_and_synthetic_rejections(cfg, cid):
    row = next(r for r in ROWS if r['id'] == cid)
    packet = v3.evidence_packet(row, SOURCE_CAPTURED); data = payload(row, SOURCE_CAPTURED)
    additions = ('configured or scheduled', 'runtime errors, segmentation faults', 'peak memory usage from job accounting')
    flagged = 0
    for unit, judgment in zip(packet['units'], data['units'], strict=True):
        if any(text in unit['text'] for text in additions):
            judgment.update(verdict='unsupported', evidence_ids=[],
                            explanation='Synthetic review: this addition imports history, causes or a diagnostic rule absent from the source.')
            flagged += 1
    assert flagged == (1 if cid.endswith('hypothesis') else 2)
    kept, rejected, pending = run_v3(cfg, row, data, SOURCE_CAPTURED)
    assert not kept and not pending and rejected[0]['reject_reason'] == 'unsupported_claim'
    # Omitting even one unsupported addendum makes the new report incomplete.
    data['units'] = [u for u in data['units'] if u['verdict'] != 'unsupported']
    kept, rejected, pending = run_v3(cfg, row, data, SOURCE_CAPTURED)
    assert not kept and not rejected and pending[0]['grounding_checks']['reason'] == 'unreviewed assistant target units'


def test_valid_citations_and_wrong_nonfactual_labels_are_not_semantic_proof(cfg):
    row = prose_record(cfg, 'The count proves an imaginary hardware cause.')
    wrong = payload(row)
    assert support.check_support(row, SYNTHETIC_SOURCE, request(row, wrong), protocol=v3.PROTOCOL)['status'] == 'supported'
    wrong['units'][0].update(verdict='nonfactual', evidence_ids=[], explanation='Deliberately wrong synthetic nonfactual label.')
    result = support.check_support(row, SYNTHETIC_SOURCE, request(row, wrong), protocol=v3.PROTOCOL)
    assert result['status'] == 'supported' and 'fallible' in result['limitation']
    # No deterministic grammar/ID heuristic claims to prove entailment or factuality.


def test_v1_v2_defaults_requests_and_captured_outcomes_are_unchanged(cfg):
    assert cfg.verify.grounding_review_protocol == support.PROTOCOL
    (cfg.seeds.dir / 'distillkit_job_status.md').write_text(SOURCE_CAPTURED)
    write_jsonl(cfg.run_dir / 'generated.jsonl', ROWS)
    write_jsonl(cfg.verify.grounding_reviews, REVIEWS)
    verify.run(cfg)
    assert len(read_jsonl(cfg.run_dir / 'pending_review.jsonl')) == 10
    assert not read_jsonl(cfg.run_dir / 'verified.jsonl') and not read_jsonl(cfg.run_dir / 'rejected.jsonl')
    cfg.verify.grounding_review_protocol = v2.PROTOCOL
    write_jsonl(cfg.verify.grounding_reviews, GEMMA_REVIEWS)
    verify.run(cfg)
    assert not read_jsonl(cfg.run_dir / 'verified.jsonl')
    assert len(read_jsonl(cfg.run_dir / 'rejected.jsonl')) == 9
    pending = read_jsonl(cfg.run_dir / 'pending_review.jsonl')
    assert len(pending) == 1 and pending[0]['id'] == 'js4-supported-hypothesis'
    # Historical reports cannot be silently treated as V3 reviews.
    assert support.check_support(ROWS[0], SOURCE_CAPTURED, GEMMA_REVIEWS[0], protocol=v3.PROTOCOL)['reason'] == 'stale or mismatched review binding'
    fresh = record(cfg, 'Those fields alone do not establish the cause.')
    assert v2_request(fresh, v2_payload(fresh))['protocol'] == v2.PROTOCOL
    manifest = json.loads((GEMMA / 'manifest.json').read_text())
    for name, sha in manifest['fixture_hashes'].items():
        assert hashlib.sha256((GEMMA / name).read_bytes()).hexdigest() == sha
