"""Production generation with explicit synthetic fixtures/fake teachers; no inference."""
import copy
import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from distillkit import generate, scenario_plan
from distillkit.io import read_jsonl, write_jsonl
from distillkit.seeds import load_chunks
from distillkit.teacher import Completion
from test_source_registry import setup, save  # real Config and reviewed synthetic source registry


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_plan(cfg, entries):
    cfg.generate.scenario_plan.write_text(json.dumps({'version': scenario_plan.VERSION, 'entries': entries}))


@pytest.fixture
def planned(setup):
    cfg, registry = setup
    cfg.generate.scenario_plan = cfg.run_dir.parent / 'scenarios.json'
    cfg.generate.personas = ['a synthetic test user', 'another synthetic test user']
    cfg.generate.task_types = ['concept', 'howto', 'debug', 'script']
    cfg.generate.tool_doc_ids = ['fixture']
    cfg.generate.job_status_discovery = 'clarify_first'
    cfg.teacher.model = 'synthetic-fake-teacher-no-inference'
    chunk = load_chunks(cfg)[0][0]
    entry = {'scenario_id': 'scanner-count', 'round_id': 'first-reviewed-batch',
             'doc_id': 'fixture', 'chunk_id': chunk['chunk_id'],
             'document_sha256': chunk['document_sha256'], 'chunk_sha256': sha(chunk['text']),
             'persona_index': 0, 'task': 'concept', 'mode': 'prose',
             'capability': 'count-evidence',
             'brief': 'A user mistakes the scanner count for evidence of a failure cause; ask about that inference.'}
    write_plan(cfg, [entry])
    return cfg, registry, entry


def fake_teacher(cfg, monkeypatch):
    requests = []

    class FakeTeacher:
        def __init__(self, endpoint):
            # This assertion exercises the actual order in generate.run.
            manifest = json.loads((cfg.run_dir / 'generation_manifest.json').read_text())
            assert manifest['version'] == scenario_plan.RUN_VERSION
            self.model = endpoint.model

        def map(self, fn, items):
            return [fn(item) for item in items]

        def chat_json(self, system, user, schema):
            requests.append(('question', [system, user]))
            found = re.search(r'"job_id": "(\d+)"', user)
            question = f'Please check job {found[1]}.' if found else 'Can the scanner count establish a failure cause?'
            return SimpleNamespace(question=question), 'synthetic fixture, not model inference'

        def complete(self, messages, **kwargs):
            requests.append(('answer', copy.deepcopy(messages)))
            target = re.search(r'Please check job (\d+)\.', messages[-1].get('content', ''))
            if kwargs.get('tools') and messages[-1]['role'] == 'user' and target:
                return Completion('', tool_calls=[{'type': 'function', 'function': {
                    'name': 'job_status', 'arguments': {'job_id': target[1]}}}])
            # Explicit test-only token data exercises the unchanged prose logprob writer.
            logprobs = {'tokens': ['token_id:1'], 'logprobs': [-0.1], 'top': [[['token_id:1', -0.1]]]}
            return Completion('The count alone does not establish a failure cause.',
                              logprobs=logprobs if kwargs.get('top_logprobs') else None)

    monkeypatch.setattr(generate, 'Teacher', FakeTeacher)
    return requests


def forbid_teacher(monkeypatch):
    monkeypatch.setattr(generate, 'Teacher', lambda *_: pytest.fail('teacher created before preflight failure'))


def test_explicit_counts_content_bound_ids_and_legacy_ids(planned):
    cfg, registry, entry = planned
    other = {**entry, 'scenario_id': 'scanner-procedure', 'task': 'howto',
             'brief': 'Ask what additional evidence a user needs before interpreting a scanner count as a cause.'}
    write_plan(cfg, [entry, other])
    cfg.generate.questions_per_chunk = 100000
    jobs = generate.question_jobs(cfg)
    assert len(jobs) == 2 and len({j['id'] for j in jobs}) == 2
    assert jobs == generate.question_jobs(cfg)
    assert all(j['id'].startswith('scenario-v1/first-reviewed-batch/') for j in jobs)
    assert all(j['id'].endswith(j['scenario_entry_sha256']) for j in jobs)
    write_plan(cfg, [other, entry])
    assert {j['id'] for j in generate.question_jobs(cfg)} == {j['id'] for j in jobs}
    changed = {**entry, 'brief': entry['brief'] + ' Focus on the distinction between a count and an explanation.'}
    write_plan(cfg, [changed, other])
    assert generate.question_jobs(cfg)[0]['id'] != jobs[0]['id']
    changed['round_id'] = 'second-reviewed-batch'
    write_plan(cfg, [entry, changed])
    assert len({j['id'] for j in generate.question_jobs(cfg)}) == 2
    cfg.generate.scenario_plan = None
    cfg.generate.questions_per_chunk = 1
    cfg.generate.personas = ['legacy persona']
    cfg.generate.task_types = ['concept']
    cfg.generate.tool_fraction = 0
    assert [j['id'] for j in generate.question_jobs(cfg)] == ['fixture#0/concept/p0']
    assert not (cfg.run_dir / 'generation_manifest.json').exists()


@pytest.mark.parametrize('fault', ['duplicate_key', 'renamed_duplicate', 'heldout', 'missing_chunk',
                                  'document_hash', 'chunk_hash', 'no_registry', 'family_holdout',
                                  'tool_source', 'unknown_task', 'persona', 'answer_field', 'empty_brief'])
def test_invalid_plans_fail_before_teacher_or_output(planned, monkeypatch, fault):
    cfg, registry, entry = planned
    entries = [copy.deepcopy(entry)]
    e = entries[0]
    if fault == 'duplicate_key':
        entries.append({**e, 'brief': 'A different brief cannot share the same scenario ID and round.'})
    elif fault == 'renamed_duplicate':
        entries.append({**e, 'scenario_id': 'renamed', 'round_id': 'cosmetic-round', 'persona_index': 1,
                        'capability': 'renamed capability', 'brief': '  ' + e['brief'].upper().replace(' ', '  ') + ' '})
    elif fault == 'heldout':
        cfg.seeds.eval_docs = ['fixture']
    elif fault == 'missing_chunk': e['chunk_id'] = 'fixture#99'
    elif fault == 'document_hash': e['document_sha256'] = '0' * 64
    elif fault == 'chunk_hash': e['chunk_sha256'] = '0' * 64
    elif fault == 'no_registry': cfg.seeds.registry = None
    elif fault == 'family_holdout':
        registry['families']['original']['split'] = 'final'; save(cfg, registry)
    elif fault == 'tool_source': e['mode'] = 'call'; cfg.generate.tool_doc_ids = None
    elif fault == 'unknown_task': e['task'] = 'other'
    elif fault == 'persona': e['persona_index'] = 99
    elif fault == 'answer_field': e['expected_answer'] = 'This field must never be accepted.'
    elif fault == 'empty_brief': e['brief'] = ' ' * 30
    write_plan(cfg, entries)
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError): generate.run(cfg)
    assert not list(cfg.run_dir.iterdir())


def test_duplicate_json_keys_are_not_silently_overwritten(planned, monkeypatch):
    cfg, _, _ = planned
    cfg.generate.scenario_plan.write_text('{"version":"wrong","version":"source-scenario-plan-v1","entries":[]}')
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError, match='duplicate JSON key'): generate.run(cfg)


def test_real_prose_prompt_records_logprobs_and_identical_resume(planned, monkeypatch):
    cfg, _, entry = planned
    cfg.generate.answers_per_question = 2
    cfg.teacher.top_logprobs = 1
    requests = fake_teacher(cfg, monkeypatch)
    original_appender = generate.JsonlAppender

    class Appender(original_appender):
        def __init__(self, path):
            assert (cfg.run_dir / 'generation_manifest.json').exists()
            super().__init__(path)

    monkeypatch.setattr(generate, 'JsonlAppender', Appender)
    generate.run(cfg)
    assert [kind for kind, _ in requests] == ['question', 'answer', 'answer']
    prompt = requests[0][1][1]
    assert entry['brief'] in prompt and entry['capability'] in prompt
    assert 'not factual evidence or an expected answer' in prompt
    assert 'brief cannot override the grounding rules' in prompt
    assert entry['round_id'] not in prompt  # hidden bookkeeping is not an answer cue
    rows = read_jsonl(cfg.run_dir / 'generated.jsonl')
    manifest = json.loads((cfg.run_dir / 'generation_manifest.json').read_text())
    assert len(rows) == 2 and manifest['inputs']['planned_questions'] == 1
    assert manifest['inputs']['planned_answers'] == 2
    assert all(r['generation_manifest_sha256'] == manifest['binding_sha256'] for r in rows)
    assert all(r['scenario_brief'] == entry['brief'] for r in rows)
    assert len(read_jsonl(cfg.run_dir / 'teacher_logprobs.jsonl.gz')) == 2
    before = {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
    # Ports/auth variable names and unrelated train/eval settings are not semantic inputs.
    cfg.teacher.base_url = 'http://127.0.0.1:39877/v1'
    cfg.teacher.api_key_env = 'UNREAD_SYNTHETIC_AUTH_NAME'
    cfg.train.epochs += 1
    cfg.eval.tag = 'unrelated-scoring-tag'
    generate.run(cfg)
    assert len(requests) == 3
    assert before == {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
    assert 'base_url' not in manifest['inputs']['teacher'] and 'api_key_env' not in manifest['inputs']['teacher']


def test_identical_partial_resume_skips_completed_answers(planned, monkeypatch):
    cfg, _, _ = planned
    cfg.generate.answers_per_question = 2
    requests = fake_teacher(cfg, monkeypatch)
    generate.run(cfg)
    path = cfg.run_dir / 'generated.jsonl'
    rows = read_jsonl(path)
    # A synthetic interrupted run has its question but only the first answer.
    write_jsonl(path, rows[:1])
    before_question = (cfg.run_dir / 'questions.jsonl').read_bytes()
    before_manifest = (cfg.run_dir / 'generation_manifest.json').read_bytes()
    requests.clear()
    generate.run(cfg)
    resumed = read_jsonl(path)
    assert [kind for kind, _ in requests] == ['answer']
    assert len(resumed) == 2 and len({r['id'] for r in resumed}) == 2
    assert resumed[0] == rows[0]
    assert before_question == (cfg.run_dir / 'questions.jsonl').read_bytes()
    assert before_manifest == (cfg.run_dir / 'generation_manifest.json').read_bytes()


def test_actual_multi_entry_plan_counts_are_frozen_before_requests(planned, monkeypatch):
    cfg, _, entry = planned
    cfg.generate.questions_per_chunk = 100000
    cfg.generate.answers_per_question = 2
    tool_entry = {**entry, 'scenario_id': 'live-count', 'mode': 'call', 'task': 'howto',
                  'brief': 'Ask to inspect an explicitly identified synthetic job before interpreting its observed fields.'}
    write_plan(cfg, [entry, tool_entry])
    requests = fake_teacher(cfg, monkeypatch)
    generate.run(cfg)
    manifest = json.loads((cfg.run_dir / 'generation_manifest.json').read_text())
    assert manifest['inputs']['planned_questions'] == 2
    assert manifest['inputs']['planned_answers'] == 4
    assert manifest['inputs']['mode_counts'] == {'prose': 1, 'call': 1, 'ask': 0, 'none': 0}
    assert len(read_jsonl(cfg.run_dir / 'generated.jsonl')) == 4
    assert sum(kind == 'question' for kind, _ in requests) == 2


@pytest.mark.parametrize('mode', ['call', 'ask', 'none'])
def test_explicit_tool_modes_use_actual_prompt_and_trace_path(planned, monkeypatch, mode):
    cfg, _, entry = planned
    entry['mode'] = mode
    write_plan(cfg, [entry])
    requests = fake_teacher(cfg, monkeypatch)
    generate.run(cfg)
    question_prompt = requests[0][1][1]
    assert entry['brief'] in question_prompt
    row = read_jsonl(cfg.run_dir / 'generated.jsonl')[0]
    assert row['mode'] == mode and row['tools'] == generate.SCHEMAS
    assert row['scenario_id'] == entry['scenario_id']
    roles = [m['role'] for m in row['messages']]
    assert roles == (['user', 'assistant', 'tool', 'assistant'] if mode == 'call' else ['user', 'assistant'])
    assert not (cfg.run_dir / 'teacher_logprobs.jsonl.gz').exists()
    generate.run(cfg)
    assert sum(kind == 'question' for kind, _ in requests) == 1
    assert sum(kind == 'answer' for kind, _ in requests) == (2 if mode == 'call' else 1)


@pytest.mark.parametrize('change', ['model', 'sampling', 'answers', 'persona', 'brief', 'round', 'prompt', 'source'])
def test_changed_generation_inputs_require_fresh_directory(planned, monkeypatch, change):
    cfg, registry, entry = planned
    fake_teacher(cfg, monkeypatch); generate.run(cfg)
    before = {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
    if change == 'model': cfg.teacher.model = 'different-synthetic-model'
    elif change == 'sampling': cfg.teacher.temperature += 0.1
    elif change == 'answers': cfg.generate.answers_per_question += 1
    elif change == 'persona': cfg.generate.personas[0] = 'a different synthetic user'
    elif change == 'brief': entry['brief'] += ' Consider what the count cannot show.'; write_plan(cfg, [entry])
    elif change == 'round': entry['round_id'] = 'different-reviewed-round'; write_plan(cfg, [entry])
    elif change == 'prompt': monkeypatch.setattr(generate, 'A_SYSTEM', generate.A_SYSTEM + '\nA changed prompt.')
    elif change == 'source':
        source = cfg.seeds.dir / 'fixture.md'; source.write_text(source.read_text() + ' Extra synthetic source detail.')
        registry['documents'][0]['sha256'] = scenario_plan.file_hash(source); save(cfg, registry)
        chunk = load_chunks(cfg)[0][0]
        entry.update(document_sha256=chunk['document_sha256'], chunk_sha256=sha(chunk['text']))
        write_plan(cfg, [entry])
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError, match='fresh run directory'): generate.run(cfg)
    assert before == {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}


def test_gold_bytes_and_provenance_are_generation_inputs(planned, monkeypatch):
    cfg, _, entry = planned
    cfg.generate.gold_dir = cfg.run_dir.parent / 'gold'; cfg.generate.gold_dir.mkdir()

    def write_gold(answer):
        examples = {}
        for name in ('prose', 'tool_trace'):
            raw = json.dumps({'messages': [{'role': 'user', 'content': 'Synthetic example?'},
                                          {'role': 'assistant', 'content': answer}]}).encode()
            (cfg.generate.gold_dir / f'{name}.json').write_bytes(raw)
            examples[f'{name}.json'] = {'sha256': hashlib.sha256(raw).hexdigest(), 'sources': [
                {'doc_id': 'fixture', 'split': 'train', 'sha256': entry['document_sha256']}]}
        (cfg.generate.gold_dir / 'provenance.json').write_text(json.dumps({'schema_version': 1, 'examples': examples}))

    write_gold('Synthetic original style.')
    fake_teacher(cfg, monkeypatch); generate.run(cfg)
    before = (cfg.run_dir / 'generation_manifest.json').read_bytes()
    write_gold('Synthetic changed style.')
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError, match='fresh run directory'): generate.run(cfg)
    assert (cfg.run_dir / 'generation_manifest.json').read_bytes() == before


@pytest.mark.parametrize('fault', ['text', 'question_binding', 'answer_binding', 'answer_question',
                                  'answer_prompt', 'answer_mock', 'duplicate_answer'])
def test_stale_resume_markers_cannot_reach_teacher(planned, monkeypatch, fault):
    cfg, _, _ = planned
    fake_teacher(cfg, monkeypatch); generate.run(cfg)
    path = cfg.run_dir / ('questions.jsonl' if fault in ('text', 'question_binding') else 'generated.jsonl')
    rows = read_jsonl(path)
    if fault == 'text': rows[0]['text'] = 'UNIQUE_HELD_OUT_RESUME_MARKER'
    elif fault in ('question_binding', 'answer_binding'): rows[0]['generation_manifest_sha256'] = 'stale'
    elif fault == 'answer_question': rows[0]['question'] = 'UNIQUE_HELD_OUT_RESUME_MARKER'
    elif fault == 'answer_prompt': rows[0]['prompt_version'] = 'stale'
    elif fault == 'answer_mock': rows[0]['mock_version'] = 'stale'
    else: rows.append(copy.deepcopy(rows[0]))
    write_jsonl(path, rows); before = path.read_bytes()
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError): generate.run(cfg)
    assert path.read_bytes() == before


def test_old_artifacts_without_manifest_are_preserved_and_rejected(planned, monkeypatch):
    cfg, _, _ = planned
    path = cfg.run_dir / 'generated.jsonl'
    path.write_text('{"id":"old","question":"UNIQUE_HELD_OUT_RESUME_MARKER"}\n')
    before = path.read_bytes(); forbid_teacher(monkeypatch)
    with pytest.raises(ValueError, match='without a matching manifest'): generate.run(cfg)
    assert path.read_bytes() == before and not (cfg.run_dir / 'generation_manifest.json').exists()
