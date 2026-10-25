"""Actual generation/planner/records with synthetic inputs and a fake teacher only."""
import copy
import hashlib
import json

import pytest

from distillkit import generate, scenario_plan
from distillkit.io import read_jsonl, write_jsonl
from distillkit.records import training_example
from distillkit.seeds import load_chunks
from test_prose_answer_scope import teacher_fixture
from test_scenario_plan import forbid_teacher, planned, write_plan
from test_source_registry import setup, save


PRECONDITION = 'The input named π has three entries; only the shown fragment is in scope.'
FRAGMENT = 'int count = 3;\nint total = count + 1;'
SOURCE = ('# Synthetic context fixture\n\n' + PRECONDITION + '\n\n```c\n' + FRAGMENT +
          '\n```\n\n@@expect: SYNTHETIC_ANSWER_MARKER_NOT_PROBLEM_INPUT\n\n'
          'This last explanatory paragraph is not selected as user input.')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_v2(cfg, entries):
    cfg.generate.scenario_plan.write_text(json.dumps({
        'version': scenario_plan.CONTEXT_PLAN_VERSION, 'entries': entries}, ensure_ascii=False))


def bind_source(cfg, registry, entry, source, parts):
    """Synthetic fixtures only: all context positions address the real loaded chunk."""
    path = cfg.seeds.dir / 'fixture.md'
    path.write_bytes(source.encode())
    registry['documents'][0]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(cfg, registry)
    chunk = load_chunks(cfg)[0][0]
    entry.update(document_sha256=chunk['document_sha256'], chunk_sha256=sha(chunk['text']))
    spans, cursor = [], 0
    for part in parts:
        start = chunk['text'].index(part, cursor)
        spans.append({'start': start, 'end': start + len(part), 'sha256': sha(part)})
        cursor = start + len(part)
    entry['context_spans'] = spans
    write_v2(cfg, [entry])
    return chunk


@pytest.fixture
def context_plan(planned):
    cfg, registry, entry = planned
    chunk = bind_source(cfg, registry, entry, SOURCE, [PRECONDITION, FRAGMENT])
    return cfg, registry, entry, chunk


def test_v1_ids_digests_and_rendered_prompts_remain_unchanged(planned, monkeypatch):
    cfg, _, entry = planned
    # The pre-v2 formula is intentional: an optional field must not silently
    # enter v1 model_dump() and change historical IDs.
    digest = sha(json.dumps({'version': 'source-scenario-plan-v1', 'entry': entry,
                            'persona': cfg.generate.personas[0]}, sort_keys=True,
                           ensure_ascii=False, separators=(',', ':')))
    job = generate.question_jobs(cfg)[0]
    assert job['scenario_entry_sha256'] == digest
    assert job['id'] == f"scenario-v1/{entry['round_id']}/{entry['scenario_id']}/{digest}"
    assert 'context_spans' not in scenario_plan.Entry.model_validate(entry).model_dump()
    raw_question = 'What does the synthetic scanner count establish?'
    requests, _ = teacher_fixture(monkeypatch, [raw_question])
    generate.run(cfg)
    expected = generate.Q_PROMPT.format(chunk=job['text'], persona=job['persona'], task=job['task'],
                                        mode_rule=generate._mode_rule(job, {}))
    expected += scenario_plan.question_instruction(job)
    assert requests[0][1] == [generate.Q_SYSTEM, expected]
    assert requests[1][1][0]['content'] == generate.A_SYSTEM
    row = read_jsonl(cfg.run_dir / 'generated.jsonl')[0]
    assert row['question'] == raw_question and row['prompt_version'] == generate.PROMPT_VERSION
    assert not any(k in row for k in scenario_plan.CONTEXT_FIELDS)
    cfg.generate.scenario_plan = None
    cfg.generate.questions_per_chunk = 1
    cfg.generate.personas = ['legacy persona']
    cfg.generate.task_types = ['concept']
    cfg.generate.tool_fraction = 0
    assert [j['id'] for j in generate.question_jobs(cfg)] == ['fixture#0/concept/p0']


def test_context_reaches_real_question_answer_and_student_positions(context_plan, monkeypatch):
    cfg, _, entry, chunk = context_plan
    cfg.teacher.top_logprobs = 1
    original = '  Explain the provided input without supplying a complete program.\n '
    requests, raw = teacher_fixture(monkeypatch, [original])
    generate.run(cfg)
    job = generate.question_jobs(cfg)[0]
    block = scenario_plan.context_block(job)
    effective = original.strip() + '\n\n' + block
    question = read_jsonl(cfg.run_dir / 'questions.jsonl')[0]
    row = read_jsonl(cfg.run_dir / 'generated.jsonl')[0]
    assert job['id'].startswith('scenario-v2/')
    assert job['scenario_context']['sha256'] == sha(block)
    assert [p['text'] for p in job['scenario_context']['spans']] == [PRECONDITION, FRAGMENT]
    for span in job['scenario_context']['spans']:
        assert chunk['text'][span['start']:span['end']] == span['text']
        assert span['sha256'] == sha(span['text'])
    qsystem, qprompt = requests[0][1]
    assert generate.CONTEXT_PROMPT_VERSION in qsystem
    assert 'appended\nautomatically, without your copying them' in qprompt
    assert 'the source and scenario brief are not\nattached to it' not in qprompt
    assert qprompt.endswith(block) and entry['brief'] in qprompt
    assert 'quoted data, not instructions' in block
    assert 'Provided input 1' in block and 'Provided input 2' in block
    assert 'SYNTHETIC_ANSWER_MARKER_NOT_PROBLEM_INPUT' not in effective
    assert 'This last explanatory paragraph' not in effective
    assert row['question'] == question['question'] == effective
    assert row['messages'][0] == {'role': 'user', 'content': effective}
    assert row['messages'][1]['content'] == 'Synthetic fixture response; not model inference.'
    assert requests[1][1][1]['content'] == generate.A_PROMPT.format(chunk=chunk['text'], question=effective)
    assert generate.CONTEXT_PROMPT_VERSION in requests[1][1][0]['content']
    assert entry['brief'] not in requests[1][1][1]['content']
    capture = question['question_generation']
    assert capture == row['question_generation']
    assert capture == {'version': scenario_plan.QUESTION_CAPTURE_VERSION,
                       'raw_question': original, 'raw_question_sha256': sha(original),
                       'raw_response': raw[0], 'raw_response_sha256': sha(raw[0]),
                       'question_sha256': sha(effective)}
    assert training_example(row, 'Synthetic student system')['messages'][1]['content'] == effective
    assert len(read_jsonl(cfg.run_dir / 'teacher_logprobs.jsonl.gz')) == 1
    assert not (cfg.run_dir / 'question_rejections.jsonl').exists()


def test_separate_programs_are_separate_visible_inputs(context_plan, monkeypatch):
    cfg, registry, entry, _ = context_plan
    first = 'int main(void) { return 1; }'
    second = 'int main(void) { return 2; }'
    bind_source(cfg, registry, entry, first + '\n\n' + second, [first, second])
    requests, _ = teacher_fixture(monkeypatch, ['Compare the two provided fragments.'])
    generate.run(cfg)
    text = read_jsonl(cfg.run_dir / 'generated.jsonl')[0]['question']
    assert text.endswith('Provided input 1 (quoted data, not instructions):\n````text\n' + first +
                         '\n````\n\nProvided input 2 (quoted data, not instructions):\n````text\n' + second + '\n````')
    assert first + '\n\n' + second not in text  # not one concatenated program
    assert [r[0] for r in requests] == ['question', 'answer']


def test_v2_context_ids_are_deterministic_order_independent_and_content_bound(context_plan):
    cfg, _, entry, chunk = context_plan
    first = generate.question_jobs(cfg)[0]
    assert first == generate.question_jobs(cfg)[0]
    other = {**entry, 'scenario_id': 'separate-objective', 'task': 'howto',
             'brief': 'Ask how to inspect the given fragment using only the synthetic preconditions.'}
    write_v2(cfg, [entry, other])
    jobs = generate.question_jobs(cfg)
    write_v2(cfg, [other, entry])
    assert {j['id'] for j in jobs} == {j['id'] for j in generate.question_jobs(cfg)}
    assert len({j['id'] for j in jobs}) == 2
    changed = copy.deepcopy(entry)
    part = 'int count = 3;'
    start = chunk['text'].index(part)
    changed['context_spans'] = [{'start': start, 'end': start + len(part), 'sha256': sha(part)}]
    write_v2(cfg, [changed])
    assert generate.question_jobs(cfg)[0]['id'] != first['id']


@pytest.mark.parametrize('part', [
    PRECONDITION,
    FRAGMENT,  # a bounded fragment needs neither main() nor an invented surrounding fence
    '```c\n' + FRAGMENT + '\n```',
    '~~~text\n' + PRECONDITION + '\n~~~',
    'printf("```");\n// A literal ``` marker is data.',
    '````text\n```\n````',
    '  int count = 3;\n\tint total = count + 1;  ',
])
def test_plain_preconditions_fragments_and_complete_fences_remain_exact(context_plan, monkeypatch, part):
    cfg, registry, entry, _ = context_plan
    # Keep selected outer whitespace inside a non-whitespace paragraph, so this
    # specifically tests selection from the actual normalized chunk.
    bind_source(cfg, registry, entry, 'Synthetic prefix.\n' + part + '\nSynthetic suffix.', [part])
    requests, _ = teacher_fixture(monkeypatch, ['Explain only the provided input.'])
    generate.run(cfg)
    row = read_jsonl(cfg.run_dir / 'generated.jsonl')[0]
    assert row['scenario_context']['spans'][0]['text'] == part
    assert '\n' + part + '\n' in row['question']
    assert [r[0] for r in requests] == ['question', 'answer']


def test_offsets_use_loaded_unicode_normalized_chunk_not_original_file(context_plan, monkeypatch):
    cfg, registry, entry, _ = context_plan
    raw_source = ('\n\n# π fixture\n\n' + PRECONDITION + '\n\n' + FRAGMENT + '\n\n').replace('\n', '\r\n')
    chunk = bind_source(cfg, registry, entry, raw_source, [PRECONDITION, FRAGMENT])
    assert '\r' not in chunk['text'] and chunk['text'].startswith('# π fixture')
    assert chunk['document_sha256'] != sha(chunk['text'])
    requests, _ = teacher_fixture(monkeypatch, ['Explain the provided fragment and its precondition.'])
    generate.run(cfg)
    row = read_jsonl(cfg.run_dir / 'generated.jsonl')[0]
    assert row['scenario_context']['spans'][0]['text'] == PRECONDITION
    assert 'π' in row['question'] and [r[0] for r in requests] == ['question', 'answer']


@pytest.mark.parametrize('fault', ['negative', 'boolean', 'fractional', 'past_end', 'reversed', 'overlap',
                                  'out_of_order', 'wrong_hash', 'empty_list', 'whitespace', 'extra_field',
                                  'v1_context', 'wrong_document', 'wrong_chunk', 'heldout', 'family_holdout'])
def test_invalid_context_fails_before_teacher_and_manifest(context_plan, monkeypatch, fault):
    cfg, registry, entry, chunk = context_plan
    entry = copy.deepcopy(entry)
    spans = entry['context_spans']
    if fault == 'negative': spans[0]['start'] = -1
    elif fault == 'boolean': spans[0]['start'] = True
    elif fault == 'fractional': spans[0]['end'] = 1.5
    elif fault == 'past_end': spans[-1]['end'] = len(chunk['text']) + 1
    elif fault == 'reversed': spans[0]['end'] = spans[0]['start']
    elif fault == 'overlap': spans.append(copy.deepcopy(spans[-1]))
    elif fault == 'out_of_order': spans.reverse()
    elif fault == 'wrong_hash': spans[0]['sha256'] = '0' * 64
    elif fault == 'empty_list': entry['context_spans'] = []
    elif fault == 'whitespace':
        start = chunk['text'].index('\n\n')
        entry['context_spans'] = [{'start': start, 'end': start + 2, 'sha256': sha('\n\n')}]
    elif fault == 'extra_field': spans[0]['expected_answer'] = 'A forbidden invented answer field.'
    elif fault == 'wrong_document': entry['document_sha256'] = '0' * 64
    elif fault == 'wrong_chunk': entry['chunk_sha256'] = '0' * 64
    elif fault == 'heldout': cfg.seeds.eval_docs = ['fixture']
    elif fault == 'family_holdout': registry['families']['original']['split'] = 'final'; save(cfg, registry)
    if fault == 'v1_context': write_plan(cfg, [entry])
    else: write_v2(cfg, [entry])
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError): generate.run(cfg)
    assert not list(cfg.run_dir.iterdir())


@pytest.mark.parametrize('mode', ['call', 'ask', 'none'])
def test_context_cannot_change_tool_authority_or_clarification(context_plan, monkeypatch, mode):
    cfg, _, entry, _ = context_plan
    entry['mode'] = mode
    write_v2(cfg, [entry])
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError, match='only for prose'): generate.run(cfg)
    assert not list(cfg.run_dir.iterdir())


def test_v2_without_context_retains_missing_job_clarification(planned, monkeypatch):
    cfg, _, entry = planned
    entry['mode'] = 'ask'
    write_v2(cfg, [entry])
    requests, _ = teacher_fixture(monkeypatch, ["Please check my job's current state."])
    generate.run(cfg)
    row = read_jsonl(cfg.run_dir / 'generated.jsonl')[0]
    assert row['mode'] == 'ask' and row['question'] == "Please check my job's current state."
    assert 'scenario_context' not in row and row['prompt_version'] == generate.PROMPT_VERSION
    assert requests[1][2]['tools'] == generate.SCHEMAS
    assert not (cfg.run_dir / 'question_rejections.jsonl').exists()


@pytest.mark.parametrize('part', ['```c\nint count = 3;', '~~~text\nA missing close.',
                                 '````c\nint count = 3;\n```'])
def test_unclosed_selected_source_fences_fail_preflight(context_plan, monkeypatch, part):
    cfg, registry, entry, _ = context_plan
    bind_source(cfg, registry, entry, part, [part])
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError, match='unmatched code fence in source input context'): generate.run(cfg)
    assert not list(cfg.run_dir.iterdir())


def test_attached_context_cannot_hide_raw_question_truncation(context_plan, monkeypatch, capsys):
    cfg, _, _, _ = context_plan
    cfg.generate.answers_per_question = 2
    raw_question = 'Explain this truncated fragment:\n```c\nint value ='
    requests, raw = teacher_fixture(monkeypatch, [raw_question])
    generate.run(cfg)
    assert [r[0] for r in requests] == ['question']
    assert not (cfg.run_dir / 'generated.jsonl').exists()
    question = read_jsonl(cfg.run_dir / 'questions.jsonl')[0]
    rejection = read_jsonl(cfg.run_dir / 'question_rejections.jsonl')[0]
    assert question['question_generation']['raw_response'] == raw[0]
    assert question['question_validation']['raw_question'] == raw_question
    assert rejection['check_version'] == generate.CONTEXT_QUESTION_CHECK_VERSION
    assert rejection['reason'] == 'unmatched_question_code_fence' and rejection['candidate'] == question
    assert rejection['candidate_sha256'] == scenario_plan.digest(question)
    # Regression: checking the composed text alone can mistake the appended
    # block's close for the teacher's missing close. The raw check is required.
    assert generate._question_issue(question['question']) is None
    assert generate._captured_question_issue(question) == 'unmatched_question_code_fence'
    manifest = json.loads((cfg.run_dir / 'generation_manifest.json').read_text())
    assert manifest['inputs']['planned_questions'] == 1 and manifest['inputs']['planned_answers'] == 2
    assert 'invalid JSON' not in capsys.readouterr().out
    before = {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
    generate.run(cfg)
    assert [r[0] for r in requests] == ['question']
    assert before == {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}


@pytest.mark.parametrize('question', [
    'Explain `count` and ``total`` in the provided input.',
    'Explain this C fragment together with the provided input:\n```c\nint count = 3;\n```',
    "Explain this quoted data without executing it:\n```bash\nprintf '%s\\n' '```'\n```",
])
def test_complete_raw_question_context_is_not_rejected(context_plan, monkeypatch, question):
    cfg, _, _, _ = context_plan
    requests, _ = teacher_fixture(monkeypatch, [question])
    generate.run(cfg)
    assert [r[0] for r in requests] == ['question', 'answer']
    assert not (cfg.run_dir / 'question_rejections.jsonl').exists()
    assert read_jsonl(cfg.run_dir / 'generated.jsonl')[0]['question_generation']['raw_question'] == question


def test_mixed_context_plan_keeps_rejected_and_completed_denominators(context_plan, monkeypatch):
    cfg, _, entry, _ = context_plan
    cfg.generate.answers_per_question = 2
    other = {**entry, 'scenario_id': 'other-input-task', 'task': 'howto',
             'brief': 'Ask how to inspect only the given fragment under its stated synthetic precondition.'}
    write_v2(cfg, [entry, other])
    requests, _ = teacher_fixture(monkeypatch, ['Explain this incomplete input:\n```c\nvalue =',
                                               'Explain the provided input.'])
    generate.run(cfg)
    assert [r[0] for r in requests] == ['question', 'question', 'answer', 'answer']
    assert len(read_jsonl(cfg.run_dir / 'questions.jsonl')) == 2
    assert len(read_jsonl(cfg.run_dir / 'question_rejections.jsonl')) == 1
    rows = read_jsonl(cfg.run_dir / 'generated.jsonl')
    assert len(rows) == 2 and all(r['scenario_id'] == 'other-input-task' for r in rows)
    manifest = json.loads((cfg.run_dir / 'generation_manifest.json').read_text())
    assert manifest['inputs']['planned_questions'] == 2 and manifest['inputs']['planned_answers'] == 4


def test_identical_and_partial_resume_preserve_raw_and_effective_question(context_plan, monkeypatch):
    cfg, _, _, _ = context_plan
    cfg.generate.answers_per_question = 2
    requests, _ = teacher_fixture(monkeypatch, ['Explain the provided input.'])
    generate.run(cfg)
    before = {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
    requests.clear()
    cfg.teacher.base_url = 'http://127.0.0.1:39877/v1'
    generate.run(cfg)
    assert requests == [] and before == {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
    rows = read_jsonl(cfg.run_dir / 'generated.jsonl')
    write_jsonl(cfg.run_dir / 'generated.jsonl', rows[:1])
    generate.run(cfg)
    assert [r[0] for r in requests] == ['answer']
    resumed = read_jsonl(cfg.run_dir / 'generated.jsonl')
    assert len(resumed) == 2 and resumed[0] == rows[0]
    assert all(r['question_generation'] == rows[0]['question_generation'] for r in resumed)
    assert (cfg.run_dir / 'questions.jsonl').read_bytes() == before['questions.jsonl']


@pytest.mark.parametrize('fault', ['question_text', 'span_text', 'span_hash', 'span_offset', 'context_hash',
                                  'raw_question', 'raw_response', 'raw_hash', 'response_question_mismatch', 'capture_version',
                                  'answer_context', 'answer_capture', 'answer_user', 'answer_extra_user',
                                  'answer_extra_system', 'answer_extra_assistant', 'answer_tools',
                                  'answer_tool_calls', 'answer_assistant_role'])
def test_stale_context_raw_or_actual_user_binding_stops_before_teacher(context_plan, monkeypatch, fault):
    cfg, _, _, _ = context_plan
    teacher_fixture(monkeypatch, ['Explain the provided input.'])
    generate.run(cfg)
    path = cfg.run_dir / ('generated.jsonl' if fault.startswith('answer_') else 'questions.jsonl')
    rows = read_jsonl(path)
    row = rows[0]
    if fault == 'question_text': row['question'] += ' UNIQUE_HELD_OUT_RESUME_MARKER'
    elif fault == 'span_text': row['scenario_context']['spans'][0]['text'] = 'UNIQUE_HELD_OUT_RESUME_MARKER'
    elif fault == 'span_hash': row['scenario_context']['spans'][0]['sha256'] = '0' * 64
    elif fault == 'span_offset': row['scenario_context']['spans'][0]['start'] += 1
    elif fault == 'context_hash': row['scenario_context']['sha256'] = '0' * 64
    elif fault == 'raw_question': row['question_generation']['raw_question'] = 'UNIQUE_HELD_OUT_RESUME_MARKER'
    elif fault == 'raw_response': row['question_generation']['raw_response'] = '{}'
    elif fault == 'raw_hash': row['question_generation']['raw_response_sha256'] = '0' * 64
    elif fault == 'response_question_mismatch':
        raw = json.dumps({'question': 'UNIQUE_HELD_OUT_RESUME_MARKER'})
        row['question_generation'].update(raw_response=raw, raw_response_sha256=sha(raw))
    elif fault == 'capture_version': row['question_generation']['version'] = 'unrecognized-capture'
    elif fault == 'answer_context': row['scenario_context']['sha256'] = '0' * 64
    elif fault == 'answer_capture': del row['question_generation']
    elif fault == 'answer_extra_user': row['messages'].append({'role': 'user', 'content': 'UNIQUE_HELD_OUT_RESUME_MARKER'})
    elif fault == 'answer_extra_system': row['messages'].append({'role': 'system', 'content': 'UNIQUE_HELD_OUT_RESUME_MARKER'})
    elif fault == 'answer_extra_assistant': row['messages'].append({'role': 'assistant', 'content': 'UNIQUE_HELD_OUT_RESUME_MARKER'})
    elif fault == 'answer_tools': row['tools'] = []
    elif fault == 'answer_tool_calls': row['messages'][1]['tool_calls'] = []
    elif fault == 'answer_assistant_role': row['messages'][1]['role'] = 'user'
    else: row['messages'][0]['content'] = 'UNIQUE_HELD_OUT_RESUME_MARKER'
    write_jsonl(path, rows)
    before = {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError, match='fresh run directory'): generate.run(cfg)
    assert before == {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}


@pytest.mark.parametrize('change', ['span_selection', 'context_prompt', 'context_version', 'check_version'])
def test_changed_context_inputs_require_fresh_run(context_plan, monkeypatch, change):
    cfg, _, entry, _ = context_plan
    teacher_fixture(monkeypatch, ['Explain the provided input.'])
    generate.run(cfg)
    before = {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}
    if change == 'span_selection':
        entry['context_spans'] = entry['context_spans'][:1]
        write_v2(cfg, [entry])
    elif change == 'context_prompt': monkeypatch.setattr(generate, 'CONTEXT_Q_PROMPT', generate.CONTEXT_Q_PROMPT + '\nChanged.')
    elif change == 'context_version': monkeypatch.setattr(scenario_plan, 'CONTEXT_VERSION', 'synthetic-next-context')
    else: monkeypatch.setattr(generate, 'CONTEXT_QUESTION_CHECK_VERSION', 'synthetic-next-check')
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError, match='fresh run directory'): generate.run(cfg)
    assert before == {p.name: p.read_bytes() for p in cfg.run_dir.iterdir()}


def test_v1_cached_row_cannot_opt_itself_into_context(planned, monkeypatch):
    cfg, _, _ = planned
    teacher_fixture(monkeypatch, ['What does the synthetic scanner count establish?'])
    generate.run(cfg)
    path = cfg.run_dir / 'questions.jsonl'
    rows = read_jsonl(path)
    rows[0]['scenario_context'] = {'text': 'UNIQUE_HELD_OUT_RESUME_MARKER'}
    write_jsonl(path, rows)
    forbid_teacher(monkeypatch)
    with pytest.raises(ValueError, match='input-context/raw bindings'): generate.run(cfg)
