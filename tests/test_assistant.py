"""hpc-assistant: built-in retrieval must give the student the exact prompts it was evaluated with."""
import importlib.util
import json
import sqlite3
from io import StringIO
from pathlib import Path

from distillkit import assistant

ROOT = Path(__file__).resolve().parents[1]


def _index(path, rows, site_col):
    db = sqlite3.connect(path)
    db.execute(f'CREATE TABLE chunks (id INTEGER PRIMARY KEY, url TEXT, {site_col} TEXT, title TEXT, text TEXT, retrieved_at TEXT)')
    db.execute("CREATE VIRTUAL TABLE search USING fts5(title, text, content='chunks', content_rowid='id')")
    for i, (url, site, text) in enumerate(rows, 1):
        db.execute(f'INSERT INTO chunks (id, url, {site_col}, title, text, retrieved_at) VALUES (?,?,?,?,?,?)',
                   (i, url, site, 'T', text, '2026-10-13'))
        db.execute('INSERT INTO search (rowid, title, text) VALUES (?,?,?)', (i, 'T', text))
    db.commit()
    db.close()


def _indexes(tmp_path):
    _index(tmp_path / 'documents.sqlite', [
        ('https://docs.lumi/gpu', 'LUMI', 'Request GPUs on LUMI with --gpus-per-node in the standard-g partition.'),
        ('https://slurm/sbatch', 'Slurm', 'sbatch submits a batch script; use --time to set the wall clock limit.'),
        ('https://cineca/leo', 'CINECA', 'Leonardo booster partition boost_usr_prod has four A100 GPUs per node.')], 'site')
    _index(tmp_path / 'factory_docs.sqlite', [
        ('https://jsc/juwels', 'JUPITER AI Factory', 'On JUWELS Booster request GPUs with --gres=gpu:4 per node.')],
        'factories')
    return tmp_path


def test_retrieval_matches_eval_retriever(tmp_path):
    spec = importlib.util.spec_from_file_location('rag65_retrieve', ROOT / 'research/rag65/retrieve.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    idx = _indexes(tmp_path)
    p, f = idx / 'documents.sqlite', idx / 'factory_docs.sqlite'
    for q in ['How do I request GPUs on LUMI?', 'Set a time limit with sbatch', 'GPUs per node on Leonardo booster?',
              'How many GPUs per node on JUWELS?', 'the and for']:
        assert assistant.retrieve(q, p, f) == ref.retrieve(q, p, f)
        assert assistant.with_context(q, assistant.retrieve(q, p, f)) == ref.with_context(q, ref.retrieve(q, p, f))
    assert assistant.FACTORY_SITES == ref.FACTORY_SITES and assistant.STOP == ref.STOP


def test_system_prompt_matches_training_config():
    cfg = json.loads((ROOT / 'configs/rag65v4_seed42.json').read_text())
    assert assistant.SYSTEM_PROMPT == cfg['student']['system_prompt']


def test_missing_index_answers_closed_book(tmp_path):
    bot = assistant.Assistant(tmp_path / 'none')
    assert not bot.builtin
    assert bot.prompt('How do I request GPUs on LUMI?') == ('How do I request GPUs on LUMI?', [])


def test_ask_streams_answer_with_sources(tmp_path, monkeypatch):
    seen = {}

    def fake_stream(self, messages):  # the conftest blocks sockets, so the model server is stubbed
        seen['messages'] = messages
        yield from ['Use ', '--gpus-per-node.']

    monkeypatch.setattr(assistant.Assistant, '_stream', fake_stream)
    bot = assistant.Assistant(_indexes(tmp_path))
    out = StringIO()
    answer, sources = bot.ask('How do I request GPUs on LUMI?', out)
    assert answer == 'Use --gpus-per-node.' and sources == ['https://docs.lumi/gpu']
    assert seen['messages'][0] == {'role': 'system', 'content': assistant.SYSTEM_PROMPT}
    assert '<quoted_source>' in seen['messages'][1]['content'] and 'standard-g' in seen['messages'][1]['content']
    assert 'Sources:\n- https://docs.lumi/gpu' in out.getvalue()


def test_user_manual_is_indexed_searched_first_and_replaced_on_readd(tmp_path):
    idx = _indexes(tmp_path)
    manual = tmp_path / 'mycluster.md'
    manual.write_text('# MyCluster\n\nSubmit GPU jobs to the partition gpu-long with --gres=gpu:a100:2 and a '
                      'maximum wall time of 48 hours on MyCluster.\n\nUnrelated paragraph about the cafeteria menu, '
                      'which lists soups and salads every weekday for staff.')
    bot = assistant.Assistant(idx)
    assert bot.add(manual) == 1  # short paragraphs pack into one excerpt
    docs = bot.search('Which partition for GPU jobs on MyCluster, and how do I set the time with sbatch?')
    assert docs[0]['url'] == 'mycluster.md' and docs[0]['site'] == 'Your documents'
    assert len(docs) == 2 and docs[1]['url'] == 'https://slurm/sbatch'  # built-in fills the rest
    manual.write_text('MyCluster moved GPU jobs to the partition gpu-new; use --gres=gpu:h100:1 from now on please.')
    assert bot.add(manual) == 1
    texts = [d['text'] for d in bot.search('partition GPU MyCluster')]
    assert any('gpu-new' in t for t in texts) and not any('gpu-long' in t for t in texts)


def test_html_manual_drops_markup_and_scripts(tmp_path):
    page = tmp_path / 'guide.html'
    page.write_text('<html><script>var x = "secret";</script><body><h1>Guide</h1><p>' + 'Use sbatch to submit jobs. ' * 5 +
                    '</p></body></html>')
    [(loc, text)] = assistant.read_document(page)
    assert loc == 'guide.html' and 'sbatch' in text and 'secret' not in text and '<p>' not in text


def test_model_can_call_search_docs_tool_and_chat_keeps_history(tmp_path, monkeypatch):
    calls = []

    def fake_stream(self, messages):
        calls.append([dict(m) for m in messages])
        if len(calls) == 1:
            yield {'tool_calls': [{'name': 'search_docs', 'arguments': '{"query": "JUWELS GPUs per node"}'}]}
        else:
            yield 'Four GPUs per node.'

    monkeypatch.setattr(assistant.Assistant, '_stream', fake_stream)
    bot = assistant.Assistant(_indexes(tmp_path))
    out = StringIO()
    answer, sources = bot.ask('And what about the other system I mentioned?', out)
    assert answer == 'Four GPUs per node.'
    assert calls[1][-1]['role'] == 'tool' and 'gres=gpu:4' in calls[1][-1]['content']
    assert 'https://jsc/juwels' in sources and '[searched: JUWELS GPUs per node]' in out.getvalue()
    bot.ask('Thanks, and on LUMI?', StringIO())
    roles = [m['role'] for m in calls[-1]]
    assert roles[:2] == ['system', 'user'] and roles.count('user') == 2  # earlier turn kept in the chat


def test_generic_question_falls_back_to_any_site(tmp_path):
    bot = assistant.Assistant(_indexes(tmp_path))
    # Routed to the generic Slurm docs, which have nothing on Leonardo's booster: the fallback finds it anyway.
    docs = bot.search('What does the booster partition boost_usr_prod offer?')
    assert docs and docs[0]['url'] == 'https://cineca/leo'
    # A routed search that already fills k slots is left exactly as in training and evaluation.
    p, f = tmp_path / 'documents.sqlite', tmp_path / 'factory_docs.sqlite'
    q = 'Set a time limit with sbatch'
    assert bot.search(q, k=1) == assistant.retrieve(q, p, f, 1)


def test_setup_explains_missing_ollama(tmp_path, monkeypatch):
    monkeypatch.setattr(assistant.shutil, 'which', lambda name: None)
    try:
        assistant.setup(tmp_path)
    except SystemExit as e:
        assert 'ollama.com' in str(e)
    else:
        raise AssertionError('setup must stop when Ollama is missing')


def test_scanned_pdf_page_without_ocr_tools_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(assistant.shutil, 'which', lambda name: None)
    assert assistant._ocr_page(tmp_path / 'scan.pdf', 1) == ''
