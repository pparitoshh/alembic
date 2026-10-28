"""hpc-assistant: built-in retrieval must give the student the exact prompts it was evaluated with."""
import importlib.util
import json
import sqlite3
import subprocess
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


def test_improved_retriever_matches_eval_retriever(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_VARIANT', 'b')
    spec = importlib.util.spec_from_file_location('rag65_retrieve_b', ROOT / 'research/rag65/retrieve.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    idx = _indexes(tmp_path)
    p, f = idx / 'documents.sqlite', idx / 'factory_docs.sqlite'
    for q in ['How do I request GPUs on LUMI?', 'Set a time limit with sbatch', 'GPUs per node on Leonardo booster?',
              'How many GPUs per node on JUWELS?', 'How do I cancel a pending array job?', 'the and for']:
        assert assistant.retrieve(q, p, f, variant='b') == ref.retrieve(q, p, f)
    for name in ['VARIANT_K', 'VARIANT_GENERIC', 'VARIANT_BUDGET', 'VARIANT_DUP', 'COMMANDS', 'INTENT']:
        assert getattr(assistant, name) == getattr(ref, name), name


def test_relevance_header_matches_eval_variant_e(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_VARIANT', 'e')
    spec = importlib.util.spec_from_file_location('rag65_retrieve_e', ROOT / 'research/rag65/retrieve.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    idx = _indexes(tmp_path)
    p, f = idx / 'documents.sqlite', idx / 'factory_docs.sqlite'
    for q in ['How do I request GPUs on LUMI?', 'How do I cancel a pending array job?', 'the and for']:
        docs = assistant.retrieve(q, p, f, variant='e')
        assert docs == assistant.retrieve(q, p, f, variant='b') == ref.retrieve(q, p, f)
        assert assistant.with_context(q, docs, relevance=True) == ref.with_context(q, ref.retrieve(q, p, f))


def _general_index(idx):
    _index(idx / 'general_docs.sqlite', [
        ('https://docs.nvidia/nsight', 'NVIDIA', 'Profile a CUDA kernel with Nsight Compute: ncu --set full ./app.'),
        ('https://openmp/offload', 'OpenMP', 'OpenMP target offload moves loops to the GPU with omp target teams.')], 'site')


def test_general_retrievers_match_eval_retriever(tmp_path, monkeypatch):
    idx = _indexes(tmp_path)
    _general_index(idx)
    p, f = idx / 'documents.sqlite', idx / 'factory_docs.sqlite'
    qs = ['How do I profile a CUDA kernel with Nsight?', 'How do I cancel a pending Slurm array job?',
          'GPU offload with OpenMP on LUMI?', 'How do I request GPUs on LUMI?', 'the and for']
    for v in ['c', 'd', 'f']:
        monkeypatch.setenv('RAG_VARIANT', v)
        spec = importlib.util.spec_from_file_location('rag65_retrieve_' + v, ROOT / 'research/rag65/retrieve.py')
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        for q in qs:
            assert assistant.retrieve(q, p, f, variant=v) == ref.retrieve(q, p, f), (v, q)
    assert assistant.VARIANT_GENERAL == ref.VARIANT_GENERAL
    # d adds general excerpts only to questions that do not name Slurm, a Slurm command or LUMI/CINECA
    assert any('nvidia' in d['url'] for d in assistant.retrieve(qs[0], p, f, variant='d'))
    assert not any(d['url'].startswith(('https://docs.nvidia', 'https://openmp'))
                   for d in assistant.retrieve(qs[2], p, f, variant='d'))


def test_retriever_d_uses_general_index_and_falls_back_without_it(tmp_path):
    idx = _indexes(tmp_path)
    q = 'How do I profile a CUDA kernel with Nsight?'
    assert assistant.Assistant(idx, retriever='d').retriever == 'b'  # no general index yet
    _general_index(idx)
    bot = assistant.Assistant(idx, retriever='d')
    assert bot.retriever == 'd' and bot.search(q) == assistant.retrieve(q, idx / 'documents.sqlite',
                                                                         idx / 'factory_docs.sqlite', variant='d')


def test_retriever_f_is_the_default_with_a_general_index_and_falls_back_to_e(tmp_path):
    idx = _indexes(tmp_path)
    p, f = idx / 'documents.sqlite', idx / 'factory_docs.sqlite'
    q = 'How do I profile a CUDA kernel with Nsight?'
    assert assistant.Assistant(idx, retriever='f').retriever == 'e'  # no general index yet
    assert assistant.Assistant(idx).retriever == 'e'
    _general_index(idx)
    bot = assistant.Assistant(idx)
    assert bot.retriever == 'f' and bot.search(q) == assistant.retrieve(q, p, f, variant='d')
    assert bot.prompt(q)[0] == assistant.with_context(q, bot.search(q), relevance=True)
    assert assistant.Assistant(idx, retriever='e').retriever == 'e'  # an explicit choice is kept
    assert assistant.Assistant(idx, retriever='b').retriever == 'b'


def test_improved_retriever_is_default_and_asks_ollama_for_16k_context(tmp_path, monkeypatch):
    idx = _indexes(tmp_path)
    p, f = idx / 'documents.sqlite', idx / 'factory_docs.sqlite'
    q = 'How do I request GPUs on LUMI?'
    bot = assistant.Assistant(idx)
    assert bot.retriever == 'e' and bot.search(q) == assistant.retrieve(q, p, f, variant='b')
    assert bot.prompt(q)[0].startswith(q + assistant.RELEVANCE_HEADER)
    assert assistant.Assistant(idx, retriever='b').prompt(q)[0] == assistant.with_context(q, bot.search(q))
    sent = []

    class Resp:  # stands in for urlopen's response: one Ollama chat chunk
        def __enter__(self):
            return iter([json.dumps({'message': {'content': 'ok'}}).encode()])

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(assistant.urllib.request, 'urlopen', lambda req, timeout: sent.append(json.loads(req.data)) or Resp())
    assistant.Assistant(idx).ask(q, StringIO())
    assistant.Assistant(idx, retriever='classic').ask(q, StringIO())
    assert sent[0]['options'] == {'temperature': 0, 'num_ctx': 16384} and sent[1]['options'] == {'temperature': 0}


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
    bot = assistant.Assistant(_indexes(tmp_path), retriever='classic')
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
    bot = assistant.Assistant(_indexes(tmp_path), retriever='classic')
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


def test_web_upload_names_are_safe_and_indexed(tmp_path):
    from distillkit import assistant_web as web
    assert web.safe_name('../../etc/passwd.md') == 'passwd.md'
    assert web.safe_name('C:\\Users\\me\\Cluster Guide (v2).pdf') == 'Cluster Guide _v2_.pdf'
    for bad in ['script.sh', 'archive.tar.gz', 'noext']:
        try:
            web.safe_name(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad} must be rejected')
    bot = assistant.Assistant(_indexes(tmp_path))
    text = ('# MyCluster\n\nSubmit GPU jobs to the partition gpu-long with --gres=gpu:a100:2 and a maximum wall time of '
            '48 hours on MyCluster.').encode()
    name, n = web.save_upload(bot, '../guide.md', text)
    assert name == 'guide.md' and n == 1 and (tmp_path / 'uploads' / 'guide.md').read_bytes() == text
    assert web.user_documents(bot) == [('guide.md', 1)]
    assert bot.search('GPU partition on MyCluster')[0]['url'] == 'guide.md'


def test_web_failed_question_leaves_history_clean(tmp_path, monkeypatch):
    from distillkit import assistant_web as web

    def down(self, messages):
        raise OSError('connection refused')
        yield  # pragma: no cover

    monkeypatch.setattr(assistant.Assistant, '_stream', down)
    bot = assistant.Assistant(_indexes(tmp_path))
    handler = web.make_handler(bot)

    class Fake(handler):
        def __init__(self, body):
            from io import BytesIO
            self.path, self.rfile, self.wfile = '/api/ask', BytesIO(body), BytesIO()
            self.headers = {'Content-Length': str(len(body))}
            self.request_version, self.requestline, self.command = 'HTTP/1.1', 'POST /api/ask HTTP/1.1', 'POST'

    req = Fake(b'{"question": "How do I request GPUs on LUMI?"}')
    req.do_POST()
    assert b'not reachable' in req.wfile.getvalue()
    assert [m['role'] for m in bot.history] == ['system']


def test_setup_installs_then_updates_only_for_a_new_release(tmp_path, monkeypatch):
    monkeypatch.setattr(assistant.shutil, 'which', lambda name: '/usr/bin/ollama')
    models, created, fetched = [''], [], []

    def run(cmd, **kw):
        if cmd[:2] == ['ollama', 'create']:
            created.append((cmd[2], (kw['cwd'] / assistant.GGUF_FILE).read_text()))
            models[0] = cmd[2] + ':latest  abc  2.5 GB  now\n'
        return subprocess.CompletedProcess(cmd, 0, 'NAME ID SIZE MODIFIED\n' + models[0], '')

    def download(url, dest):
        if dest.suffix != '.sqlite':  # the docs index is fetched once, on the first setup
            fetched.append(dest.name)
        else:
            assert url == assistant.HF_INDEX + dest.name
        dest.write_text(release[0])
    release = ['v1']
    monkeypatch.setattr(assistant.subprocess, 'run', run)
    monkeypatch.setattr(assistant, '_download', download)
    monkeypatch.setattr(assistant, '_release_id', lambda: release[0])
    assistant.setup(tmp_path, 'cp')
    assert created == [('cp', 'v1')] and sorted(fetched) == sorted([assistant.GGUF_FILE, 'Modelfile'])
    assistant.setup(tmp_path, 'cp')  # nothing new: nothing downloaded or created
    assert len(created) == 1 and len(fetched) == 2
    release[0] = 'v2'
    assistant.setup(tmp_path, 'cp')  # a new release replaces the cached files and the Ollama model
    assert created[-1] == ('cp', 'v2') and len(fetched) == 4
    monkeypatch.setattr(assistant, '_release_id', lambda: None)
    assistant.setup(tmp_path, 'cp')  # offline: the installed model stays
    assert len(created) == 2


def test_notes_are_added_to_the_index_once_and_retrieved(tmp_path):
    spec = importlib.util.spec_from_file_location('add_notes', Path(__file__).resolve().parent.parent / 'scripts/add_notes_to_index.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    index = _indexes(tmp_path)
    for _ in range(2):  # running it again replaces the notes instead of adding copies
        assert mod.add_notes(index / 'documents.sqlite', Path(__file__).resolve().parent.parent / 'data/clusterpilot_notes') == 3
    docs = assistant.retrieve('requeue a job with a checkpoint before the time limit', index / 'documents.sqlite',
                              index / 'factory_docs.sqlite', variant='e')
    assert docs[0]['url'].endswith('notes/requeue_checkpoint.md') and 'B:USR1' in docs[0]['text']
