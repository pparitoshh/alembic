"""hpc-assistant: built-in retrieval must give the student the exact prompts it was evaluated with."""
import importlib.util
import io
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
        seen['messages'] = [dict(m) for m in messages]  # as sent (the conversation keeps only the question)
        yield from ['Use ', '--gpus-per-node.']

    monkeypatch.setattr(assistant.Assistant, '_stream', fake_stream)
    bot = assistant.Assistant(_indexes(tmp_path), retriever='classic')
    out = StringIO()
    answer, sources = bot.ask('How do I request GPUs on LUMI?', out)
    assert answer == 'Use --gpus-per-node.' and sources == ['https://docs.lumi/gpu']
    assert seen['messages'][0] == {'role': 'system', 'content': assistant.SYSTEM_PROMPT}
    assert '<quoted_source>' in seen['messages'][1]['content'] and 'standard-g' in seen['messages'][1]['content']
    assert 'Sources:\n- https://docs.lumi/gpu' in out.getvalue()
    assert bot.history[1] == {'role': 'user', 'content': 'How do I request GPUs on LUMI?'}


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
    assert b'cannot reach the model server' in req.wfile.getvalue() and b'start Ollama' in req.wfile.getvalue()
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
    release = ['g1:m1']
    monkeypatch.setattr(assistant.subprocess, 'run', run)
    monkeypatch.setattr(assistant, '_download', download)
    monkeypatch.setattr(assistant, '_release_id', lambda: release[0])
    monkeypatch.setattr(assistant, '_file_ids', lambda url: {})  # index side: offline
    assistant.setup(tmp_path, 'cp')
    assert created == [('cp', 'g1:m1')] and sorted(fetched) == sorted([assistant.GGUF_FILE, 'Modelfile'])
    assistant.setup(tmp_path, 'cp')  # nothing new: nothing downloaded or created
    assert len(created) == 1 and len(fetched) == 2
    release[0] = 'g2:m2'
    assistant.setup(tmp_path, 'cp')  # a new release replaces the cached files and the Ollama model
    assert created[-1] == ('cp', 'g2:m2') and len(fetched) == 4
    release[0] = 'g2:m3'  # only the Modelfile changed: the GGUF is not downloaded again
    assistant.setup(tmp_path, 'cp')
    assert len(created) == 3 and fetched[-1] == 'Modelfile' and len(fetched) == 5
    monkeypatch.setattr(assistant, '_release_id', lambda: None)
    assistant.setup(tmp_path, 'cp')  # offline: the installed model stays
    assert len(created) == 3


def test_notes_are_added_to_the_index_once_and_retrieved(tmp_path):
    spec = importlib.util.spec_from_file_location('add_notes', Path(__file__).resolve().parent.parent / 'scripts/add_notes_to_index.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    index = _indexes(tmp_path)
    for _ in range(2):  # running it again replaces the notes instead of adding copies
        assert mod.add_notes(index / 'documents.sqlite', Path(__file__).resolve().parent.parent / 'data/clusterpilot_notes') == 8
    docs = assistant.retrieve('requeue a job with a checkpoint before the time limit', index / 'documents.sqlite',
                              index / 'factory_docs.sqlite', variant='e')
    assert docs[0]['url'].endswith('notes/requeue_checkpoint.md') and 'B:USR1' in docs[0]['text']


def test_a_looping_answer_is_stopped(tmp_path, monkeypatch):
    bot = assistant.Assistant(tmp_path, docs=False)
    produced = []

    def stream(messages):
        yield 'Options:\n--requeue : Allow requeue.\n'
        for _ in range(500):  # the model repeats two lines until something stops it
            produced.append(1)
            yield '--no-requeue : Do not requeue.\n--no-kill : Do not kill.\n'
    monkeypatch.setattr(bot, '_stream', stream)
    out = StringIO()
    answer, _ = bot.ask('list every sbatch option', out=out)
    assert 'started repeating' in answer and len(produced) < 20
    assert not assistant.Assistant.repeating('#SBATCH --nodes=1\n#SBATCH --ntasks=4\n#SBATCH --time=01:00:00\n' * 2)
    assert not assistant.Assistant.repeating('srun --exact -n 1 ./step &\n' * 5 + 'wait\n')  # a real script


def test_agent_mode_asks_without_excerpts_first(tmp_path, monkeypatch):
    from distillkit.slurm_live import SlurmLive

    def runner(argv, stdin=None):
        return 0, '', ''  # an empty queue
    bot = assistant.Assistant(_indexes(tmp_path), slurm=SlurmLive(runner=runner))
    sent = []

    def stream(messages):  # live question: a tool call, then the final answer
        sent.append(messages[-1]['content'] if messages[-1]['role'] == 'user' else messages[-1]['role'])
        if messages[-1]['role'] == 'user':
            yield {'tool_calls': [{'name': 'list_queue', 'arguments': {}}]}
        else:
            yield 'You have no jobs in the queue.'
    monkeypatch.setattr(bot, '_stream', stream)
    answer, sources = bot.ask('Show my jobs in the queue on Leonardo', out=StringIO())
    assert sent == ['Show my jobs in the queue on Leonardo', 'tool'] and sources == []  # one request, no excerpts
    assert answer == 'You have no jobs in the queue.'

    bot.reset()
    sent.clear()

    def stream2(messages):  # knowledge question: plain text, so it is asked again with the excerpts
        sent.append(messages[-1]['content'])
        yield 'Use --gpus-per-node in the standard-g partition. ' * 10
    monkeypatch.setattr(bot, '_stream', stream2)
    answer, sources = bot.ask('How do I request GPUs on LUMI?', out=StringIO())
    assert sent[0] == 'How do I request GPUs on LUMI?' and 'Quoted retrieval results' in sent[1]
    assert 'https://docs.lumi/gpu' in sources and bot.history[1]['content'] == 'How do I request GPUs on LUMI?'


def test_setup_refreshes_only_an_index_it_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(assistant.shutil, 'which', lambda name: '/usr/bin/ollama')
    monkeypatch.setattr(assistant.subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, 'NAME\ncp:latest x 2.5 GB now\n', ''))
    monkeypatch.setattr(assistant, '_release_id', lambda: None)  # model side: offline, nothing to do
    ids, fetched = {'documents.sqlite': 'd1', 'factory_docs.sqlite': 'f1'}, []
    monkeypatch.setattr(assistant, '_file_ids', lambda url: dict(ids))

    def download(url, dest):
        fetched.append(dest.name)
        dest.write_text(ids[dest.name])
    monkeypatch.setattr(assistant, '_download', download)
    own = tmp_path / 'own'
    own.mkdir()
    (own / 'documents.sqlite').write_text('hand-built')
    (own / 'factory_docs.sqlite').write_text('hand-built')
    assistant.setup(tmp_path, 'cp')  # fresh install: both files
    assert sorted(fetched) == ['documents.sqlite', 'factory_docs.sqlite']
    assistant.setup(tmp_path, 'cp')  # unchanged: nothing
    assert len(fetched) == 2
    ids['documents.sqlite'] = 'd2'
    assistant.setup(tmp_path, 'cp')  # the published documents index changed: only that file
    assert fetched[2:] == ['documents.sqlite'] and (tmp_path / 'documents.sqlite').read_text() == 'd2'
    assistant.setup(own, 'cp')  # a hand-built index (no stamp) is never replaced
    assert len(fetched) == 3 and (own / 'documents.sqlite').read_text() == 'hand-built'


def test_download_replaces_an_existing_file_and_documents_read_as_utf8(tmp_path, monkeypatch):
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(assistant.urllib.request, 'urlopen', lambda url, timeout=0: Resp(b'new index'))
    dest = tmp_path / 'documents.sqlite'
    dest.write_text('old index')
    assistant._download('https://example/documents.sqlite', dest)  # an update: os.replace, not rename
    assert dest.read_text() == 'new index' and not (tmp_path / 'documents.sqlite.part').exists()
    doc = tmp_path / 'guide.md'
    doc.write_bytes('# Guide\n\nUse `--gpus-per-node` – not `--gpus` – on MyCluster → standard-g.'.encode())
    assert '– not `--gpus` – on MyCluster → standard-g' in assistant.read_document(doc)[0][1]  # explicit UTF-8


def test_long_chats_keep_plain_questions_and_drop_the_oldest_turns(tmp_path, monkeypatch):
    sent = []

    def stream(self, messages):
        sent.append(sum(len(m['content']) for m in messages))
        yield 'A' * 3000
    monkeypatch.setattr(assistant.Assistant, '_stream', stream)
    bot = assistant.Assistant(_indexes(tmp_path))
    for i in range(12):
        bot.ask(f'Question {i}: how do I request GPUs on LUMI with sbatch?', out=StringIO())
    assert bot.history[0]['role'] == 'system' and 'Quoted retrieval' not in str(bot.history)
    assert sum(len(m['content']) for m in bot.history[1:]) <= assistant.Assistant.HISTORY_CHARS + 3100
    assert bot.history[-2]['content'].startswith('Question 11') and max(sent) < 40000


def test_community_pages_only_for_their_community():
    page = {'url': 'https://docs.hpc.cineca.it/specific_users/specific_users.html', 'text': 'qos_fualowprio ...'}
    assert not assistant.community_ok(page, 'Which QOS should I use on Leonardo for a long job?')
    assert assistant.community_ok(page, 'I am a EUROfusion user: which QOS do I use on Leonardo?')
    assert assistant.community_ok({'url': 'https://docs.hpc.cineca.it/hpc/leonardo.html', 'text': ''}, 'QOS on Leonardo?')


def test_model_errors_become_messages_and_agent_mode_needs_slurm(tmp_path, monkeypatch):
    import urllib.error

    def missing(self, messages):
        raise urllib.error.HTTPError('http://127.0.0.1:11434/api/chat', 404, 'Not Found', {}, None)
        yield  # pragma: no cover
    monkeypatch.setattr(assistant.Assistant, '_stream', missing)
    bot = assistant.Assistant(_indexes(tmp_path), model='no-such-model')
    out = StringIO()
    assert bot.safe_ask('hi', out) is None and len(bot.history) == 1
    assert "the model 'no-such-model' is not available" in out.getvalue() and 'ollama list' in out.getvalue()
    monkeypatch.setattr(assistant.shutil, 'which', lambda name: None)
    try:
        assistant.main(['--agent', '--index', str(tmp_path), 'show my jobs'])
    except SystemExit as e:
        assert 'no Slurm' in str(e) and '--ssh' in str(e)
    else:
        raise AssertionError('agent mode without Slurm must stop')
