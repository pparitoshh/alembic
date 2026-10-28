"""`hpc-assistant --web`: a local upload-and-chat page for the assistant. Standard library only.

Serves one page on 127.0.0.1 (documents never leave the machine). The user drops manuals (PDF, Markdown, text, HTML;
scanned PDFs are OCRed when tesseract is installed) and asks questions; answers stream in with their sources. The
same Assistant does the work as on the command line, so retrieval and prompts are identical to the evaluated setup.

    hpc-assistant --web              # http://127.0.0.1:8765
    hpc-assistant --web --port 9000
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import uuid

ALLOWED = {'.pdf', '.md', '.markdown', '.txt', '.text', '.html', '.htm', '.rst'}
MAX_UPLOAD = 200 * 1024 * 1024


def safe_name(name):
    """A plain file name for an upload: no directories, only safe characters, an allowed extension."""
    name = Path(name.replace('\\', '/')).name
    name = re.sub(r'[^A-Za-z0-9._ -]', '_', name).strip(' .') or 'document'
    if Path(name).suffix.lower() not in ALLOWED:
        raise ValueError(f'unsupported file type: {Path(name).suffix or "none"} (use PDF, Markdown, text or HTML)')
    return name


def save_upload(bot, name, data):
    """Store an uploaded manual under the index directory and index it; returns (file name, excerpts)."""
    name = safe_name(name)
    folder = bot.user_db.parent / 'uploads'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(data)
    return name, bot.add(path)


def user_documents(bot):
    """[(file, excerpts)] already in the user index."""
    if not bot.user_db.exists():
        return []
    import sqlite3
    db = sqlite3.connect('file:' + str(bot.user_db) + '?mode=ro', uri=True)
    rows = db.execute('SELECT file, COUNT(*) FROM chunks GROUP BY file ORDER BY file').fetchall()
    db.close()
    return rows


class _Stream:
    """File-like object that sends what Assistant.ask writes as HTTP chunks."""

    def __init__(self, wfile):
        self.wfile = wfile

    def write(self, text):
        if text:
            data = text.encode()
            self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')

    def flush(self):
        self.wfile.flush()


def make_handler(bot, approve_in_page=False):
    lock = threading.Lock()  # one conversation, one question at a time
    pending = {}  # agent actions waiting for the Approve/Deny buttons: id -> [threading.Event, approved]
    stream = [None]  # the answer being streamed, where approval requests are written

    def confirm(name, args):
        """Agent mode in the browser: show the action with Approve/Deny buttons and wait for the click."""
        from distillkit.slurm_live import describe
        cid = uuid.uuid4().hex[:12]
        pending[cid] = [threading.Event(), False]
        stream[0].write(f'\n[[approve {cid} {json.dumps(describe(name, args))}]]\n')
        stream[0].flush()
        approved = pending[cid][0].wait(APPROVAL_WAIT) and pending[cid][1]
        del pending[cid]
        return True if approved else 'the user did not approve this action'

    if approve_in_page and bot.slurm is not None:
        bot.slurm.confirm = confirm

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self, limit):
            n = int(self.headers.get('Content-Length') or 0)
            if n > limit:
                raise ValueError('file too large (limit 200 MB)')
            return self.rfile.read(n)

        def do_GET(self):
            if self.path == '/':
                self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
            elif self.path == '/api/docs':
                self._send(200, {'documents': [{'file': f, 'excerpts': n} for f, n in user_documents(bot)],
                                 'builtin': bot.builtin, 'model': bot.model})
            else:
                self._send(404, {'error': 'not found'})

        def do_POST(self):
            try:
                if self.path == '/api/upload':
                    from urllib.parse import unquote
                    data = self._body(MAX_UPLOAD)
                    with lock:
                        name, n = save_upload(bot, unquote(self.headers.get('X-Filename', 'document')), data)
                    self._send(200, {'file': name, 'excerpts': n})
                elif self.path == '/api/approve':  # no lock: the question that asked is holding it
                    req = json.loads(self._body(1 << 10) or b'{}')
                    item = pending.get(req.get('id'))
                    if item is None:
                        self._send(404, {'error': 'no such pending action'})
                        return
                    item[1] = req.get('approve') is True
                    item[0].set()
                    self._send(200, {'ok': True})
                elif self.path == '/api/reset':
                    with lock:
                        bot.reset()
                    self._send(200, {'ok': True})
                elif self.path == '/api/ask':
                    question = json.loads(self._body(1 << 20) or b'{}').get('question', '').strip()
                    if not question:
                        self._send(400, {'error': 'empty question'})
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.send_header('Transfer-Encoding', 'chunked')
                    self.end_headers()
                    out = _Stream(self.wfile)
                    with lock:
                        stream[0] = out
                        try:
                            bot.ask(question, out)
                        except OSError as e:  # model server not reachable
                            if len(bot.history) > 1 and bot.history[-1]['role'] == 'user':
                                bot.history.pop()  # keep the conversation consistent for the next question
                            out.write(f'\n\n[The model is not reachable: {e}. Is Ollama running?]')
                    self.wfile.write(b'0\r\n\r\n')
                else:
                    self._send(404, {'error': 'not found'})
            except ValueError as e:
                self._send(400, {'error': str(e)})

    return Handler


APPROVAL_WAIT = 600  # seconds an action waits for a click before it counts as not approved


def serve(bot, host='127.0.0.1', port=8765, approve_in_page=False):
    server = ThreadingHTTPServer((host, port), make_handler(bot, approve_in_page))
    print(f'[hpc-assistant] web page on http://{host}:{port} (Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ClusterPilot</title>
<style>
:root { --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --line: #e3e2dc; --accent: #2457c5;
        --accent-soft: #e8eefb; --code: #f1f0ec; --user: #eef1f6; --on-accent: #ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161615; --panel: #1f1f1d; --text: #ecebe6; --muted: #9b9a93; --line: #34332f; --accent: #7aa2ff;
          --accent-soft: #23304d; --code: #2a2a27; --user: #262a32; --on-accent: #0f1626; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
.app { max-width: 920px; margin: 0 auto; padding: 20px 16px 120px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 4px 12px; margin-bottom: 16px; }
h1 { font-size: 20px; margin: 0; letter-spacing: -0.01em; white-space: nowrap; }
.meta { color: var(--muted); font-size: 13px; }
.docs { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; margin-bottom: 18px; }
.drop { border: 1.5px dashed var(--line); border-radius: 10px; padding: 16px; text-align: center; color: var(--muted);
        cursor: pointer; transition: border-color .15s, background .15s; }
.drop.over { border-color: var(--accent); background: var(--accent-soft); color: var(--text); }
.drop strong { color: var(--text); }
.files { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.chip { font-size: 12.5px; background: var(--code); border-radius: 999px; padding: 3px 10px; }
.chip.busy { color: var(--muted); }
.chip.err { background: #fde8e8; color: #9b1c1c; }
.chat { display: flex; flex-direction: column; gap: 14px; }
.msg { padding: 12px 14px; border-radius: 12px; overflow-wrap: anywhere; }
.msg.user { background: var(--user); align-self: flex-end; max-width: 85%; white-space: pre-wrap; }
.msg.bot { background: var(--panel); border: 1px solid var(--line); }
.msg.bot pre { background: var(--code); padding: 10px 12px; border-radius: 8px; overflow-x: auto; font-size: 13px; }
.msg.bot code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
.msg.bot :not(pre) > code { background: var(--code); padding: 1px 5px; border-radius: 4px; }
.msg.bot h3, .msg.bot h4 { margin: 12px 0 6px; font-size: 15px; }
.msg.bot p { margin: 6px 0; }
.msg.bot ul { margin: 6px 0; padding-left: 20px; }
.msg.bot blockquote { margin: 8px 0; padding: 6px 12px; border-left: 3px solid var(--accent); background: var(--accent-soft); border-radius: 0 8px 8px 0; }
.sources { margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--line); font-size: 12.5px; color: var(--muted); }
.sources a { color: var(--accent); text-decoration: none; }
.tool { font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); margin: 4px 0; }
.approve { border: 1px solid var(--accent); background: var(--accent-soft); border-radius: 10px; padding: 10px 12px; margin: 8px 0; }
.approve pre { margin: 6px 0; background: transparent; padding: 0; white-space: pre-wrap; }
.approve button { padding: 6px 14px; margin-right: 6px; }
.empty { color: var(--muted); text-align: center; padding: 30px 10px; }
form { position: fixed; left: 0; right: 0; bottom: 0; background: linear-gradient(transparent, var(--bg) 30%); padding: 20px 16px 16px; }
.bar { max-width: 920px; margin: 0 auto; display: flex; gap: 8px; }
textarea { flex: 1; resize: none; min-height: 48px; max-height: 160px; padding: 12px 14px; border-radius: 12px;
           border: 1px solid var(--line); background: var(--panel); color: var(--text); font: inherit; }
textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
button { border: 0; border-radius: 12px; padding: 0 18px; font: inherit; font-weight: 600; cursor: pointer;
         background: var(--accent); color: var(--on-accent); }
button.ghost { background: transparent; color: var(--muted); border: 1px solid var(--line); font-weight: 500; }
button:disabled { opacity: .5; cursor: default; }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>ClusterPilot</h1>
    <span class="meta" id="meta">Runs locally</span>
  </header>
  <section class="docs">
    <div class="drop" id="drop" tabindex="0" role="button" aria-label="Add documents">
      <strong>Drop your cluster docs here</strong> or click to choose · PDF (scans too), Markdown, HTML, text
      <input type="file" id="file" multiple accept=".pdf,.md,.markdown,.txt,.text,.html,.htm,.rst" hidden>
    </div>
    <div class="files" id="files"></div>
  </section>
  <div class="chat" id="chat"><div class="empty" id="empty">Ask about Slurm, GPUs, modules, storage or your own cluster's documents.</div></div>
</div>
<form id="form">
  <div class="bar">
    <button type="button" class="ghost" id="reset" title="Start a new conversation">New</button>
    <textarea id="q" rows="1" placeholder="e.g. How do I request 2 GPUs for 4 hours?"></textarea>
    <button id="send">Ask</button>
  </div>
</form>
<script>
const $ = id => document.getElementById(id);
const excerpts = n => n + (n === 1 ? ' excerpt' : ' excerpts');
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function inline(s) {
  return esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
               .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, '$1<em>$2</em>');
}
function render(md) {
  const out = []; let list = false;
  const parts = md.split(/```[^\n]*\n?/);
  parts.forEach((part, i) => {
    if (i % 2) { out.push('<pre><code>' + esc(part.replace(/\n$/, '')) + '</code></pre>'); return; }
    for (const line of part.split('\n')) {
      const ap = line.match(/^\[\[approve (\w+) (.*)\]\]$/), tool = line.match(/^\[(\w+ \{.*\}|searched: .*)\]$/);
      if (ap) { out.push(approval(ap[1], JSON.parse(ap[2]))); continue; }
      if (tool) { out.push('<div class="tool">⚙ ' + esc(tool[1]) + '</div>'); continue; }
      const h = line.match(/^#{1,4}\s+(.*)/), li = line.match(/^\s*[-*]\s+(.*)/), q = line.match(/^\s*>\s?(.*)/);
      if (li) { if (!list) { out.push('<ul>'); list = true; } out.push('<li>' + inline(li[1]) + '</li>'); continue; }
      if (list) { out.push('</ul>'); list = false; }
      if (h) out.push('<h4>' + inline(h[1]) + '</h4>');
      else if (q) out.push('<blockquote>' + inline(q[1]) + '</blockquote>');
      else if (/^\s*-{3,}\s*$/.test(line)) continue;
      else if (line.trim()) out.push('<p>' + inline(line) + '</p>');
    }
    if (list) { out.push('</ul>'); list = false; }
  });
  return out.join('');
}
const decided = {};  // approval id -> true/false once clicked
function approval(id, text) {
  const state = id in decided ? `<strong>${decided[id] ? 'Approved' : 'Denied'}</strong>`
    : `<button type="button" data-approve="${id}" data-ok="1">Approve</button><button type="button" class="ghost" data-approve="${id}" data-ok="0">Deny</button>`;
  return `<div class="approve">The assistant wants to <pre>${esc(text)}</pre>${state}</div>`;
}
$('chat').addEventListener('click', async e => {
  const b = e.target.closest('button[data-approve]'); if (!b) return;
  const id = b.dataset.approve, ok = b.dataset.ok === '1';
  decided[id] = ok; b.parentElement.innerHTML = `The assistant wants to <pre>${esc(b.parentElement.querySelector('pre').textContent)}</pre><strong>${ok ? 'Approved' : 'Denied'}</strong>`;
  await fetch('/api/approve', {method: 'POST', body: JSON.stringify({id, approve: ok})});
});
function bubble(cls, html) {
  $('empty')?.remove();
  const d = document.createElement('div'); d.className = 'msg ' + cls; d.innerHTML = html;
  $('chat').appendChild(d); d.scrollIntoView({block: 'end'}); return d;
}
function showAnswer(el, text) {
  const i = text.indexOf('\n\nSources:\n');
  let body = i < 0 ? text : text.slice(0, i), src = i < 0 ? '' : text.slice(i + 11);
  let html = render(body);
  if (src.trim()) {
    const items = src.trim().split('\n').map(l => l.replace(/^- /, '')).map(l => {
      const k = l.endsWith(')') ? l.lastIndexOf(' (') : -1;  // "url (title)"; file names may contain spaces
      const url = k < 0 ? l : l.slice(0, k), title = k < 0 ? url : l.slice(k + 2, -1);
      return /^https?:/.test(url) ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(title)}</a>`
                                  : esc(url.replace('#page=', ', page '));  // own files: the name is the title
    });
    html += '<div class="sources">Sources: ' + items.join(' · ') + '</div>';
  }
  el.innerHTML = html;
}
async function ask(q) {
  bubble('user', esc(q));
  const el = bubble('bot', '<span class="meta">Searching the docs and thinking…</span>');
  $('send').disabled = true;
  try {
    const r = await fetch('/api/ask', {method: 'POST', body: JSON.stringify({question: q})});
    const reader = r.body.getReader(), dec = new TextDecoder(); let text = '';
    for (;;) { const {value, done} = await reader.read(); if (done) break; text += dec.decode(value, {stream: true}); showAnswer(el, text); }
  } catch (e) { el.textContent = 'Error: ' + e; }
  $('send').disabled = false; $('q').focus();
}
async function upload(file) {
  const chip = document.createElement('span'); chip.className = 'chip busy'; chip.textContent = file.name + ' · indexing…';
  $('files').appendChild(chip);
  const r = await fetch('/api/upload', {method: 'POST', headers: {'X-Filename': encodeURIComponent(file.name)}, body: file});
  const j = await r.json();
  if (r.ok) { chip.className = 'chip'; chip.textContent = `${j.file} · ${excerpts(j.excerpts)}`; }
  else { chip.className = 'chip err'; chip.textContent = `${file.name}: ${j.error}`; }
}
async function loadDocs() {
  const j = await (await fetch('/api/docs')).json();
  $('meta').textContent = `Runs locally · model ${j.model}` + (j.builtin ? ' · built-in HPC docs on' : ' · no built-in docs index');
  for (const d of j.documents) { const c = document.createElement('span'); c.className = 'chip'; c.textContent = `${d.file} · ${excerpts(d.excerpts)}`; $('files').appendChild(c); }
}
$('drop').onclick = () => $('file').click();
$('drop').onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') $('file').click(); };
$('file').onchange = async e => { for (const f of e.target.files) await upload(f); e.target.value = ''; };
['dragenter', 'dragover'].forEach(t => $('drop').addEventListener(t, e => { e.preventDefault(); $('drop').classList.add('over'); }));
['dragleave', 'drop'].forEach(t => $('drop').addEventListener(t, e => { e.preventDefault(); $('drop').classList.remove('over'); }));
$('drop').addEventListener('drop', async e => { for (const f of e.dataTransfer.files) await upload(f); });
$('form').onsubmit = e => {  // Enter while an answer is streaming must not queue another question
  e.preventDefault(); const q = $('q').value.trim(); if (q && !$('send').disabled) { $('q').value = ''; ask(q); } };
$('q').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('form').requestSubmit(); } };
$('reset').onclick = async () => { await fetch('/api/reset', {method: 'POST'}); $('chat').innerHTML = '<div class="empty" id="empty">New conversation.</div>'; };
loadDocs();
</script>
</body>
</html>
'''
