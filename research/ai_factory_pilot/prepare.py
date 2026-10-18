"""Build an isolated, source-bound pilot from the downloaded archive. No generation here."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit, urljoin

from bs4 import BeautifulSoup
from markdownify import markdownify

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path('/Users/vaibhavmangroliya/Downloads/EU_AI_Factory_Documentation_2026-10-13')
OUT = REPO / 'runs/ai_factory_pilot_20261013'
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'generation_manifest.json').exists():
    raise SystemExit('Generation has started; use a new immutable preparation directory.')
for name in ['seeds', 'heldout_sources', 'preparation']:
    (OUT / name).mkdir(exist_ok=True)


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


# Reuse the production chunker without importing GPU/client dependencies in the extraction venv.
module = ast.parse((REPO / 'src/distillkit/seeds.py').read_text())
function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'chunk_text')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'production_chunk_text', 'exec'), namespace)
chunk_text = namespace['chunk_text']
held_pattern = re.compile(json.loads((REPO / 'data/sources/hpc10k_sources.json').read_text())['heldout_pattern'])
holdouts = []
for path in (REPO / 'data/eval').rglob('*.holdout.json'):
    holdouts.extend(json.loads(path.read_text())['sources'])
if not any(p.name == 'eval_v3.holdout.json' for p in (REPO / 'data/eval').glob('*.holdout.json')):
    raise RuntimeError('Missing frozen v3 source holdout manifest.')


def shingles(text):
    words = re.findall(r'\w+', text.casefold())
    return {tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


held_shingles = [shingles(h['text']) for h in holdouts if h.get('text')]


def overlaps_held(text):
    s = shingles(text)
    return len(s) >= 8 and any(len(s & h) / min(len(s), len(h)) >= 0.7
                               for h in held_shingles if len(h) >= 8)


def extract(record):
    raw = (ARCHIVE / record['original_path']).read_bytes()
    if sha(raw) != record['sha256']:
        raise RuntimeError('Original checksum mismatch: ' + record['url'])
    soup = BeautifulSoup(raw, 'lxml')
    main = soup.select_one('article.md-content__inner')
    if main is None:
        raise RuntimeError('Expected LUMI article container absent: ' + record['url'])
    for tag in main.select('script, style, .headerlink, .md-content__button, .linenos, .md-source-file'):
        tag.decompose()
    code = []
    for pre in main.select('pre'):
        block = pre.get_text().strip()
        if not block:
            pre.decompose()
            continue
        token = 'DKCODEBLOCK%05dTOKEN' % len(code)
        code.append(block)
        replacement = soup.new_tag('p')
        replacement.string = token
        pre.replace_with(replacement)
    for link in main.select('a[href]'):
        link['href'] = urljoin(record.get('resolved_url') or record['url'], link['href'])
    text = markdownify(str(main), heading_style='ATX', bullets='-', strip=['img'])
    for i, block in enumerate(code):
        fence = '`' * max(3, 1 + max([0] + [len(m.group()) for m in re.finditer(r'`+', block)]))
        token = 'DKCODEBLOCK%05dTOKEN' % i
        if token not in text:
            raise RuntimeError('Code placeholder lost during extraction: ' + record['url'])
        text = re.sub(r'(?m)^[ \t]*' + token + r'[ \t]*$', lambda m: fence + '\n' + block + '\n' + fence, text)
    # A hard extraction check: every saved code block must survive, including newlines.
    for block in code:
        if block not in text:
            raise RuntimeError('Command block lost during extraction: ' + record['url'])
    return text.strip() + '\n', len(code)


db = sqlite3.connect('file:' + str(ARCHIVE / 'catalog.sqlite') + '?mode=ro', uri=True)
records = [json.loads(x[0]) for x in db.execute('SELECT metadata FROM records ORDER BY url')]
db.close()
inventory = []
seen_text = set()
for r in records:
    if r['status'] != 'downloaded':
        route = 'fetch_gap'
    elif not r.get('text_path'):
        route = 'preserve_original_text_extraction_or_transcription_pending'
    elif r.get('text_sha256') in seen_text:
        route = 'exact_text_duplicate'
    else:
        route = 'needs_topic_version_permission_and_content_review'
        seen_text.add(r.get('text_sha256'))
    inventory.append({k: r.get(k) for k in ['url', 'source_id', 'factory_ids', 'content_type',
                      'original_path', 'text_path', 'sha256', 'text_sha256', 'bytes']} |
                     {'route': route, 'training_approved': False})
(OUT / 'preparation/archive_routing.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in inventory))

# First bounded experiment: one clearly attributed, licensed operator corpus.
# The full archive remains available for later source-by-source admission; this is not all-factory training.
prefixes = ['/runjobs/scheduled-jobs/', '/runjobs/lumi_env/', '/software/installing/',
            '/software/containers/', '/laif/software/', '/storage/', '/firststeps/',
            '/development/compiling/', '/development/debugging/', '/development/profiling/']
docs, exclusions, signatures, canonical_urls = [], [], set(), set()
for r in records:
    if r.get('source_id') != 'lumi_docs' or r['status'] != 'downloaded' or not r.get('text_path'):
        continue
    path = urlsplit(r['url']).path
    if not any(path.startswith(p) for p in prefixes):
        continue
    if not any('creativecommons.org/licenses/by/4.0' in u for u in r.get('license_links', [])):
        exclusions.append({'url': r['url'], 'reason': 'no_archived_CC_BY_4_evidence'})
        continue
    canonical_url = (r.get('resolved_url') or r['url']).rstrip('/')
    if canonical_url in canonical_urls:
        exclusions.append({'url': r['url'], 'reason': 'canonical_URL_alias'})
        continue
    text, code_count = extract(r)
    blocks = chunk_text(text, 1)
    kept = [b for b in blocks if not held_pattern.search(b) and not overlaps_held(b)]
    clean = '\n\n'.join(kept).strip() + '\n'
    if len(clean) < 800 or sha(clean) in signatures:
        exclusions.append({'url': r['url'], 'reason': 'short_or_duplicate'})
        continue
    signatures.add(sha(clean))
    canonical_urls.add(canonical_url)
    doc_id = 'lumi_' + sha(canonical_url)[:16]
    reserved = int(sha(canonical_url)[:8], 16) % 10 == 0
    destination = OUT / ('heldout_sources' if reserved else 'seeds') / (doc_id + '.md')
    destination.write_text(clean)
    docs.append({'doc_id': doc_id, 'title': r.get('title'), 'url': r['url'],
                 'path': destination.name, 'sha256': sha(clean), 'family': doc_id,
                 'original_sha256': r['sha256'], 'original_path': r['original_path'],
                 'source_version': 'public HTML snapshot ' + r['retrieved_at'],
                 'license': 'CC-BY-4.0', 'permission_evidence': r['license_links'],
                 'raw_code_blocks_preserved_by_extractor': code_count,
                 'excluded_blocks': len(blocks) - len(kept), 'heldout': reserved})

admission = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'reviewer': 'Codex automated source admission; no human or generated-answer review claimed',
    'scope': 'LUMI technical/AI-factory operational manuals only; first bounded pilot',
    'license_evidence': 'CC-BY-4.0 links saved in each original page; original attribution retained',
    'extraction': 'Article-only Markdown conversion with exact preformatted-code preservation assertion',
    'heldouts': 'Frozen v2 topic regex; v3 five-word containment screen; deterministic whole-page pilot holdout',
    'limits': 'Lexical overlap checks do not establish absence of semantic overlap. Pilot is not release approval.',
    'documents': docs, 'excluded_sources': exclusions,
}
dump(OUT / 'preparation/source_admission.json', admission)
admission_sha = sha((OUT / 'preparation/source_admission.json').read_bytes())
registry = {'version': 'source-family-registry-v1', 'families': {}, 'documents': []}
for d in docs:
    registry['families'][d['family']] = {'split': 'development' if d['heldout'] else 'train',
                                       'parents': [], 'lineage_reviewed': True}
    if d['heldout']:
        continue
    registry['documents'].append({k: d[k] for k in ['doc_id', 'path', 'sha256', 'family', 'source_version']} |
        {'approved_for_training': True, 'origin': d['url'],
         'permission_evidence': 'CC-BY-4.0; ' + '; '.join(d['permission_evidence']),
         'admission_review_sha256': admission_sha})
dump(OUT / 'source_registry.json', registry)
dump(OUT / 'preparation/pilot_source_holdout.json', [d for d in docs if d['heldout']])

chunks = []
for d in docs:
    if not d['heldout']:
        for i, text in enumerate(chunk_text((OUT / 'seeds' / d['path']).read_text(), 2200)):
            if 650 <= len(text) <= 4500 and not overlaps_held(text):
                chunks.append((d, i, text))
# Round-robin across documents so long manuals cannot dominate the pilot.
chunks.sort(key=lambda c: (c[1], c[0]['doc_id']))
chunks = chunks[:180]
entries = []
for n, (d, i, text) in enumerate(chunks):
    # One prose question per excerpt: repeating a long attached source would dominate
    # the production lexical dedup score and waste review on false duplicate pairs.
    tasks = [('howto', 'debug')[n % 2] if '```' in text else ('howto', 'concept')[n % 2]]
    for variant, task in enumerate(tasks):
        entries.append({'scenario_id': 'context-%03d-%d' % (n, variant), 'round_id': 'ai-factory-pilot-1',
            'doc_id': d['doc_id'], 'chunk_id': d['doc_id'] + '#' + str(i),
            'document_sha256': d['sha256'], 'chunk_sha256': sha(text), 'persona_index': n % 4,
            'task': task, 'mode': 'prose', 'capability': 'LUMI source-backed ' + task,
            'brief': 'For LUMI specifically, ask a practical ' + task + ' question about ' + str(d['title']) +
                     ', excerpt ' + str(i + 1) + '. Name LUMI in the question. The attached excerpt is the '
                     'only source for site settings. Request the explanation needed to act correctly, '
                     'including prerequisites or uncertainty when applicable; do not ask to memorize an absent source.',
            'context_spans': [{'start': 0, 'end': len(text), 'sha256': sha(text)}]})

# Tool questions use operational Slurm sections; live results are explicitly the mock contract,
# never asserted to describe the real LUMI scheduler. Independent review and replay are mandatory.
tool_chunks = [(d, i, t) for d, i, t in chunks
               if '/runjobs/scheduled-jobs/' in d['url'] and re.search(r'\b(sbatch|squeue|sacct|srun|Slurm)\b', t)]
tool_count = min(round(len(entries) * 3 / 7), len(tool_chunks) * 8)
for n in range(tool_count):
    d, i, text = tool_chunks[n % len(tool_chunks)]
    variant = n // len(tool_chunks)
    mode = ['call', 'none', 'call', 'ask', 'none', 'call', 'ask', 'none', 'call', 'none'][n % 10]
    task = ['howto', 'debug'][variant % 2]
    entries.append({'scenario_id': 'tool-%03d' % n, 'round_id': 'ai-factory-pilot-1',
        'doc_id': d['doc_id'], 'chunk_id': d['doc_id'] + '#' + str(i),
        'document_sha256': d['sha256'], 'chunk_sha256': sha(text), 'persona_index': variant % 4,
        'task': task, 'mode': mode, 'capability': 'Slurm tool decision ' + mode,
        'brief': 'Create a ' + task + ' request in mode ' + mode + ' concerning the Slurm workflow in '
                 + str(d['title']) + ', excerpt ' + str(i + 1) + ', scenario ' + str(variant + 1) +
                 '. Site-independent tool observations come exclusively from the supplied mock cluster, '
                 'not from real LUMI partitions. Use a different practical user objective for each scenario. '
                 'Never invent a workload, real queue state, missing required job ID, or undocumented flag.'})
dump(OUT / 'scenario_plan.json', {'version': 'source-scenario-plan-v2', 'entries': entries})
dump(OUT / 'preparation/summary.json', {
    'archive_resources_accounted_for': len(inventory), 'archive_routes': dict(Counter(x['route'] for x in inventory)),
    'pilot_training_documents': sum(not d['heldout'] for d in docs),
    'pilot_reserved_documents': sum(d['heldout'] for d in docs),
    'planned_questions': len(entries), 'modes': dict(Counter(x['mode'] for x in entries)),
    'reviewed_source_admission_sha256': admission_sha, 'generated_examples': 0,
    'fine_tuning_started': False, 'all_factory_training_claim': False,
})
print((OUT / 'preparation/summary.json').read_text())
