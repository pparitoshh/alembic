"""Build the sharded all-factory generation batch from the server archive. No generation here.

usage: prepare_factory_v1.py ARCHIVE EXCLUDE_URLS.txt TARGET_QUESTIONS SHARDS
Every European AI Factory (and EuroHPC JU, CINECA) page plus HPC example repositories become sources.
Each source document carries a licence tag (CC-BY-4.0, CC-BY-NC-SA-4.0, a repository licence, or
'unspecified'); the merge copies it into every row so a public release can keep only permissive rows.
Development and evaluation sources (EXCLUDE_URLS, slurm.schedmd.com) never enter.
COVERAGE=1 (BATCH=ai_factory_all_v3) covers what batches 1-2 left out: only chunks they never used, with
a looser relevance bar (score >= 2: still HPC terms or code), so no source chunk is asked about twice.
TOPUP=1 builds the second batch (BATCH=ai_factory_all_v2): same chunks, new question angles and variant
numbers, and an ask-heavy tool mix, since batch 1 showed ask rows verify best and no-tool rows worst.
"""
from collections import Counter
from datetime import datetime, timezone
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

from distillkit.generate import _question_issue

REPO = Path(__file__).resolve().parents[2]
ARCHIVE, EXCLUDE = Path(sys.argv[1]), Path(sys.argv[2])
TARGET, SHARDS = int(sys.argv[3]), int(sys.argv[4])
BATCH = os.environ.get('BATCH', 'ai_factory_all_v1')
TOPUP = os.environ.get('TOPUP') == '1'
COVERAGE = os.environ.get('COVERAGE') == '1'
OUT = REPO / 'runs' / BATCH
ROUND = os.environ.get('ROUND', 'ai-factory-all-1')
if OUT.exists():
    raise SystemExit('Preparation exists; use a new version instead of overwriting')
(OUT / 'seeds').mkdir(parents=True)


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


module = ast.parse((REPO / 'src/distillkit/seeds.py').read_text())
function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'chunk_text')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'production_chunk_text', 'exec'), namespace)
chunk_text = namespace['chunk_text']

factories = {}
with open(ARCHIVE / 'FACTORY_COVERAGE.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        factories[row['factory_id']] = row['name']
excluded = {u.strip().rstrip('/') for u in EXCLUDE.read_text().split() if u.strip()}
SKIP_PATH = ('/news', '/changelog', '/legal', '/privacy', '/cookie', '/events', '/blog', '/press', '/vacanc', '/jobs')
SKIP_KINDS = {'primary_project_profiles', 'primary_project_information'}
CC_BY_SOURCES = {'lumi_docs', 'lumi_training', 'csc_docs'}
NC_SA = ('idris', 'slaif', 'jsc', 'juwels', 'jupiter', 'gwdg')
REPO_FILES = re.compile(r'(readme[^/]*|\.md|\.rst|\.sh|\.slurm|\.sbatch|\.job)$', re.I)
REPO_SKIP = re.compile(r'(license|licence|changelog|contributing|code_of_conduct|authors|security)[^/]*$', re.I)


def licence_of(source_id, url, repo_licence=None):
    if isinstance(repo_licence, dict):  # GitHub licence object
        repo_licence = repo_licence.get('spdx_id') or repo_licence.get('key') or repo_licence.get('name')
    if repo_licence in ('NOASSERTION', 'other'):
        repo_licence = None
    host = urlparse(url).netloc
    if source_id in CC_BY_SOURCES or host.endswith('europa.eu'):
        return 'CC-BY-4.0'  # LUMI/CSC docs; europa.eu content under the Commission reuse decision
    if any(k in source_id for k in NC_SA):
        return 'CC-BY-NC-SA-4.0'
    return repo_licence or 'unspecified'


documents, seen_text, skipped = [], set(), Counter()


def admit(url, text, text_sha, site, source_id, kind, licence):
    if text_sha in seen_text:
        skipped['duplicate_text'] += 1
        return
    seen_text.add(text_sha)
    documents.append({'url': url, 'text': text, 'sha': text_sha, 'site': site, 'source_id': source_id,
                      'kind': kind, 'licence': licence})


for line in open(ARCHIVE / 'manifest.jsonl'):
    r = json.loads(line)
    if r.get('status') != 'downloaded' or not r.get('text_path') or r.get('source_kind') in SKIP_KINDS:
        continue
    url = (r.get('resolved_url') or r['url'])
    host = urlparse(url).netloc
    if (url.rstrip('/') in excluded or host == 'slurm.schedmd.com' or any(s in url.lower() for s in SKIP_PATH)
            or (r.get('language') or 'en') != 'en'):
        skipped['excluded_eval_scope_or_language'] += 1
        continue
    path = ARCHIVE / r['text_path']
    if not path.exists():
        skipped['missing_text'] += 1
        continue
    text = path.read_text(errors='replace')
    if sha(text) != r.get('text_sha256'):
        skipped['text_hash_mismatch'] += 1
        continue
    ids = r.get('factory_ids') or []
    site = ('the EuroHPC JU supercomputers' if host.endswith('europa.eu') else
            '; '.join(factories.get(f, f) for f in ids) or host)
    admit(url, text, r['text_sha256'], site, r['source_id'], r['source_kind'], licence_of(r['source_id'], url))

for line in open(ARCHIVE / 'repository_documents_manifest.jsonl'):
    r = json.loads(line)
    name = r['repository_path']
    if not REPO_FILES.search(name) or REPO_SKIP.search(name) or not r.get('text_path'):
        continue
    path = ARCHIVE / r['text_path']
    if not path.exists() or r.get('words', 0) < 80:
        continue
    text = path.read_text(errors='replace')
    if sha(text) != r.get('text_sha256'):
        skipped['text_hash_mismatch'] += 1
        continue
    admit(r['source_url'], text, r['text_sha256'], 'the HPC site of the ' + r['repository'] + ' repository',
          'repo:' + r['repository'], 'repository_document', licence_of('', r['source_url'], r.get('license')))

# Task relevance for an HPC/AI-factory assistant: operational terms, commands, scripts, or access procedures.
TECH = re.compile(r'\b(sbatch|srun|salloc|squeue|sacct|scancel|sinfo|#SBATCH|module (?:load|avail|spider|use)|'
                  r'partition|--account|--gres|--gpus|--nodes|--ntasks|GPU|CUDA|ROCm|MPI|OpenMP|NCCL|RCCL|'
                  r'container|Singularity|Apptainer|conda|venv|pip install|PyTorch|TensorFlow|JAX|vLLM|'
                  r'scratch|quota|\$HOME|\$WORK|ssh|scp|rsync|login node|compute node|job script|queue|'
                  r'compil\w+|gcc|cmake|Lustre|filesystem|allocation|billing|node-hours|GPU-hours)\b', re.I)
ACCESS = re.compile(r'\b(access|apply|application|call|eligib\w+|allocation|account|project|onboard\w*|'
                    r'support|helpdesk|GPU hours|compute time|free of charge|SME|start-?up)\b', re.I)
SLURM = re.compile(r'\b(sbatch|squeue|sacct|srun|salloc|scancel|Slurm)\b')
TASKS = ['howto', 'debug', 'concept', 'script']
ANGLES = ['the exact steps a user follows', 'a common mistake or failure and how to diagnose it',
          'how to verify that it worked', 'adapting it to a different job size or resource request',
          'the reasoning behind the recommended setting', 'what to check before starting']
if TOPUP:  # disjoint from batch 1 so the two batches do not ask the same thing
    ANGLES = ['interpreting an error message or unexpected output', 'choosing between two documented options',
              'estimating or requesting the right resources', 'a first-time user setting this up from scratch',
              'moving an existing workflow onto this system', 'performance or efficiency of the setup']
VARIANT_BASE = 3 if TOPUP else 0
chunks = []
for d in documents:
    for i, text in enumerate(chunk_text(d['text'], 2200)):
        # The generator rejects excerpts cut inside a code block (unmatched fences).
        if not 650 <= len(text) <= 4500 or _question_issue(text, line_openings=True):
            continue
        tech = len({m.lower() for m in TECH.findall(text)})
        code = '```' in text or bool(re.search(r'^\s*[$#] \w', text, re.M))
        services = d['kind'] in ('factory_services', 'antenna_services', 'official_inventory_and_access')
        score = tech + 3 * code
        if services:
            score = max(score, len({m.lower() for m in ACCESS.findall(text)}))
        if score < (2 if COVERAGE else 4 if not services else 3):
            continue
        chunks.append((score, d, i, text, code, services))
chunks.sort(key=lambda c: (-c[0], c[1]['url'], c[2]))
seen_chunk = set()
unique = []
for c in chunks:
    key = re.sub(r'\s+', ' ', c[3].lower())[:1500]
    if key not in seen_chunk:
        seen_chunk.add(key)
        unique.append(c)
if COVERAGE:  # drop every chunk an earlier batch planned (same chunking, same doc ids, same text keys)
    used_ids, used_text = set(), set()
    for plan in (REPO / 'runs').glob('ai_factory_all_v[12]/shard_*/scenario_plan.json'):
        for e in json.loads(plan.read_text())['entries']:
            used_ids.add(e['chunk_id'])
    for c in unique:
        host = urlparse(c[1]['url']).netloc.split(':')[0].replace('.', '-')
        if 'fac_%s_%s#%d' % (host[:40], c[1]['sha'][:16], c[2]) in used_ids:
            used_text.add(re.sub(r'\s+', ' ', c[3].lower())[:1500])
    before = len(unique)
    unique = [c for c in unique if re.sub(r'\s+', ' ', c[3].lower())[:1500] not in used_text]
    print('coverage: %d relevant chunks, %d already used by batches 1-2, %d new' % (before, before - len(unique), len(unique)))
tool_sources = [c for c in unique if SLURM.search(c[3]) and not c[5]]
prose_budget = round(TARGET * 0.7)
variants = max(1, min(3, -(-prose_budget // max(1, len(unique)))))
selected = unique[:max(1, prose_budget // variants)]

docs_used = {}
for _, d, _, _, _, _ in selected + tool_sources:
    if d['sha'] not in docs_used:
        host = urlparse(d['url']).netloc.split(':')[0].replace('.', '-')
        docs_used[d['sha']] = 'fac_%s_%s' % (host[:40], d['sha'][:16])
registry = {'version': 'source-family-registry-v1', 'families': {}, 'documents': []}
licences, doc_by_id = {}, {}
review_sha = sha(json.dumps({'archive': str(ARCHIVE), 'excluded': sorted(excluded), 'round': ROUND}))
for d in documents:
    doc_id = docs_used.get(d['sha'])
    if not doc_id:
        continue
    family = 'factory:' + d['source_id']
    registry['families'][family] = {'split': 'train', 'parents': [], 'lineage_reviewed': True}
    (OUT / 'seeds' / (doc_id + '.md')).write_text(d['text'])
    assert sha((OUT / 'seeds' / (doc_id + '.md')).read_bytes()) == d['sha']
    registry['documents'].append({'doc_id': doc_id, 'path': doc_id + '.md', 'sha256': d['sha'], 'family': family,
                                  'approved_for_training': True, 'origin': d['url'], 'source_version': ROUND,
                                  'permission_evidence': 'licence tag: ' + d['licence'] + '; user decision 2026-10-17 to train on all factory sources',
                                  'admission_review_sha256': review_sha})
    licences[doc_id] = {'licence': d['licence'], 'url': d['url'], 'site': d['site'], 'source_id': d['source_id']}
    doc_by_id[d['sha']] = doc_id
dump(OUT / 'source_registry.json', registry)
dump(OUT / 'licences.json', licences)

entries = []
for n, (score, d, i, text, code, services) in enumerate(c for c in selected for _ in range(variants)):
    v = n % variants
    doc_id = doc_by_id[d['sha']]
    options = (['concept', 'howto'] if services else TASKS if code else ['howto', 'concept', 'debug'])
    start = int(sha(doc_id + str(i))[:4], 16)
    task, angle = options[(start + v) % len(options)], ANGLES[(start + v) % len(ANGLES)]
    focus = ('how a user gets access, support or resources from this AI Factory' if services else angle)
    entries.append({'scenario_id': 'ctx-%06d' % n, 'round_id': ROUND, 'doc_id': doc_id,
        'chunk_id': doc_id + '#' + str(i), 'document_sha256': d['sha'], 'chunk_sha256': sha(text),
        'persona_index': (start + v + 2) % 4, 'task': task, 'mode': 'prose',
        'capability': 'AI Factory source-backed ' + task,
        'brief': 'For ' + d['site'] + ' specifically (whichever system the excerpt covers), ask a practical ' + task +
                 ' question about excerpt ' + str(i + 1) + ' of ' + d['url'] + ', focused on ' + focus + ' (variant ' + str(VARIANT_BASE + v + 1) + '). Name the site or system in the question. The attached excerpt '
                 'is the only source for site settings. Ask for a full explanation needed to act correctly, including '
                 'prerequisites, the reasoning behind each step and checks, or uncertainty when applicable; do not '
                 'ask to memorize an absent source.',
        'context_spans': [{'start': 0, 'end': len(text), 'sha256': sha(text)}]})

MODES = ['call', 'none', 'call', 'ask', 'none', 'call', 'ask', 'none', 'call', 'none']
tool_count = min(round(len(entries) * 3 / 7), len(tool_sources) * 6)
if TOPUP or COVERAGE:
    MODES = ['ask', 'call', 'ask', 'call', 'ask', 'none', 'ask', 'call', 'ask', 'call']
    tool_count = min(round(len(entries) * 0.25), len(tool_sources) * 3)
for n in range(tool_count):
    _, d, i, text, _, _ = tool_sources[n % len(tool_sources)]
    variant = n // len(tool_sources) + (50 if COVERAGE else 40 if TOPUP else 30)
    mode = MODES[(n + variant) % 10]
    task = ['howto', 'debug'][variant % 2]
    doc_id = doc_by_id[d['sha']]
    entries.append({'scenario_id': 'tool-%06d' % n, 'round_id': ROUND, 'doc_id': doc_id,
        'chunk_id': doc_id + '#' + str(i), 'document_sha256': d['sha'], 'chunk_sha256': sha(text),
        'persona_index': variant % 4, 'task': task, 'mode': mode, 'capability': 'Slurm tool decision ' + mode,
        'brief': 'Create a ' + task + ' request in mode ' + mode + ' concerning the Slurm workflow in excerpt ' + str(i + 1) +
                 ' of ' + d['url'] + ' (' + d['site'] + '), scenario ' + str(variant + 1) + '. Site-independent tool observations come '
                 'exclusively from the supplied mock cluster, not from the real scheduler. Use a different practical '
                 'user objective for each scenario. Never invent a workload, real queue state, missing required job '
                 'ID, or undocumented flag.'})

base = json.loads((REPO / 'configs/ai_factory_pilot.json').read_text())
# Whole documents go to one shard (largest first, onto the lightest shard), and each shard gets its own
# seeds directory and registry holding only its documents. The pipeline re-validates the whole seeds
# directory for every record, so one shared 8,835-file directory made each shard do millions of reads.
# Seeds are hard links to OUT/seeds: same bytes and hashes, no extra space.
per_doc = {}
for e in entries:
    per_doc.setdefault(e['doc_id'], []).append(e)
load, assigned = [0] * SHARDS, [[] for _ in range(SHARDS)]
for doc_id, group in sorted(per_doc.items(), key=lambda x: (-len(x[1]), x[0])):
    k = min(range(SHARDS), key=lambda i: (load[i], i))
    assigned[k].append(doc_id)
    load[k] += len(group)
docs_meta = {d['doc_id']: d for d in registry['documents']}
for k in range(SHARDS):
    run = 'runs/%s/shard_%03d' % (BATCH, k)
    seeds = REPO / run / 'seeds'
    seeds.mkdir(parents=True)
    shard_registry = {'version': 'source-family-registry-v1', 'families': {}, 'documents': []}
    for doc_id in assigned[k]:
        meta = docs_meta[doc_id]
        os.link(OUT / 'seeds' / meta['path'], seeds / meta['path'])
        shard_registry['documents'].append(meta)
        shard_registry['families'][meta['family']] = registry['families'][meta['family']]
    dump(REPO / run / 'source_registry.json', shard_registry)
    part = [e for doc_id in assigned[k] for e in per_doc[doc_id]]
    dump(REPO / run / 'scenario_plan.json', {'version': 'source-scenario-plan-v2', 'entries': part})
    cfg = json.loads(json.dumps(base))
    cfg['run_dir'] = run
    cfg['seeds']['dir'] = run + '/seeds'
    cfg['seeds']['registry'] = run + '/source_registry.json'
    cfg['generate']['scenario_plan'] = run + '/scenario_plan.json'
    cfg['generate']['tool_doc_ids'] = sorted({e['doc_id'] for e in part if e['mode'] != 'prose'})
    cfg['teacher']['max_tokens'] = 2048
    cfg['teacher']['concurrency'] = 48
    cfg['judge']['concurrency'] = 32
    cfg['verify']['grounding_reviews'] = run + '/grounding_reviews.jsonl'
    dump(REPO / ('configs/%s_shard_%03d.json' % (BATCH, k)), cfg)

dump(OUT / 'summary.json', {
    'created_at': datetime.now(timezone.utc).isoformat(), 'round_id': ROUND,
    'licence_scope': 'all factory sources, licence-tagged per document (licences.json)',
    'archive_documents_considered': len(documents), 'relevant_chunks': len(unique),
    'selected_chunks': len(selected), 'variants_per_chunk': variants, 'tool_source_chunks': len(tool_sources),
    'training_documents': len(docs_used), 'planned_questions': len(entries),
    'modes': dict(Counter(e['mode'] for e in entries)), 'shards': SHARDS,
    'questions_per_shard_min_max': [min(load), max(load)],
    'licences_by_document': dict(Counter(v['licence'] for v in licences.values())),
    'planned_by_licence': dict(Counter(licences[e['doc_id']]['licence'] for e in entries)),
    'skipped': dict(skipped), 'generated_examples': 0})
print((OUT / 'summary.json').read_text())
