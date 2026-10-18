"""Build the sharded CC-BY AI Factory generation batch (LUMI + CSC). No generation here.

Inputs are already-admitted sources only: the 48-page LUMI pilot registry and the
admission_v4 CSC/LUMI pool. Development-split and held-out documents never enter.
"""
from collections import Counter
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
ADMISSION = Path(sys.argv[1]).resolve()  # campaigns/.../next_5000/admission_v4
SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PILOT = REPO / 'runs/ai_factory_pilot_20261013'
OUT = REPO / 'runs/ai_factory_ccby_v2'
ROUND = 'ai-factory-ccby-2'
ANGLES = ['a common mistake or failure and how to diagnose it', 'how to verify that it worked', 'adapting it to a different job size or resource request', 'the reasoning behind the recommended setting']
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

pilot_registry = json.loads((PILOT / 'source_registry.json').read_text())
admission_sha = sha((ADMISSION / 'summary.json').read_bytes())
registry = {'version': 'source-family-registry-v1', 'families': dict(pilot_registry['families']), 'documents': []}
docs = []
for d in pilot_registry['documents']:
    text = (PILOT / 'seeds' / d['path']).read_text()
    assert sha(text) == d['sha256'], d['doc_id']
    shutil.copyfile(PILOT / 'seeds' / d['path'], OUT / 'seeds' / d['path'])
    registry['documents'].append(d)
    docs.append({'doc_id': d['doc_id'], 'site': 'LUMI', 'url': d['origin'], 'title': d['doc_id'], 'text': text,
                 'sha256': d['sha256']})
for line in (ADMISSION / 'documents.jsonl').read_text().splitlines():
    d = json.loads(line)
    registry['families'][d['family']] = {'split': d['split'], 'parents': [], 'lineage_reviewed': True}
    if d['split'] != 'train':
        continue
    text = (ADMISSION / d['path']).read_text()
    assert sha(text) == d['sha256'], d['doc_id']
    (OUT / 'seeds' / (d['doc_id'] + '.md')).write_text(text)
    registry['documents'].append({'doc_id': d['doc_id'], 'path': d['doc_id'] + '.md', 'sha256': d['sha256'],
                                  'family': d['family'], 'source_version': d['version'],
                                  'approved_for_training': True, 'origin': d['url'],
                                  'permission_evidence': d['permission']['license'] + '; ' + d['permission']['url'],
                                  'admission_review_sha256': admission_sha})
    docs.append({'doc_id': d['doc_id'], 'site': d['operator'], 'url': d['url'], 'title': d['title'], 'text': text,
                 'sha256': d['sha256']})
dump(OUT / 'source_registry.json', registry)

SITE = {'LUMI': 'LUMI', 'CSC': "CSC's supercomputers (Puhti, Mahti or Roihu, whichever the excerpt covers)"}
TASKS = ['howto', 'debug', 'concept', 'script']
SLURM = re.compile(r'\b(sbatch|squeue|sacct|srun|salloc|scancel|Slurm)\b')
prose, tool_sources = [], []
for d in docs:
    for i, text in enumerate(chunk_text(d['text'], 2200)):
        if not 650 <= len(text) <= 4500:
            continue
        options = TASKS if '```' in text else ['howto', 'concept', 'debug']
        start = int(sha(d['doc_id'] + str(i))[:4], 16)
        for v in range(3):
            prose.append((d, i, text, options[(start + v + 1) % len(options)], (start + v + 2) % 4, ANGLES[(start + v) % 4]))
        if SLURM.search(text) and re.search(r'/runjobs/|/computing/running/', d['url']):
            tool_sources.append((d, i, text))

entries = []
for n, (d, i, text, task, persona, angle) in enumerate(prose):
    site = SITE[d['site']]
    entries.append({'scenario_id': 'ctx-%05d' % n, 'round_id': ROUND,
        'doc_id': d['doc_id'], 'chunk_id': d['doc_id'] + '#' + str(i),
        'document_sha256': d['sha256'], 'chunk_sha256': sha(text), 'persona_index': persona,
        'task': task, 'mode': 'prose', 'capability': d['site'] + ' source-backed ' + task,
        'brief': 'For ' + site + ' specifically, ask a practical ' + task + ' question focused on ' + angle + ', about ' + d['title'] +
                 ', excerpt ' + str(i + 1) + '. Name the site in the question. The attached excerpt is the only '
                 'source for site settings. Ask for a full explanation needed to act correctly, including '
                 'prerequisites, the reasoning behind each step and checks, or uncertainty when applicable; '
                 'do not ask to memorize an absent source.',
        'context_spans': [{'start': 0, 'end': len(text), 'sha256': sha(text)}]})

# ~30% tool-decision traces; mock-cluster observations, never claimed to be the real site scheduler.
MODES = ['call', 'none', 'call', 'ask', 'none', 'call', 'ask', 'none', 'call', 'none']
tool_count = min(round(len(entries) * 3 / 7), len(tool_sources) * 20)
for n in range(tool_count):
    d, i, text = tool_sources[n % len(tool_sources)]
    variant = n // len(tool_sources) + 10
    mode = MODES[(n + variant) % 10]
    task = ['howto', 'debug'][variant % 2]
    entries.append({'scenario_id': 'tool-%05d' % n, 'round_id': ROUND,
        'doc_id': d['doc_id'], 'chunk_id': d['doc_id'] + '#' + str(i),
        'document_sha256': d['sha256'], 'chunk_sha256': sha(text), 'persona_index': variant % 4,
        'task': task, 'mode': mode, 'capability': 'Slurm tool decision ' + mode,
        'brief': 'Create a ' + task + ' request in mode ' + mode + ' concerning the Slurm workflow in ' +
                 d['title'] + ', excerpt ' + str(i + 1) + ', scenario ' + str(variant + 1) +
                 '. Site-independent tool observations come exclusively from the supplied mock cluster, not from '
                 'the real ' + d['site'] + ' scheduler. Use a different practical user objective for each scenario. '
                 'Never invent a workload, real queue state, missing required job ID, or undocumented flag.'})

base = json.loads((REPO / 'configs/ai_factory_pilot.json').read_text())
tool_ids = sorted({d['doc_id'] for d, _, _ in tool_sources})
# Round-robin shards fit the 30-minute debug QoS; the merge step deduplicates across shards.
for k in range(SHARDS):
    part = entries[k::SHARDS]
    run = 'runs/ai_factory_ccby_v2/shard_%d' % k
    dump(REPO / run / 'scenario_plan.json', {'version': 'source-scenario-plan-v2', 'entries': part})
    cfg = json.loads(json.dumps(base))
    cfg['run_dir'] = run
    cfg['seeds']['dir'] = 'runs/ai_factory_ccby_v2/seeds'
    cfg['seeds']['registry'] = 'runs/ai_factory_ccby_v2/source_registry.json'
    cfg['generate']['scenario_plan'] = run + '/scenario_plan.json'
    cfg['generate']['tool_doc_ids'] = tool_ids
    cfg['teacher']['max_tokens'] = 2048
    cfg['verify']['grounding_reviews'] = run + '/grounding_reviews.jsonl'
    dump(REPO / ('configs/ai_factory_ccby_v2_shard_%d.json' % k), cfg)

dump(OUT / 'summary.json', {
    'created_at': datetime.now(timezone.utc).isoformat(), 'round_id': ROUND,
    'licence_scope': 'CC-BY-4.0 only (LUMI docs/training, CSC user guide); all-factory coverage is RAG-only',
    'training_documents': len(docs), 'sites': dict(Counter(d['site'] for d in docs)),
    'planned_questions': len(entries), 'modes': dict(Counter(e['mode'] for e in entries)),
    'shards': SHARDS, 'tool_source_chunks': len(tool_sources),
    'admission_summary_sha256': admission_sha, 'generated_examples': 0})
print((OUT / 'summary.json').read_text())
