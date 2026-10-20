"""Build the sharded general-HPC generation batch (ai_factory_all_v5) from the 2026-10-19 archive. No generation here.

usage: [BATCH=ai_factory_all_v5 ROUND=general-hpc-5] prepare_general_v5.py ARCHIVE EXCLUDE_URLS.txt TARGET_QUESTIONS [SHARD_SIZE]
Same output layout and config format as prepare_factory_v1.py (FINAL mode: prose plus ~10% Slurm ask-back, the modes
that verify best), so factory_worker.sh, factory_dbg.sbatch and merge_factory_v1.py run unchanged. Differences:
- sources: the General_HPC_Documentation archive's downloaded, non-duplicate rows (manifest read up to its last
  complete line, since a crawler may still be appending; the prefix used is kept as manifest_snapshot.jsonl);
- excluded: eval URLs (EXCLUDE_URLS, compared without scheme/www/trailing slash/fragment), slurm.schedmd.com,
  '/pgi-compilers/' (old PGI PDFs), news/blog-like paths, non-English pages;
- licence per document from the manifest 'license' field, normalised to an SPDX id when the field is one
  (optionally followed by a parenthetical note); otherwise the stated string, or 'unspecified' when none is stated;
  licences.json also carries topic and a licence_class for annotate_general_v5.py;
- no chunk asked about in batches v1-v4 (or the CC-BY batches): dropped by exact chunk sha (their scenario plans)
  and by the normalised chunk key (their seeds, re-chunked the same way);
- at most 3 variants per chunk; large documents are split over several shards so no shard exceeds SHARD_SIZE.
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
TARGET = int(sys.argv[3])
SHARD_SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 450
BATCH = os.environ.get('BATCH', 'ai_factory_all_v5')
ROUND = os.environ.get('ROUND', 'general-hpc-5')
OUT = REPO / 'runs' / BATCH
EARLIER = ['ai_factory_all_v1', 'ai_factory_all_v2', 'ai_factory_all_v3', 'ai_factory_all_v4',
           'ai_factory_ccby_v1', 'ai_factory_ccby_v2']
if OUT.exists():
    raise SystemExit('Preparation exists; use a new version instead of overwriting')
(OUT / 'seeds').mkdir(parents=True)


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def text_key(text):
    return re.sub(r'\s+', ' ', text.lower())[:1500]


def url_key(url):
    p = urlparse(url.strip())
    host = p.netloc.lower().removeprefix('www.')
    return host + p.path.rstrip('/') + ('?' + p.query if p.query else '')


module = ast.parse((REPO / 'src/distillkit/seeds.py').read_text())
function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'chunk_text')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'production_chunk_text', 'exec'), namespace)
chunk_text = namespace['chunk_text']
CHUNK_CHARS = json.loads((REPO / 'configs/ai_factory_pilot.json').read_text())['seeds']['chunk_chars']

# Names the question can use for the software or system ("Name the site or system in the question").
NAMES = {
    'llnl_hpc_tutorials': 'the LLNL HPC tutorials', 'cornell_cvw': 'the Cornell Virtual Workshop',
    'spack_tutorial': 'the Spack tutorial', 'easybuild_tutorial': 'the EasyBuild tutorial',
    'ulhpc_tutorials': 'the University of Luxembourg HPC tutorials', 'hpc_carpentry_shell_python': 'the HPC Carpentry lessons',
    'hpc_carpentry_incubator': 'the HPC Carpentry incubator lessons', 'swc_shell_git': 'the Software Carpentry lessons',
    'enccs_lessons': 'the ENCCS lessons', 'enccs_lessons_src': 'the ENCCS lessons', 'coderefinery_lessons': 'the CodeRefinery lessons',
    'aalto_python_for_scicomp': 'the Aalto Python for Scientific Computing course', 'aalto_scicomp': 'the Aalto Scientific Computing (Triton) docs',
    'hpc_wiki': 'the HPC Wiki', 'fortran_lang_learn': 'the fortran-lang.org tutorials', 'nersc_docs': 'NERSC (Perlmutter)',
    'olcf_docs': 'OLCF (Frontier)', 'tacc_docs': 'TACC', 'princeton_rc_kb': 'Princeton Research Computing',
    'openmpi_docs': 'Open MPI', 'mpich_docs': 'MPICH', 'mpi4py_docs': 'mpi4py', 'intel_mpi_docs': 'Intel MPI',
    'nvidia_cuda_docs': 'NVIDIA CUDA', 'nvidia_nccl': 'NVIDIA NCCL', 'nvidia_nsight': 'NVIDIA Nsight',
    'nvidia_hpc_sdk': 'the NVIDIA HPC SDK', 'nvidia_nemo_megatron': 'NVIDIA NeMo / Megatron', 'rocm_hip_docs': 'AMD HIP',
    'rocm_docs': 'AMD ROCm', 'openacc_best_practices': 'OpenACC', 'openacc_org': 'OpenACC', 'openmp_org': 'OpenMP',
    'numba_cuda': 'Numba CUDA', 'cupy_user_guide': 'CuPy', 'pytorch_distributed_docs': 'PyTorch distributed',
    'pytorch_distributed_tutorials': 'PyTorch distributed', 'deepspeed_docs': 'DeepSpeed', 'hf_accelerate': 'Hugging Face Accelerate',
    'jax_multihost': 'JAX multi-host', 'tensorflow_distributed': 'TensorFlow distributed', 'vllm_docs': 'vLLM',
    'horovod_docs': 'Horovod', 'ray_cluster_slurm': 'Ray on Slurm', 'apptainer_user_guide': 'Apptainer',
    'apptainer_admin_guide': 'Apptainer', 'spack_docs': 'Spack', 'easybuild_docs': 'EasyBuild', 'lmod_docs': 'Lmod',
    'conda_docs': 'conda', 'mamba_docs': 'mamba', 'gcc_docs': 'GCC', 'gfortran_docs': 'GNU Fortran',
    'hpe_cray_pe': 'the HPE Cray Programming Environment', 'julia_mpi_jl': 'MPI.jl', 'julia_cuda_jl': 'CUDA.jl',
    'julia_hpc_guide': 'Julia on HPC', 'lapack_users_guide': 'LAPACK', 'scalapack_users_guide': 'ScaLAPACK',
    'fftw_manual': 'FFTW', 'petsc_manual': 'PETSc', 'hdf5_docs': 'HDF5', 'netcdf_c_docs': 'NetCDF-C', 'adios2_docs': 'ADIOS2',
    'lustre_manual': 'Lustre', 'lustre_wiki': 'Lustre', 'beegfs_docs': 'BeeGFS', 'daos_docs': 'DAOS', 'globus_docs': 'Globus',
    'scorep_user_guide': 'Score-P', 'scalasca_user_guide': 'Scalasca', 'linaro_forge': 'Linaro Forge (DDT/MAP)',
    'gdb_manual': 'GDB', 'valgrind_manual': 'Valgrind', 'perf_wiki': 'Linux perf', 'likwid_docs': 'LIKWID', 'tau_guide': 'TAU',
    'hpctoolkit_docs': 'HPCToolkit', 'darshan_docs': 'Darshan', 'snakemake_docs': 'Snakemake',
    'snakemake_slurm_plugin': 'the Snakemake Slurm executor plugin', 'nextflow_docs': 'Nextflow', 'parsl_docs': 'Parsl',
    'dask_jobqueue': 'Dask-Jobqueue', 'jupyterhub_docs': 'JupyterHub', 'fireworks_docs': 'FireWorks', 'htcondor_docs': 'HTCondor',
    'flux_docs': 'Flux', 'openpbs_docs': 'PBS', 'openssh_manuals': 'OpenSSH', 'rsync_manual': 'rsync',
    'jsc_juwels': 'JUWELS (JSC)', 'jsc_jureca': 'JURECA (JSC)', 'jsc_jupiter': 'JUPITER (JSC)', 'jsc_jusuf': 'JUSUF (JSC)',
    'bsc_supportkc': 'BSC (MareNostrum 5)', 'cineca_leonardo_gaps': 'CINECA Leonardo', 'lxp_meluxina_gaps': 'MeluXina (LuxProvide)',
    'it4i_karolina_gaps': 'IT4I Karolina', 'vega_docs': 'Vega (IZUM)', 'deucalion_docs': 'Deucalion (MACC)',
    'cran_hpc_task_view': 'the CRAN HPC task view'}

excluded = {url_key(u) for u in EXCLUDE.read_text().split() if u.strip()}
SKIP_PATH = ('/news', '/changelog', '/legal', '/privacy', '/cookie', '/events', '/blog', '/press', '/vacanc', '/jobs')
EXCLUDE_SUBSTR = ('/pgi-compilers/',)  # old PGI compiler PDFs (superseded by the NVIDIA HPC SDK)
CC_BY_SOURCES = {'lumi_docs', 'lumi_training', 'csc_docs'}  # prepare_factory_v1 rules, kept for consistency
NC_SA = ('idris', 'slaif', 'jsc', 'juwels', 'jupiter', 'gwdg')
SPDX = (r'(CC-BY-4\.0|CC-BY-SA-[34]\.0(?:-US)?|CC-BY-NC-SA-4\.0|CC0-1\.0|MIT|Apache-2\.0|BSD-[23]-Clause(?:-Patent)?|ISC|'
        r'A?GPL-[23]\.0|LGPL-[23]\.\d|GFDL-1\.[23])')
PERMISSIVE = re.compile(r'^(CC-BY-4\.0|CC0-1\.0|MIT|Apache-2\.0|BSD-[23]-Clause(-Patent)?|ISC|Apache-2\.0 OR MIT)$')
COPYLEFT = re.compile(r'^(CC-BY-SA-.*|A?GPL-.*|LGPL-.*|GFDL-.*)$')


def licence_of(source_id, url, stated):
    """prepare_factory_v1 rules first, then the manifest's stated licence."""
    if source_id in CC_BY_SOURCES or urlparse(url).netloc.endswith('europa.eu'):
        return 'CC-BY-4.0'
    if any(k in source_id for k in NC_SA):
        return 'CC-BY-NC-SA-4.0'
    s = (stated or '').strip()
    if not s or s in ('NOASSERTION', 'other') or s.lower().startswith('unstated'):
        return 'unspecified'
    m = re.fullmatch(SPDX + r'(?:\s*\([^)]*\))?', s)
    if m:
        return m.group(1)
    if s.startswith('CC-BY-4.0 (content)'):
        return 'CC-BY-4.0'
    return s  # proprietary, copyright-notice or compound terms: keep the stated string for the licence review


def licence_class(lic):
    if lic == 'unspecified':
        return 'unspecified'
    if lic == 'CC-BY-NC-SA-4.0':
        return 'non_commercial'
    if PERMISSIVE.match(lic):
        return 'permissive'
    if COPYLEFT.match(lic):
        return 'copyleft'
    return 'other_needs_review'


# The crawler may still be appending: use complete lines only and keep the prefix used.
raw = (ARCHIVE / 'manifest.jsonl').read_bytes()
raw = raw[:raw.rfind(b'\n') + 1]
(OUT / 'manifest_snapshot.jsonl').write_bytes(raw)
manifest_rows = []
for line in raw.decode('utf-8', errors='replace').splitlines():
    try:
        manifest_rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass

documents, seen_text, skipped = [], set(), Counter()
for r in manifest_rows:
    if r.get('status') != 'downloaded' or not r.get('text_path') or r.get('duplicate_of'):
        skipped['not_downloaded_or_duplicate_row'] += 1
        continue
    url = r.get('resolved_url') or r['url']
    host = urlparse(url).netloc
    if any(s in url for s in EXCLUDE_SUBSTR) or any(s in r['url'] for s in EXCLUDE_SUBSTR):
        skipped['excluded_pgi_compilers'] += 1
        continue
    if url_key(url) in excluded or url_key(r['url']) in excluded:
        skipped['excluded_eval_url'] += 1
        continue
    if host == 'slurm.schedmd.com':
        skipped['excluded_slurm_schedmd'] += 1
        continue
    if any(s in url.lower() for s in SKIP_PATH):
        skipped['excluded_news_like_path'] += 1
        continue
    if not (r.get('language') or 'en').lower().startswith('en'):
        skipped['excluded_language'] += 1
        continue
    path = ARCHIVE / r['text_path']
    if not path.exists():
        skipped['missing_text'] += 1
        continue
    text = path.read_text(errors='replace')
    if sha(text) != r.get('text_sha256'):
        skipped['text_hash_mismatch'] += 1
        continue
    if r['text_sha256'] in seen_text:
        skipped['duplicate_text'] += 1
        continue
    seen_text.add(r['text_sha256'])
    documents.append({'url': url, 'text': text, 'sha': r['text_sha256'], 'source_id': r['source_id'],
                      'site': NAMES.get(r['source_id'], 'the ' + host + ' documentation'),
                      'topic': r.get('topic') or 'unknown', 'kind': r.get('source_kind'),
                      'licence': licence_of(r['source_id'], url, r.get('license')),
                      'licence_stated': r.get('license'), 'licence_basis': r.get('license_basis')})

# Relevance: prepare_factory_v1's terms plus general-HPC tool and library names (MPI, GPU, I/O, profiling, builds).
TECH = re.compile(r'\b(sbatch|srun|salloc|squeue|sacct|scancel|sinfo|#SBATCH|module (?:load|avail|spider|use)|'
                  r'partition|--account|--gres|--gpus|--nodes|--ntasks|GPU|CUDA|ROCm|MPI|OpenMP|NCCL|RCCL|'
                  r'container|Singularity|Apptainer|conda|venv|pip install|PyTorch|TensorFlow|JAX|vLLM|'
                  r'scratch|quota|\$HOME|\$WORK|ssh|scp|rsync|login node|compute node|job script|queue|'
                  r'compil\w+|gcc|cmake|Lustre|filesystem|allocation|billing|node-hours|GPU-hours|'
                  r'mpirun|mpiexec|mpicc|mpif90|nvcc|hipcc|nvfortran|gfortran|ifx|icx|OpenACC|HIP|kernel|'
                  r'thread|rank|collective|HDF5|NetCDF|ADIOS2|MPI-IO|striping|lfs|Spack|EasyBuild|Lmod|'
                  r'profil\w+|tracing|debugger|gdb|valgrind|nsys|ncu|perf|LIKWID|Score-P|Scalasca|TAU|'
                  r'environment variable|LD_LIBRARY_PATH|DeepSpeed|Accelerate|torchrun|Horovod|'
                  r'Snakemake|Nextflow|Parsl|Dask|HTCondor|Flux|PBS|Globus|BeeGFS|DAOS)\b', re.I)
SLURM = re.compile(r'\b(sbatch|squeue|sacct|srun|salloc|scancel|Slurm)\b')
TASKS = ['howto', 'debug', 'concept', 'script']
# Disjoint from the v1-v4 angle sets, same style as v4's FINAL set.
ANGLES = ['troubleshooting a program or job that fails, hangs or gives wrong results',
          'choosing between two documented options or settings', 'how to verify that it worked as intended',
          'performance, scaling or efficiency of the setup', 'what to check or configure before starting',
          'adapting a laptop or workstation workflow to a cluster']
VARIANT_BASE = 9  # v1 used variants 1-3, v2 4-6, v4 7-9

# Every chunk an earlier batch could have asked about: exact chunk sha from their plans, normalised key from their seeds.
used_sha, used_key = set(), set()
for b in EARLIER:
    for plan in (REPO / 'runs' / b).glob('shard_*/scenario_plan.json'):
        for e in json.loads(plan.read_text()).get('entries', []):
            if e.get('chunk_sha256'):
                used_sha.add(e['chunk_sha256'])
    for seed in (REPO / 'runs' / b / 'seeds').glob('*.md'):
        for text in chunk_text(seed.read_text(errors='replace'), CHUNK_CHARS):
            used_sha.add(sha(text))
            used_key.add(text_key(text))

chunks, dropped = [], Counter()
for d in documents:
    for i, text in enumerate(chunk_text(d['text'], CHUNK_CHARS)):
        if not 650 <= len(text) <= 4500 or _question_issue(text, line_openings=True):
            dropped['length_or_unmatched_fence'] += 1
            continue
        tech = len({m.lower() for m in TECH.findall(text)})
        code = '```' in text or bool(re.search(r'^\s*[$#] \w', text, re.M))
        score = tech + 3 * code
        if score < 2:  # v3/v4 bar: still HPC terms or code
            dropped['low_relevance'] += 1
            continue
        if sha(text) in used_sha or text_key(text) in used_key:
            dropped['used_in_earlier_batch'] += 1
            continue
        chunks.append((score, d, i, text, code))
chunks.sort(key=lambda c: (-c[0], c[1]['url'], c[2]))
seen_chunk, unique = set(), []
for c in chunks:
    key = text_key(c[3])
    if key in seen_chunk:
        dropped['duplicate_chunk_in_archive'] += 1
        continue
    seen_chunk.add(key)
    unique.append(c)
tool_sources = [c for c in unique if SLURM.search(c[3])]
prose_budget = round(TARGET / 1.1)  # FINAL mix: tool (ask) rows add ~10% on top of prose
variants = max(1, min(3, -(-prose_budget // max(1, len(unique)))))
selected = unique[:max(1, prose_budget // variants)]

docs_used = {}
for _, d, _, _, _ in selected + tool_sources:
    if d['sha'] not in docs_used:
        host = urlparse(d['url']).netloc.split(':')[0].replace('.', '-')
        docs_used[d['sha']] = 'gen_%s_%s' % (host[:40], d['sha'][:16])
registry = {'version': 'source-family-registry-v1', 'families': {}, 'documents': []}
licences, doc_by_id = {}, {}
review_sha = sha(json.dumps({'archive': str(ARCHIVE), 'excluded': sorted(excluded), 'round': ROUND,
                             'manifest_rows': len(manifest_rows)}))
for d in documents:
    doc_id = docs_used.get(d['sha'])
    if not doc_id:
        continue
    family = 'general:' + d['source_id']
    registry['families'][family] = {'split': 'train', 'parents': [], 'lineage_reviewed': True}
    (OUT / 'seeds' / (doc_id + '.md')).write_text(d['text'])
    assert sha((OUT / 'seeds' / (doc_id + '.md')).read_bytes()) == d['sha']
    registry['documents'].append({'doc_id': doc_id, 'path': doc_id + '.md', 'sha256': d['sha'], 'family': family,
                                  'approved_for_training': True, 'origin': d['url'], 'source_version': ROUND,
                                  'permission_evidence': 'licence tag: ' + d['licence'] + ' (manifest: ' + str(d['licence_stated']) +
                                  '); archive rows marked not_approved_for_training/unreviewed, licence review pending; '
                                  'prepared on user request 2026-10-20',
                                  'admission_review_sha256': review_sha})
    licences[doc_id] = {'licence': d['licence'], 'licence_class': licence_class(d['licence']),
                        'licence_stated': d['licence_stated'], 'url': d['url'], 'site': d['site'],
                        'source_id': d['source_id'], 'topic': d['topic']}
    doc_by_id[d['sha']] = doc_id
dump(OUT / 'source_registry.json', registry)
dump(OUT / 'licences.json', licences)

entries = []
for n, (score, d, i, text, code) in enumerate(c for c in selected for _ in range(variants)):
    v = n % variants
    doc_id = doc_by_id[d['sha']]
    options = TASKS if code else ['howto', 'concept', 'debug']
    start = int(sha(doc_id + str(i))[:4], 16)
    task, angle = options[(start + v) % len(options)], ANGLES[(start + v) % len(ANGLES)]
    entries.append({'scenario_id': 'ctx-%06d' % n, 'round_id': ROUND, 'doc_id': doc_id,
        'chunk_id': doc_id + '#' + str(i), 'document_sha256': d['sha'], 'chunk_sha256': sha(text),
        'persona_index': (start + v + 2) % 4, 'task': task, 'mode': 'prose',
        'capability': 'General HPC source-backed ' + task,
        'brief': 'For ' + d['site'] + ' specifically (whichever software, tool or system the excerpt covers), ask a practical ' + task +
                 ' question about excerpt ' + str(i + 1) + ' of ' + d['url'] + ', focused on ' + angle + ' (variant ' + str(VARIANT_BASE + v + 1) + '). '
                 'Name the software, tool or system in the question. The attached excerpt '
                 'is the only source for its settings, options and commands. Ask for a full explanation needed to act correctly, including '
                 'prerequisites, the reasoning behind each step and checks, or uncertainty when applicable; do not '
                 'ask to memorize an absent source.',
        'context_spans': [{'start': 0, 'end': len(text), 'sha256': sha(text)}]})

tool_count = min(round(len(entries) * 0.1), len(tool_sources) * 3) if tool_sources else 0
for n in range(tool_count):
    _, d, i, text, _ = tool_sources[n % len(tool_sources)]
    variant = n // len(tool_sources) + 70  # v4 used 60+
    mode = 'ask'
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
# Shards of at most SHARD_SIZE questions. A document's entries stay together when they fit; a larger document is cut
# into SHARD_SIZE pieces (each piece's shard hard-links the same seed file), so no shard needs many review jobs.
per_doc = {}
for e in entries:
    per_doc.setdefault(e['doc_id'], []).append(e)
pieces = []
for doc_id, group in per_doc.items():
    group.sort(key=lambda e: e['scenario_id'])
    for s in range(0, len(group), SHARD_SIZE):
        pieces.append((doc_id, group[s:s + SHARD_SIZE]))
pieces.sort(key=lambda p: (-len(p[1]), p[0], p[1][0]['scenario_id']))
SHARDS = max(1, -(-len(entries) // SHARD_SIZE))
load, assigned = [0] * SHARDS, [[] for _ in range(SHARDS)]
for doc_id, group in pieces:
    k = min(range(SHARDS), key=lambda i: (load[i], i))
    assigned[k].append((doc_id, group))
    load[k] += len(group)
docs_meta = {d['doc_id']: d for d in registry['documents']}
for k in range(SHARDS):
    run = 'runs/%s/shard_%03d' % (BATCH, k)
    seeds = REPO / run / 'seeds'
    seeds.mkdir(parents=True)
    shard_registry = {'version': 'source-family-registry-v1', 'families': {}, 'documents': []}
    part = []
    for doc_id, group in assigned[k]:
        meta = docs_meta[doc_id]
        if not (seeds / meta['path']).exists():
            os.link(OUT / 'seeds' / meta['path'], seeds / meta['path'])
            shard_registry['documents'].append(meta)
            shard_registry['families'][meta['family']] = registry['families'][meta['family']]
        part.extend(group)
    dump(REPO / run / 'source_registry.json', shard_registry)
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
    'created_at': datetime.now(timezone.utc).isoformat(), 'round_id': ROUND, 'archive': str(ARCHIVE),
    'manifest_rows_used': len(manifest_rows), 'manifest_bytes_used': len(raw),
    'licence_scope': 'general HPC sources, licence-tagged per document (licences.json: licence, licence_class, licence_stated, topic)',
    'archive_documents_considered': len(documents), 'relevant_chunks': len(unique),
    'selected_chunks': len(selected), 'variants_per_chunk': variants, 'tool_source_chunks': len(tool_sources),
    'training_documents': len(docs_used), 'planned_questions': len(entries),
    'modes': dict(Counter(e['mode'] for e in entries)), 'shards': SHARDS,
    'questions_per_shard_min_max': [min(load), max(load)],
    'licences_by_document': dict(Counter(v['licence'] for v in licences.values()).most_common()),
    'planned_by_licence': dict(Counter(licences[e['doc_id']]['licence'] for e in entries).most_common()),
    'planned_by_licence_class': dict(Counter(licences[e['doc_id']]['licence_class'] for e in entries).most_common()),
    'planned_by_topic': dict(Counter(licences[e['doc_id']]['topic'] for e in entries).most_common()),
    'planned_by_source': dict(Counter(licences[e['doc_id']]['source_id'] for e in entries).most_common()),
    'skipped_documents': dict(skipped), 'dropped_chunks': dict(dropped),
    'earlier_batches_deduped': EARLIER, 'earlier_chunk_shas': len(used_sha), 'generated_examples': 0})
print((OUT / 'summary.json').read_text())
