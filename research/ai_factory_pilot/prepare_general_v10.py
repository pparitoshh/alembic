"""Topic-targeted general-HPC generation batch (ai_factory_all_v10): prepare_general_v9.py plus a v9 exclusion.

v10 = v9 with three changes: (1) defaults BATCH=ai_factory_all_v10, ROUND=general-hpc-10, TOOL_VARIANT_BASE=89 (Slurm
tool scenarios 90+; v9 used 81+); (2) no (chunk sha256, variant) pair planned by an earlier same-scheme batch
(SAME_SCHEME_BATCHES, default $REPO/runs/ai_factory_all_v9; prose variants 16-21) is planned again: such a chunk only
gets the variants v9 left free (none free -> chunk skipped); (3) the angle-overlap report also counts angles asked by
those batches (v5-v8 and v9 reported separately). The v9 docstring follows.

usage: BATCH=ai_factory_all_v10 ROUND=general-hpc-10 LEAVES=area/leaf,... \
       prepare_general_v9.py ARCHIVE[,ARCHIVE...] EXCLUDE_URLS.txt TARGET_QUESTIONS [SHARD_SIZE]
Same output layout and config format as prepare_general_v5.py (factory_worker.sh, factory_dbg.sbatch,
merge_factory_v1.py and annotate_general_v5.py run unchanged). Differences from v5:
- several archives in one batch (comma-separated; documents deduplicated by text sha across archives; one
  manifest_snapshot_<archive>.jsonl each);
- only chunks that match a target leaf (LEAVES) with the taxonomy's own regexes (research/general_hpc_docs/taxonomy.json,
  case-insensitive, as coverage.py compiles them): a pattern in the document URL, or >= MIN_CHUNK_HITS matches in the
  chunk text. mpi_containers' generic patterns ("hybrid model", "bind model") also need a container word in the
  chunk (they otherwise match MPI+OpenMP text). Each chunk counts for one leaf: the one with most hits (URL match = +2);
  ties go to the leaf listed first in LEAVES;
- per-leaf budget: TARGET is shared out by water-filling, no leaf above PER_LEAF_CAP planned questions (tool rows
  included);
- angles: there is no third angle set. Both archives' chunks were already asked with 3 angles of set 1 (v5/v6) and 3 of
  set 2 (v7/v8), always at offsets start..start+2 of the 6-angle list. v9 uses the six combinations never asked for a
  chunk: set 1 and set 2 at offsets start+3..start+5 (variants VARIANT_BASE+1..+6, up to MAX_VARIANTS per chunk, so
  thin leaves get more questions per chunk; a chunk with a single passing mention of the topic, i.e. one regex hit and no
  URL match, gets at most WEAK_MAX_VARIANTS=3). summary.json reports how many planned (chunk, angle) pairs repeat an
  earlier v5-v8 plan (expected 0);
- the brief names the target topic (leaf name) so the question is about it where the excerpt allows;
- Slurm tool rows use scenario numbers TOOL_VARIANT_BASE+ (v5-v8 used 70+), only from target-leaf chunks;
- configs name the teacher TEACHER (default QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ);
- leaf_plan.json maps scenario_id -> leaf (the scenario plan schema forbids extra keys);
- DRY_PLAN=1 prints the per-leaf plan and writes nothing.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

from distillkit.generate import _question_issue

REPO = Path(__file__).resolve().parents[2]
ARCHIVES = [Path(a) for a in sys.argv[1].split(',') if a]
EXCLUDE = Path(sys.argv[2])
TARGET = int(sys.argv[3])
SHARD_SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 450
BATCH = os.environ.get('BATCH', 'ai_factory_all_v10')
ROUND = os.environ.get('ROUND', 'general-hpc-10')
LEAVES = [x.strip() for x in os.environ.get('LEAVES', '').split(',') if x.strip()]
PER_LEAF_CAP = int(os.environ.get('PER_LEAF_CAP', '1500'))
MAX_VARIANTS = int(os.environ.get('MAX_VARIANTS', '6'))
MIN_CHUNK_HITS = int(os.environ.get('MIN_CHUNK_HITS', '1'))
WEAK_MAX_VARIANTS = int(os.environ.get('WEAK_MAX_VARIANTS', '3'))
TEACHER = os.environ.get('TEACHER', 'QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ')
TOOL_VARIANT_BASE = int(os.environ.get('TOOL_VARIANT_BASE', '89'))
TAXONOMY = Path(os.environ.get('TAXONOMY', REPO / 'research/general_hpc_docs/taxonomy.json'))
# Earlier plans, only read (angle-overlap report): a08trc09's pipeline runs.
EARLIER_ROOT = Path(os.environ.get('EARLIER_ROOT', '/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/code/runs'))
EARLIER_ANGLE_BATCHES = ['ai_factory_all_v5', 'ai_factory_all_v6', 'ai_factory_all_v7', 'ai_factory_all_v8']
SAME_SCHEME_BATCHES = [Path(x) for x in os.environ.get('SAME_SCHEME_BATCHES', str(REPO / 'runs/ai_factory_all_v9')).split(',') if x]
DRY = os.environ.get('DRY_PLAN') == '1'
OUT = REPO / 'runs' / BATCH
if not LEAVES:
    raise SystemExit('Set LEAVES=area/leaf,...')
if OUT.exists() and not DRY:
    raise SystemExit('Preparation exists; use a new version instead of overwriting')


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

# Target leaves with the taxonomy's own patterns (compiled like coverage.py: one alternation, re.I).
tax = json.loads(TAXONOMY.read_text())
leaf_info = {}
for a in tax['areas']:
    for leaf in a['leaves']:
        leaf_info[a['id'] + '/' + leaf['id']] = leaf
missing = [k for k in LEAVES if k not in leaf_info]
if missing:
    raise SystemExit('unknown leaves: %s' % missing)
LEAF_RE = {k: re.compile('|'.join('(?:%s)' % p for p in leaf_info[k]['patterns']), re.I) for k in LEAVES}
LEAF_NAME = {k: leaf_info[k]['name'] for k in LEAVES}
CO_REQUIRE = {'containers/mpi_containers': re.compile(
    r'\b(containers?|apptainer|singularity|sarus|enroot|podman|charliecloud|docker|\.sif)\b', re.I)}

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


# The crawler may still be appending: use complete lines only and keep the prefix used (written later, not in a dry run).
snapshots, manifest_rows = {}, []
for ARCHIVE in ARCHIVES:
    raw = (ARCHIVE / 'manifest.jsonl').read_bytes()
    raw = raw[:raw.rfind(b'\n') + 1]
    snapshots[ARCHIVE.name] = raw
    for line in raw.decode('utf-8', errors='replace').splitlines():
        try:
            manifest_rows.append((ARCHIVE, json.loads(line)))
        except json.JSONDecodeError:
            pass

documents, seen_text, skipped = [], set(), Counter()
for ARCHIVE, r in manifest_rows:
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
                      'topic': r.get('topic') or 'unknown', 'kind': r.get('source_kind'), 'archive': ARCHIVE.name,
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
BOILER = re.compile(r'Notices and Disclaimers|Intel technologies may require enabled hardware|Performance varies by use, '
                    r'configuration and other factors|No product or component can be absolutely secure|'
                    r'Intel Corporation\. Intel, the Intel logo|NVIDIA Corporation \(.NVIDIA.\) makes no representations|'
                    r'Information in this document is provided in connection with|All rights reserved\. No part of', re.I)
TASKS = ['howto', 'debug', 'concept', 'script']
ANGLES_1 = ['troubleshooting a program or job that fails, hangs or gives wrong results',
            'choosing between two documented options or settings', 'how to verify that it worked as intended',
            'performance, scaling or efficiency of the setup', 'what to check or configure before starting',
            'adapting a laptop or workstation workflow to a cluster']
ANGLES_2 = ['a common mistake and how to avoid it', 'what a specific output, log line or error message means',
            'a minimal working example a new cluster user can run', 'comparing it with an alternative tool or approach',
            'limits, quotas or fair-share effects to be aware of', 'automating or scripting it for many runs']
VARIANT_BASE = int(os.environ.get('VARIANT_BASE', '15'))  # v5/v6 10-12, v7/v8 13-15


def combo(start, v, code):
    """Variant v (0..5) of a chunk: the (angle, task) combinations v5-v8 never asked (offsets 3-5 of both sets)."""
    angles = (ANGLES_1, ANGLES_2)[v % 2]
    off = 3 + v // 2
    options = TASKS if code else ['howto', 'concept', 'debug']
    return angles[(start + off) % len(angles)], options[(start + 3 + v) % len(options)]


def leaf_hits(d, text):
    best = None
    for rank, k in enumerate(LEAVES):
        n = len(LEAF_RE[k].findall(text))
        url_hit = bool(LEAF_RE[k].search(d['url']))
        if k in CO_REQUIRE and not CO_REQUIRE[k].search(text):
            continue
        if not url_hit and n < MIN_CHUNK_HITS:
            continue
        cand = (n + 2 * url_hit, -rank, k, n, url_hit)
        if best is None or cand > best:
            best = cand
    return best


def scan_doc(j):
    # v10: per-document chunk scan, run in WORKERS forked processes (login-node CPU-time limit is per process).
    d, out, drop = documents[j], [], Counter()
    for i, text in enumerate(chunk_text(d['text'], CHUNK_CHARS)):
        if not 650 <= len(text) <= 4500 or _question_issue(text, line_openings=True):
            drop['length_or_unmatched_fence'] += 1
            continue
        if BOILER.search(text):
            drop['legal_boilerplate'] += 1
            continue
        hit = leaf_hits(d, text)
        if hit is None:
            drop['no_target_leaf'] += 1
            continue
        tech = len({m.lower() for m in TECH.findall(text)})
        code = '```' in text or bool(re.search(r'^\s*[$#] \w', text, re.M))
        score = tech + 3 * code
        if score < 2:  # v3/v4 bar: still HPC terms or code
            drop['low_relevance'] += 1
            continue
        out.append({'leaf': hit[2], 'leaf_score': hit[0], 'hits': hit[3], 'url_hit': hit[4],
                    'score': score, 'j': j, 'i': i, 'text': text, 'code': code})
    return out, drop


import multiprocessing
import time
WORKERS = int(os.environ.get('WORKERS', '3'))
_t0 = time.time()
print('documents loaded: %d (%.0fs CPU so far)' % (len(documents), time.process_time()), file=sys.stderr, flush=True)
chunks, dropped = [], Counter()
with multiprocessing.get_context('fork').Pool(WORKERS) as pool:
    for out, drop in pool.imap(scan_doc, range(len(documents)), chunksize=16):
        dropped.update(drop)
        for c in out:
            c['d'] = documents[c.pop('j')]
            chunks.append(c)
print('chunk scan: %d chunks in %.0fs wall (%.0fs main CPU)' % (len(chunks), time.time() - _t0, time.process_time()),
      file=sys.stderr, flush=True)
# Strongest leaf evidence first, then v5's relevance order.
chunks.sort(key=lambda c: (-c['leaf_score'], -c['score'], c['d']['url'], c['i']))
seen_chunk, by_leaf = set(), defaultdict(list)
for c in chunks:
    key = text_key(c['text'])
    if key in seen_chunk:
        dropped['duplicate_chunk_in_archive'] += 1
        continue
    seen_chunk.add(key)
    by_leaf[c['leaf']].append(c)

# (chunk sha, prose variant number) pairs already planned by same-scheme batches (v9): never planned again.
same_used, same_angles, same_plans = defaultdict(set), defaultdict(set), 0
for b in SAME_SCHEME_BATCHES:
    for plan in b.glob('shard_*/scenario_plan.json'):
        same_plans += 1
        for e in json.loads(plan.read_text()).get('entries', []):
            if e.get('mode') != 'prose' or not e.get('chunk_sha256'):
                continue
            m = re.search(r'focused on (.*?) \(variant (\d+)\)', e.get('brief', ''))
            if m:
                same_used[e['chunk_sha256']].add(int(m.group(2)))
                same_angles[e['chunk_sha256']].add(m.group(1))
if SAME_SCHEME_BATCHES and not same_plans:
    raise SystemExit('no scenario plans found in %s' % SAME_SCHEME_BATCHES)
for c in chunks:
    c['free'] = [v for v in range(MAX_VARIANTS) if VARIANT_BASE + v + 1 not in same_used.get(sha(c['text']), ())]
for k in list(by_leaf):
    kept = [c for c in by_leaf[k] if c['free']]
    dropped['all_variants_used_by_v9'] += len(by_leaf[k]) - len(kept)
    by_leaf[k] = kept


def chunk_cap(c):
    return min(len(c['free']), MAX_VARIANTS if (c['hits'] >= 2 or c['url_hit']) else min(MAX_VARIANTS, WEAK_MAX_VARIANTS))


# Per-leaf prose budgets: water-fill TARGET/1.1 over the leaves, each at most min(cap/1.1, chunks * MAX_VARIANTS).
prose_cap = int(PER_LEAF_CAP / 1.1)
strong = lambda c: c['hits'] >= 2 or c['url_hit']
avail = {k: min(prose_cap, sum(chunk_cap(c) for c in by_leaf[k]))
         for k in LEAVES}
want = round(TARGET / 1.1)
lo, hi = 0, max(avail.values()) if avail else 0
while lo < hi:
    mid = (lo + hi + 1) // 2
    if sum(min(mid, a) for a in avail.values()) <= want:
        lo = mid
    else:
        hi = mid - 1
budget = {k: min(lo, avail[k]) for k in LEAVES}
plan_leaf, selected = {}, []  # selected: (chunk, variants)
for k in LEAVES:
    n = len(by_leaf[k])
    if not n or not budget[k]:
        plan_leaf[k] = {'chunks_matched': n, 'variant_level': 0, 'chunks_selected': 0, 'prose': 0}
        continue
    # A chunk with a single passing mention (one hit, no URL match) gets at most WEAK_MAX_VARIANTS questions; the
    # variant level is the lowest that fills the budget, chunks are taken in leaf-evidence order until it is filled.
    cap_c = [chunk_cap(c) for c in by_leaf[k]]
    level = next((L for L in range(1, MAX_VARIANTS + 1) if sum(min(L, m) for m in cap_c) >= budget[k]), MAX_VARIANTS)
    sel, total = [], 0
    for c, m in zip(by_leaf[k], cap_c):
        if total >= budget[k]:
            break
        v = min(level, m, budget[k] - total)
        sel.append((c, v))
        total += v
    selected += sel
    plan_leaf[k] = {'chunks_matched': n, 'variant_level': level, 'chunks_selected': len(sel), 'prose': total,
                    'url_matched_chunks': sum(c['url_hit'] for c in by_leaf[k]),
                    'single_mention_chunks': sum(1 for c in by_leaf[k] if c['hits'] < 2 and not c['url_hit'])}

# Angles v5-v8 already asked per chunk sha (only read, for the overlap report).
earlier_angles, earlier_plans = defaultdict(set), 0
for b in EARLIER_ANGLE_BATCHES:
    for plan in (EARLIER_ROOT / b).glob('shard_*/scenario_plan.json'):
        earlier_plans += 1
        for e in json.loads(plan.read_text()).get('entries', []):
            m = re.search(r'focused on (.*?) \(variant', e.get('brief', ''))
            if m and e.get('chunk_sha256'):
                earlier_angles[e['chunk_sha256']].add(m.group(1))

docs_used = {}


def doc_id_of(d):
    if d['sha'] not in docs_used:
        host = urlparse(d['url']).netloc.split(':')[0].replace('.', '-')
        docs_used[d['sha']] = 'gen_%s_%s' % (host[:40], d['sha'][:16])
    return docs_used[d['sha']]


entries, leaf_of, overlap, chunk_with_history = [], {}, 0, 0
overlap_v9, chunk_with_v9, same_pair = 0, 0, 0
n = 0
for c, variants in selected:
    d, i, text = c['d'], c['i'], c['text']
    doc_id = doc_id_of(d)
    start = int(sha(doc_id + str(i))[:4], 16)
    prior = earlier_angles.get(sha(text), set())
    chunk_with_history += bool(prior)
    prior9 = same_angles.get(sha(text), set())
    chunk_with_v9 += bool(prior9)
    for v in c['free'][:variants]:
        angle, task = combo(start, v, c['code'])
        overlap += angle in prior
        overlap_v9 += angle in prior9
        same_pair += (VARIANT_BASE + v + 1) in same_used.get(sha(text), ())
        sid = 'ctx-%06d' % n
        n += 1
        leaf_of[sid] = c['leaf']
        entries.append({'scenario_id': sid, 'round_id': ROUND, 'doc_id': doc_id,
            'chunk_id': doc_id + '#' + str(i), 'document_sha256': d['sha'], 'chunk_sha256': sha(text),
            'persona_index': (start + v + 3) % 4, 'task': task, 'mode': 'prose',
            'capability': 'General HPC source-backed ' + task,
            'brief': 'For ' + d['site'] + ' specifically (whichever software, tool or system the excerpt covers), ask a practical ' + task +
                     ' question about excerpt ' + str(i + 1) + ' of ' + d['url'] + ', focused on ' + angle + ' (variant ' + str(VARIANT_BASE + v + 1) + '). '
                     'Where the excerpt allows, make it about the topic "' + LEAF_NAME[c['leaf']] + '". '
                     'Name the software, tool or system in the question. The attached excerpt '
                     'is the only source for its settings, options and commands. Ask for a full explanation needed to act correctly, including '
                     'prerequisites, the reasoning behind each step and checks, or uncertainty when applicable; do not '
                     'ask to memorize an absent source.',
            'context_spans': [{'start': 0, 'end': len(text), 'sha256': sha(text)}]})

# Slurm tool rows (~10%) from target-leaf chunks, within each leaf's cap.
prose_by_leaf = Counter(leaf_of.values())
tool_sources = [c for c, _ in selected if SLURM.search(c['text'])]
tool_count = min(round(len(entries) * 0.1), len(tool_sources) * 3) if tool_sources else 0
tool_by_leaf, t = Counter(), 0
for n in range(tool_count):
    c = tool_sources[n % len(tool_sources)]
    if prose_by_leaf[c['leaf']] + tool_by_leaf[c['leaf']] >= PER_LEAF_CAP:
        continue
    d, i = c['d'], c['i']
    variant = n // len(tool_sources) + TOOL_VARIANT_BASE
    mode = 'ask'
    task = ['howto', 'debug'][variant % 2]
    doc_id = doc_id_of(d)
    sid = 'tool-%06d' % t
    t += 1
    leaf_of[sid] = c['leaf']
    tool_by_leaf[c['leaf']] += 1
    entries.append({'scenario_id': sid, 'round_id': ROUND, 'doc_id': doc_id,
        'chunk_id': doc_id + '#' + str(i), 'document_sha256': d['sha'], 'chunk_sha256': sha(c['text']),
        'persona_index': variant % 4, 'task': task, 'mode': mode, 'capability': 'Slurm tool decision ' + mode,
        'brief': 'Create a ' + task + ' request in mode ' + mode + ' concerning the Slurm workflow in excerpt ' + str(i + 1) +
                 ' of ' + d['url'] + ' (' + d['site'] + '), scenario ' + str(variant + 1) + '. Site-independent tool observations come '
                 'exclusively from the supplied mock cluster, not from the real scheduler. Use a different practical '
                 'user objective for each scenario. Never invent a workload, real queue state, missing required job '
                 'ID, or undocumented flag.'})
for k in LEAVES:
    plan_leaf[k]['tool'] = tool_by_leaf[k]
    plan_leaf[k]['planned'] = plan_leaf[k]['prose'] + tool_by_leaf[k]
SHARDS = max(1, -(-len(entries) // SHARD_SIZE))
angle_report = {'earlier_plans_read': earlier_plans, 'selected_chunks_with_earlier_angles': chunk_with_history,
                'planned_prose_rows_repeating_an_earlier_angle_on_the_same_chunk': overlap,
                'same_scheme_plans_read': same_plans, 'selected_chunks_also_in_v9': chunk_with_v9,
                'planned_prose_rows_repeating_a_v9_angle_on_the_same_chunk': overlap_v9,
                'planned_chunk_variant_pairs_already_planned_by_v9': same_pair}
assert same_pair == 0
if DRY:
    print(json.dumps({'documents': len(documents), 'leaf_matched_chunks': sum(len(v) for v in by_leaf.values()),
                      'water_level': lo, 'per_leaf': plan_leaf, 'planned_questions': len(entries), 'shards': SHARDS,
                      'angles': angle_report, 'dropped_chunks': dict(dropped), 'skipped_documents': dict(skipped),
                      'planned_by_source': dict(Counter(e['doc_id'].split('_')[1] for e in entries).most_common(25))}, indent=1))
    raise SystemExit(0)

(OUT / 'seeds').mkdir(parents=True)
for name, raw in snapshots.items():
    (OUT / ('manifest_snapshot_%s.jsonl' % name)).write_bytes(raw)
registry = {'version': 'source-family-registry-v1', 'families': {}, 'documents': []}
licences, doc_by_id = {}, {}
review_sha = sha(json.dumps({'archive': [str(a) for a in ARCHIVES], 'excluded': sorted(excluded), 'round': ROUND,
                             'manifest_rows': len(manifest_rows), 'leaves': LEAVES}))
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
                                  'prepared on user request 2026-10-23 (topic-targeted batch v10, d1 topics)',
                                  'admission_review_sha256': review_sha})
    licences[doc_id] = {'licence': d['licence'], 'licence_class': licence_class(d['licence']),
                        'licence_stated': d['licence_stated'], 'url': d['url'], 'site': d['site'],
                        'source_id': d['source_id'], 'topic': d['topic'], 'archive': d['archive']}
    doc_by_id[d['sha']] = doc_id
dump(OUT / 'source_registry.json', registry)
dump(OUT / 'licences.json', licences)
dump(OUT / 'leaf_plan.json', leaf_of)

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
            os.link(OUT / 'seeds' / meta['path'], seeds / meta['path'])  # same batch dir, same group permissions
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
    cfg['teacher']['model'] = TEACHER
    cfg['teacher']['max_tokens'] = 2048
    cfg['teacher']['concurrency'] = 48
    cfg['judge']['concurrency'] = 32
    cfg['verify']['grounding_reviews'] = run + '/grounding_reviews.jsonl'
    dump(REPO / ('configs/%s_shard_%03d.json' % (BATCH, k)), cfg)

dump(OUT / 'summary.json', {
    'created_at': datetime.now(timezone.utc).isoformat(), 'round_id': ROUND, 'archive': [str(a) for a in ARCHIVES],
    'leaves': LEAVES, 'per_leaf_cap': PER_LEAF_CAP, 'max_variants': MAX_VARIANTS, 'teacher': TEACHER,
    'variant_base': VARIANT_BASE, 'tool_variant_base': TOOL_VARIANT_BASE, 'water_level': lo,
    'manifest_rows_used': len(manifest_rows), 'manifest_bytes_used': {k: len(v) for k, v in snapshots.items()},
    'licence_scope': 'general HPC sources, licence-tagged per document (licences.json: licence, licence_class, licence_stated, topic)',
    'archive_documents_considered': len(documents), 'relevant_chunks': sum(len(v) for v in by_leaf.values()),
    'selected_chunks': len(selected), 'tool_source_chunks': len(tool_sources),
    'training_documents': len(docs_used), 'planned_questions': len(entries), 'per_leaf': plan_leaf,
    'angles': angle_report,
    'modes': dict(Counter(e['mode'] for e in entries)), 'shards': SHARDS,
    'questions_per_shard_min_max': [min(load), max(load)],
    'licences_by_document': dict(Counter(v['licence'] for v in licences.values()).most_common()),
    'planned_by_licence': dict(Counter(licences[e['doc_id']]['licence'] for e in entries).most_common()),
    'planned_by_licence_class': dict(Counter(licences[e['doc_id']]['licence_class'] for e in entries).most_common()),
    'planned_by_topic': dict(Counter(licences[e['doc_id']]['topic'] for e in entries).most_common()),
    'planned_by_source': dict(Counter(licences[e['doc_id']]['source_id'] for e in entries).most_common()),
    'skipped_documents': dict(skipped), 'dropped_chunks': dict(dropped), 'generated_examples': 0})
print((OUT / 'summary.json').read_text())
