"""G2 executable checks on saved eval answers: bash -n, invented Slurm flags, sbatch --test-only.

usage: static_checks.py ANSWERS_DIR OUT.json [ARM ...]   (default arms: base and the student strengths)
Per arm and eval slice:
- shell_pass: answers whose shell code blocks all pass `bash -n` (of answers that contain shell code)
- check_pass: answers with valid shell syntax and no invented flags (of all answers): the verify gate
- bad_flags_per_answer: invented Slurm long options per answer (flags not in the Slurm man pages)
- sbatch_ok: job scripts (#SBATCH blocks) accepted by `sbatch --test-only` on Leonardo, with account,
  partition and QoS forced to ours so generic placeholders do not fail every script. Nothing is submitted.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from distillkit.checks import check_answer, code_blocks, load_flags  # noqa: E402

answers, out = Path(sys.argv[1]), Path(sys.argv[2])
ARMS = sys.argv[3:] or ['base_closed', 'base_rag'] + [f's{s}{v}_{m}' for s in (42, 43, 44)
                                                      for v in ('', '_x05', '_x02') for m in ('closed', 'rag')]
FLAGS = load_flags()
SBATCH = shutil.which('sbatch')
FORCED = ['-A', 'EUHPC_D30_031', '-p', 'boost_usr_prod', '-q', 'normal']
PARTITION_LINE = re.compile(r'^\s*#SBATCH\s+(-p|--partition|-A|--account|-q|--qos|--reservation)\b.*$', re.M)


def sbatch_ok(script):
    """None when the block is not a job script, else whether `sbatch --test-only` accepts it."""
    if '#SBATCH' not in script or SBATCH is None:
        return None
    body = PARTITION_LINE.sub('', script if script.startswith('#!') else '#!/bin/bash\n' + script)
    with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False) as f:
        f.write(body)
    try:
        r = subprocess.run([SBATCH, '--test-only', *FORCED, f.name], capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return None
    finally:
        Path(f.name).unlink()


report = {}
for arm in ARMS:
    for label in ('v2', 'v3', 'd1'):
        path = answers / f'{label}__{arm}.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        n = len(rows)
        with_shell = shell_ok = passed = bad = scripts = scripts_ok = 0
        for r in rows:
            text = r.get('answer') or ''
            c = check_answer(text, FLAGS)
            bad += len(c['bad_flags'])
            passed += c['passed']
            if c['n_shell_blocks']:
                with_shell += 1
                shell_ok += c['bash_ok']
            for lang, block in code_blocks(text):
                if lang in ('bash', 'sh', 'shell', ''):
                    ok = sbatch_ok(block)
                    if ok is not None:
                        scripts += 1
                        scripts_ok += ok
        report[f'{label}/{arm}'] = {
            'answers': n, 'with_shell_code': with_shell,
            'shell_pass': round(shell_ok / with_shell, 4) if with_shell else None,
            'check_pass': round(passed / n, 4) if n else None,
            'bad_flags_per_answer': round(bad / n, 4) if n else None,
            'job_scripts': scripts, 'sbatch_ok': round(scripts_ok / scripts, 4) if scripts else None}
        print(label, arm, report[f'{label}/{arm}'], flush=True)
out.write_text(json.dumps(report, indent=2))
