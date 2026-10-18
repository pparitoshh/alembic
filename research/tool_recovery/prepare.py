"""Fresh tool-only question plan using the original admitted source snapshot."""
import hashlib
import json
from pathlib import Path

from distillkit.config import load_config
from distillkit.generate import question_jobs

repo = Path(__file__).resolve().parents[2]
original = repo / 'runs/ai_factory_pilot_20261013'
root = repo / 'runs/tool_recovery_20261014'
combined = repo / 'runs/ai_factory_recovery_20261014'
if (root / 'generation_manifest.json').exists():
    raise SystemExit('Generation already started; do not overwrite its inputs')
root.mkdir(parents=True, exist_ok=True)
combined.mkdir(parents=True, exist_ok=True)
plan = json.loads((original / 'scenario_plan.json').read_text())
entries = []
for old in plan['entries']:
    if old['mode'] == 'prose':
        continue
    entry = dict(old)
    index = len(entries)
    mode = entry['mode']
    entry['scenario_id'] = 'recovery-%03d' % index
    entry['round_id'] = 'ai-factory-tool-recovery-1'
    entry['task'] = 'concept' if mode == 'none' else 'debug' if mode == 'ask' else 'howto'
    entry['capability'] = 'Source-supported tool decision: ' + mode
    common = ('This is a tool-interface workflow on the supplied simulated cluster, not a shell-command tutorial. '
              'Do not request commands, code, scripts, job arrays or requeue behavior. Do not assert LUMI defaults '
              'about the synthetic cluster. Keep the objective narrow enough for the source and documented tools. ')
    if mode == 'call':
        objectives = ['current state and reported pending reason', 'current state and elapsed time',
                      'allocated nodes and current state', 'allocated GPU count and current state',
                      'reported time limit and elapsed time', 'current state without assuming a failure cause',
                      'current status even if the user has an older observation', 'current state and available exit-code information']
        detail = ('Explicitly ask the assistant to look up ' + objectives[(index // 2) % len(objectives)] +
                  ' NOW for the exact job identifier supplied in the simulated scenario. Write the target as '
                  '"job DIGITS", using the supplied ID, never "job (ID DIGITS)". '
                  'This must be an actual lookup request, not "how can I" or a hypothetical example. '
                  'No need to state the current values in the question; the tool must establish them.')
    elif mode == 'ask':
        detail = ('Request a live investigation of the user\'s own job, tied to this source topic, but deliberately '
                  'omit its job ID and provide no previously observed tool result. The missing job target must '
                  'be essential; ask-first policy applies. Do not replace this with a GPU/partition availability '
                  'question, because those tools allow an unfiltered lookup. Do not supply a guessed diagnosis.')
    else:
        detail = ('Ask for a conceptual explanation of one specific distinction or limitation actually stated '
                  'in this source excerpt, without any live lookup. Explicitly request an explanation in prose '
                  'without commands or code. No invented job snapshot, no claim that a check was performed. '
                  'Use the substantive source topic, not a generic restatement of what a tool is.')
    entry['brief'] = common + detail + ' Source-specific scenario number ' + str(index + 1) + '.'
    entry.pop('context_spans', None)
    entries.append(entry)
(root / 'scenario_plan.json').write_text(json.dumps({'version': 'source-scenario-plan-v2', 'entries': entries}, indent=2))
cfg = json.loads((repo / 'configs/ai_factory_pilot.json').read_text())
cfg['run_dir'] = 'runs/tool_recovery_20261014'
cfg['generate']['scenario_plan'] = 'runs/tool_recovery_20261014/scenario_plan.json'
cfg['teacher']['temperature'] = .3
cfg['teacher']['max_tokens'] = 2048
cfg['judge']['concurrency'] = 4
cfg['verify']['grounding_reviews'] = 'runs/tool_recovery_20261014/grounding_reviews.jsonl'
cfg['eval']['max_new_tokens'] = 2048
path = repo / 'configs/tool_recovery.json'
path.write_text(json.dumps(cfg, indent=2))
jobs = question_jobs(load_config(path))
assert len(jobs) == 77 and all(j['mode'] != 'prose' for j in jobs)
cfg['run_dir'] = 'runs/ai_factory_recovery_20261014'
cfg['verify']['grounding_reviews'] = 'runs/ai_factory_recovery_20261014/grounding_reviews.jsonl'
(repo / 'configs/ai_factory_recovery.json').write_text(json.dumps(cfg, indent=2))
(combined / 'mentor_code_snapshot.json').write_text(json.dumps({
    str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((repo / 'research/mentor_followup').iterdir()) if p.is_file()}, indent=2))
print('Prepared', len(jobs), 'fresh tool candidates; original registry and quality gates unchanged')
