"""Timestamped half-hour reports. Only completed, audited decisions supply scores."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from evidence import read_json
from pool import Pool
from supervisor import ACTIVE, Slurm, atomic, lock, now


def published_scores(config):
    result = {}
    for judge, competitors in config['leaderboard_snapshot']['judges'].items():
        release = competitors['rel12e_rag']
        mean = sum(x['score'] for x in release['slices'].values()) / 3
        result[judge] = {'mean': mean, 'rank': 1 + sum(x['mean_upper'] > mean for x in competitors.values()),
                         'scores': {label: x['score'] for label, x in release['slices'].items()}}
    return result


def capture(config, state, jobs, previous=None, timestamp=None):
    timestamp = timestamp or now()
    published = published_scores(config)
    floor = min(s for judge in published.values() for s in judge['scores'].values())
    best = {'run': 'release12e', 'arm': 'rel12e_rag', 'minimum_score': floor, 'judges': published}
    for decision in state.get('decisions', []):
        observed = decision.get('best_observed')
        if observed and observed['minimum_score'] > best['minimum_score']:
            best = {'run': decision['run'], **observed}
    previous_runs = {d['run'] for d in (previous or {}).get('decisions', [])}
    new = [d for d in state.get('decisions', []) if d['run'] not in previous_runs]
    before = (previous or {}).get('best', {}).get('minimum_score', best['minimum_score'])
    active = {k: {field: v.get(field) for field in ('job_id', 'scheduler_state', 'attempt')}
              for k, v in state.get('tasks', {}).items() if v.get('status') == 'running'}
    age = max(0, (datetime.fromisoformat(timestamp) - datetime.fromisoformat(state['updated_at'])).total_seconds())
    return {'at': timestamp, 'coordinator_updated_at': state['updated_at'], 'heartbeat_age_seconds': age,
        'status': state['status'], 'phase': state['phase'], 'round_index': state['round_index'],
        'active_tasks': active, 'gpu_jobs': jobs, 'running_nodes': sum(j['nodes'] for j in jobs if j['state'] in ('RUNNING', 'COMPLETING')),
        'background': state.get('backfill', {}), 'pool_enabled': config.get('pool', {}).get('enabled', False),
        'published': published, 'best': best, 'improvement_percentage_points': 100 * (best['minimum_score'] - before),
        'decisions': state.get('decisions', []), 'new_decisions': new, 'detail': state.get('detail', ''),
        'target': config.get('target', {}), 'scope': 'Internal development judge preference; not factual accuracy or a public world ranking.'}


def render(report):
    r = report
    lines = ['# HPC status — ' + r['at'], '',
        f"State: **{r['status']}**; phase: **{r['phase']}**; experiment index: {r['round_index']}.",
        f"Coordinator heartbeat age: {r['heartbeat_age_seconds']:.0f} seconds.", '',
        f"GPU allocation: **{r['running_nodes']}/8 nodes ({4 * r['running_nodes']}/32 A100s)**. Allocation is not measured GPU utilization."]
    for job in r['gpu_jobs']:
        lines.append(f"- Job `{job['id']}`: {job['state']}, {job['nodes']} nodes.")
    for task, info in r['active_tasks'].items():
        lines.append(f"- Foreground: `{task}`, CPU controller `{info['job_id']}`.")
    if r['background']:
        lines.append('- Background: ' + r['background'].get('detail', 'starting'))
    lines += ['', '| Published release 12e | v2 | v3 | d1 | Mean | Internal rank |',
              '|---|---:|---:|---:|---:|---:|']
    for judge, scores in r['published'].items():
        values = ' | '.join(f"{scores['scores'][s]:.2%}" for s in ('v2', 'v3', 'd1'))
        lines.append(f"| {judge} | {values} | {scores['mean']:.2%} | {scores['rank']}/13 |")
    b = r['best']
    lines += ['', f"Best verified minimum across all six scores: **{b['minimum_score']:.2%}**, `{b['run']}/{b['arm']}`.",
              f"Change in that best minimum since the previous report: **{r['improvement_percentage_points']:+.2f} percentage points**."]
    if not r['new_decisions']:
        lines.append('No new verified score since the previous report. Incomplete evaluations are not scored.')
    for decision in r['new_decisions']:
        lines.append(f"New completed comparison: `{decision['run']}`; target {'met' if decision['goal_met'] else 'not met'}.")
        if decision.get('best_observed'):
            item = decision['best_observed']
            lines.append(f"Its strongest arm by minimum score is `{item['arm']}`: {item['minimum_score']:.2%} minimum; " +
                         '; '.join(f"{j} mean {v['mean']:.2%}, rank {v['rank']}" for j, v in item['judges'].items()) + '.')
    latest = next((d for d in reversed(r['decisions']) if d.get('best_observed')), None)
    if latest:
        item = latest['best_observed']
        lines += ['', f"Latest fully evaluated candidate: `{latest['run']}/{item['arm']}` (not automatically promoted).",
                  '', '| Judge | v2 | v3 | d1 | Mean | Internal rank |', '|---|---:|---:|---:|---:|---:|']
        for judge, values in item['judges'].items():
            if values.get('scores'):
                scores = ' | '.join(f"{values['scores'][label]:.2%}" for label in ('v2', 'v3', 'd1'))
                lines.append(f"| {judge} | {scores} | {values['mean']:.2%} | {values['rank']} |")
        if 'truncated_answers' in item:
            lines += ['', f"Answers stopped by the token limit: {item['truncated_answers']}/372. These remain in the evaluation."]
    lines += ['', 'Target: at least 90% on each of v2, v3 and d1 under both judges, first place in the frozen internal comparison, and release checks passed.',
              r['scope'], '', 'Automation: ' + ('two coordinated GPU queues share eight nodes; future datasets prepared alongside evaluation.' if r['pool_enabled'] else 'sequential GPU pipeline; shared-pool upgrade not yet enabled.')]
    if r['detail']:
        lines += ['', r['detail']]
    lines += ['', 'Reports are written every 30 minutes by the remote coordinator. The local project mirror updates while the Mac is awake and can reach Leonardo.']
    return '\n'.join(lines) + '\n'


def write_report(config, state, scheduler, *, force=False, epoch=None):
    directory = Path(config['root']) / 'ctl/autopilot/reports'
    epoch = time.time() if epoch is None else epoch
    with lock(directory / 'report.lock') as acquired:
        if not acquired:
            return None
        latest = directory / 'LATEST.json'
        previous = read_json(latest) if latest.exists() else None
        if not force and previous and epoch - previous['epoch'] < 1800:
            return None
        enriched = {**state, 'decisions': json.loads(json.dumps(state.get('decisions', [])))}
        for decision in enriched['decisions']:
            path = directory.parent / 'decisions' / (decision['run'] + '.json')
            observed = decision.get('best_observed')
            if observed and path.exists():
                full = read_json(path)
                for judge, values in full['leaderboard'][observed['arm']]['judges'].items():
                    observed['judges'][judge]['scores'] = values['scores']
                metrics = full['candidates'][observed['arm']]['metrics']['gpt_oss']
                observed['truncated_answers'] = sum(m['truncated'] for m in metrics.values())
        report = capture(config, enriched, Pool(config, scheduler).live(), previous)
        report['epoch'] = epoch
        name = datetime.fromtimestamp(epoch, timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        atomic(directory / (name + '.json'), report)
        content = render(report)
        (directory / (name + '.md')).write_text(content)
        tmp = directory / 'LATEST.md.tmp'
        tmp.write_text(content)
        tmp.replace(directory / 'LATEST.md')
        atomic(latest, report)
        return report


if __name__ == '__main__':
    config = read_json(sys.argv[1])
    directory = Path(config['root']) / 'ctl/autopilot'
    result = write_report(config, read_json(directory / 'STATE.json'), Slurm(config, directory), force='--force' in sys.argv)
    print(json.dumps({'report': result['at'] if result else 'not due'}))
