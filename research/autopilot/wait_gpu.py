"""Wait for accounting evidence, including jobs already removed from Slurm's live queue."""
from pathlib import Path
import re
import sys
import time

from evidence import EvidenceError, read_json
from supervisor import RETRYABLE, SchedulerError, Slurm, now


FINISHED = RETRYABLE | {'COMPLETED', 'FAILED', 'OUT_OF_MEMORY', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'}


def wait(config_path, job, heartbeat):
    if not re.fullmatch(r'[0-9]+', job):
        raise EvidenceError('Expected a numeric GPU job ID')
    config = read_json(config_path)
    directory = Path(config['root']) / 'ctl/autopilot'
    scheduler = Slurm(config, directory)
    heartbeat = Path(heartbeat)
    while True:
        if (directory / 'STOP').exists():
            raise EvidenceError('STOP is present; GPU allocation was left alone')
        heartbeat.touch()
        try:
            state, exit_code = scheduler.status(job)
        except SchedulerError as exc:
            print(f'{now()} job {job}: {exc}; waiting for accounting', file=sys.stderr, flush=True)
        else:
            if state == 'CANCELLED':
                raise EvidenceError(f'GPU job {job} was cancelled; automatic resubmission is disabled')
            if state in FINISHED:
                print(f'{now()} job {job}: {state} {exit_code}', flush=True)
                return state
            # Unknown, delayed or active accounting is never evidence of completion.
        time.sleep(30)


def controller_block(config, runtime):
    import shlex
    command = shlex.join([config.get('python', sys.executable), str(Path(runtime) / 'wait_gpu.py'),
                         str(Path(config['root']) / 'ctl/autopilot/config.json')])
    return ('wait_job() {\n    ' + command + ' "$1" "$HERE/HEARTBEAT"'
            ' || { log "STOP: GPU job monitoring stopped"; exit 1; }\n}\n')


if __name__ == '__main__':
    wait(sys.argv[1], sys.argv[2], sys.argv[3])
