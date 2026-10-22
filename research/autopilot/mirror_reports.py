"""Local launchd entrypoint: request a due read-only report and mirror the timestamped project files."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    marker = args.destination / 'SYNC.json'
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    remote = args.root + '/ctl/autopilot'
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host]
    try:
        subprocess.run([*ssh, shlex.join([args.python, remote + '/runtime/reports.py', remote + '/config.json'])],
                       check=True, timeout=90)
        subprocess.run(['rsync', '-az', '--timeout=45', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                        args.host + ':' + remote + '/reports/', str(args.destination) + '/'], check=True, timeout=90)
        result = {'status': 'synced', 'at': stamp, 'last_success': stamp, 'interval_seconds': 1800,
                  'note': 'Remote jobs continue when this Mac is asleep. Local mirroring resumes on wake.'}
    except (OSError, subprocess.SubprocessError) as exc:
        result = {'status': 'unreachable', 'at': stamp, 'last_success': previous.get('last_success'), 'error': str(exc),
                  'note': 'Existing local reports are historical snapshots; no fresh remote state was retrieved.'}
    tmp = marker.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(marker)
    print(json.dumps(result))
    return 0 if result['status'] == 'synced' else 1


if __name__ == '__main__':
    raise SystemExit(main())
