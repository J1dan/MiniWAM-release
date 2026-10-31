"""Validate isolated workers and pin one immutable checkpoint per schedule."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def validate_workers(settings):
    if int(settings.get('--num_workers', '1')) != 1 or str(
        settings.get('--eval_batch', 'false')
    ).lower() not in {'false', '0'}:
        raise ValueError('MiniWAM requires num_workers: 1 and eval_batch: false. '
                         'Use jobs_per_gpu or additional GPUs for isolated parallel task workers.')


def main(args):
    args_file = os.environ.get('ROBOTWIN_EVAL_ARGS_FILE')
    if not args_file:
        raise ValueError('Use the official scripts/eval_policy.sh multitask entrypoint.')
    path = Path(args_file).resolve()
    lines = path.read_text().splitlines()
    validate_workers(dict(zip(lines[::2], lines[1::2])))
    # Validate before checkpoint resolution or task-log resume can bypass execution.
    plan = json.loads((path.parents[1]/'schedule_plan.json').read_text())
    protocol_config = plan.get('miniwam') or {}
    validation = protocol_config.get('benchmark_validation') or {}
    root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(root))
    from scripts.eval.robotwin_protocol import sha256
    if (validation.get('benchmark_protocol') != 'official_split_v1' or
            validation.get('validator_sha256') != sha256(root/'scripts/eval/robotwin_protocol.py')):
        raise ValueError('Schedule lacks current official/frozen protocol validation; relaunch with the current multitask scheduler')
    if protocol_config.get('evaluation_protocol') != os.environ.get('MINIWAM_EVAL_PROTOCOL'):
        raise ValueError('Worker protocol differs from its validated schedule')
    manifest = path.parents[1] / 'miniwam_checkpoint.json'
    request = [args[2], os.environ.get('MINIWAM_CHECKPOINT_PATH', ''),
               os.environ.get('MINIWAM_CKPT_ROOT', '')]
    with (manifest.parent / 'miniwam_checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            pinned = json.loads(manifest.read_text())
            if pinned['request'] != request:
                raise ValueError('Checkpoint request changed within this schedule.')
        else:
            policy = Path(__file__).resolve().parents[1]
            output = subprocess.check_output(
                ['bash', str(policy / 'setup_eval_policy_server.sh'),
                 *args[:6], args[6], args[8], '0', 'localhost'],
                env={**os.environ, 'MINIWAM_RESOLVE_ONLY': '1'}, text=True)
            pinned = {'request': request,
                      'path': re.search(r'checkpoint_path=(.*)', output)[1],
                      'sha256': re.search(r'checkpoint_sha256=(.*)', output)[1]}
            temporary = manifest.with_suffix('.tmp')
            temporary.write_text(json.dumps(pinned, indent=2) + '\n')
            temporary.replace(manifest)
    print(pinned['path'])
    print(pinned['sha256'])


if __name__ == '__main__':
    main(sys.argv[1:])
