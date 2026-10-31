"""Bind a frozen result directory to the resolved policy execution settings."""
import fcntl
import json
from pathlib import Path

from miniwam.robotwin_manifest import atomic_json


def bind_frozen_runtime(directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'runtime_identity.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory/'runtime_identity.json'
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError('Frozen ledger policy/runtime identity changed; use a new result directory')
            return
        if any(directory.glob('*__demo_*.jsonl')):
            raise ValueError('Existing frozen results have no recorded runtime identity; use a new result directory')
        atomic_json(path, identity)
