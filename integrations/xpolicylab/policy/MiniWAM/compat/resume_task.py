"""Reuse verified complete task logs while retaining the official scheduler."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

IDENTITY = re.compile(r"^\[MiniWAM\] resume_identity=([0-9a-f]{64})$", re.M)
FINAL = re.compile(r"Final(?: batch)? success rate:\s*(\d+)\s*/\s*(\d+)", re.I)
COMPLETE = "[MiniWAM] task_completed=true"
CHECKPOINT = re.compile(r"^\[MiniWAM\] checkpoint_sha256=([0-9a-f]{64})$", re.M)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reusable(text: str, identity: str, trials: int) -> bool:
    identities = IDENTITY.findall(text)
    final = FINAL.findall(text)
    return (
        set(identities) == {identity}
        and COMPLETE in text.splitlines()
        and len(final) == 1
        and int(final[0][1]) == trials
        and 0 <= int(final[0][0]) <= trials
    )


def main(argv: list[str]) -> int:
    if os.environ.get('MINIWAM_CASE_MODE') == 'replay':
        # Frozen replay owns exact episode-level reuse; old task-log reuse must
        # never bypass manifest/cache verification.
        return 10
    policy = Path(__file__).resolve().parents[1]
    args_file = os.environ.get("ROBOTWIN_EVAL_ARGS_FILE")
    source = os.environ.get("MINIWAM_EVAL_RESUME_FROM", "")
    if not args_file:
        if source:
            raise ValueError("Resume requires the official multitask entrypoint.")
        return 10
    args_path = Path(args_file).resolve()
    previous = Path(source).expanduser().resolve() if source not in {"", "auto", "off"} else None
    if previous and not (previous / "logs").is_dir():
        raise ValueError(f"Resume run has no logs directory: {previous}")
    if previous == args_path.parent.parent:
        raise ValueError("Resume source must differ from the new run directory.")

    # Reuse the launcher's artifact resolver, including explicit checkpoint paths.
    resolved = subprocess.check_output(
        ["bash", str(policy / "setup_eval_policy_server.sh"),
         *argv[:6], argv[6], argv[8], "0", "localhost"],
        env={**os.environ, "MINIWAM_RESOLVE_ONLY": "1"}, text=True,
    )
    hashes = re.findall(r"\[MiniWAM\] (\w+_sha256)=([0-9a-f]{64})", resolved)
    if len(hashes) != 2:
        raise ValueError("Cannot fingerprint checkpoint and statistics.")
    checkpoint_hash = dict(hashes)["checkpoint_sha256"]
    if previous:
        recorded = set()
        for log in (previous / "logs").glob("*.log"):
            recorded.update(CHECKPOINT.findall(log.read_text()))
        if recorded != {checkpoint_hash}:
            raise ValueError(
                f"Explicit resume checkpoint mismatch: current={checkpoint_hash}, "
                f"recorded={sorted(recorded)}. Refusing this resume source."
            )
    # The official scheduler runs policy/eval.sh with RoboTwin as its cwd.
    root = Path.cwd()
    overrides = args_path.read_text().splitlines()
    settings = dict(zip(overrides[::2], overrides[1::2]))
    trials = int(settings.get("--test_num", "100"))
    checkpoint = Path(re.search(r"checkpoint_path=(.*)", resolved)[1])
    files = [policy / "deploy.yml", checkpoint / "config.yaml"]
    files += list((root / "env_cfg").rglob("*.yml"))
    files += list((root / "env_cfg").rglob("*.yaml"))
    files += list(policy.glob("*.py")) + list(policy.glob("*.sh"))
    files += list((policy / "compat").glob("*.py"))
    files += list((policy.parents[3] / "deploy").glob("*robotwin*.py"))
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    try:
        gpu = subprocess.check_output([
            "nvidia-smi", f"--id={argv[7]}", "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader",
        ], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = "unavailable"
    payload = {
        "schema": 1, "job": argv[:6], "overrides": overrides,
        "artifacts": hashes, "robotwin_revision": revision,
        "simulator_gpu": gpu, "conda_environments": argv[8:],
        "files": {str(p.resolve()): digest(p) for p in files if p.is_file()},
        "environment": {k: v for k, v in os.environ.items()
                        if k.startswith("MINIWAM_") and k not in {
                            "MINIWAM_EVAL_RESUME_FROM", "MINIWAM_EVAL_INACTIVITY_TIMEOUT_S",
                            "MINIWAM_EXPECTED_CHECKPOINT_SHA256", "MINIWAM_CHECKPOINT_PATH"}},
    }
    payload = {**payload, "job": [*argv[:2], checkpoint_hash, *argv[3:6]]}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    runs = [previous] if previous else (
        sorted(args_path.parents[2].iterdir(), reverse=True) if source != "off" else []
    )
    candidates = [log for run in runs if run != args_path.parent.parent
                  for log in sorted((run / "logs").glob(f"*_{argv[1]}_seed{argv[5]}.log"))]
    for old in candidates:
        text = old.read_text()
        if set(CHECKPOINT.findall(text)) != {checkpoint_hash}:
            continue
        if reusable(text, identity, trials):
            print(f"[MiniWAM][RESUME] Reusing completed {argv[1]} from {old}", flush=True)
            print(f"[MiniWAM] resumed_from={old}")
            print(f"[MiniWAM] resume_identity={identity}")
            print(resolved, end="")
            successes, count = FINAL.findall(text)[0]
            print(f"Final success rate: {successes}/{count}")
            print(COMPLETE, flush=True)
            return 0
    print("[MiniWAM][RESUME] No compatible completed task found; evaluating.", flush=True)
    print(f"[MiniWAM] resume_identity={identity}", flush=True)
    return 10


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
