"""Durable recovery helpers for per-trial LIBERO evaluation results."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable


TRIAL_PROGRESS_SCHEMA_VERSION = 1
TRIAL_PROGRESS_FILENAME = "trial_progress.json"
_VIDEO_RESULT_RE = re.compile(
    r"--episode=(?P<trial_idx>\d+)--success=(?P<success>True|False)--"
)


def _resolved_path(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def atomic_write_json(path: Path, payload: Any) -> None:
    """Replace *path* atomically after flushing a complete JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    """Replace *path* atomically after flushing a complete text document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _identity(
    task_suite_name: str,
    task_id: int,
    checkpoint: str | Path,
    num_trials: int,
    base_seed: int,
) -> dict[str, Any]:
    return {
        "task_suite_name": str(task_suite_name),
        "task_id": int(task_id),
        "checkpoint": _resolved_path(checkpoint),
        "num_trials": int(num_trials),
        "base_seed": int(base_seed),
    }


def _record_is_valid(record: Any, num_trials: int, base_seed: int) -> bool:
    if not isinstance(record, dict):
        return False
    trial_idx = record.get("trial_idx")
    success = record.get("success")
    seed = record.get("seed")
    return (
        isinstance(trial_idx, int)
        and 0 <= trial_idx < num_trials
        and isinstance(success, bool)
        and seed == base_seed + trial_idx
    )


def _records_by_id(
    records: Iterable[Any], num_trials: int, base_seed: int,
) -> dict[int, dict[str, Any]]:
    recovered: dict[int, dict[str, Any]] = {}
    for record in records:
        if _record_is_valid(record, num_trials, base_seed):
            recovered[int(record["trial_idx"])] = dict(record)
    return recovered


def discover_video_results(
    video_dir: Path, num_trials: int, base_seed: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[Path]]]:
    """Recover the latest completed result for each trial from its rollout video name."""
    candidates: dict[int, list[Path]] = {}
    if not video_dir.is_dir():
        return {}, candidates
    for video_path in video_dir.glob("*.mp4"):
        match = _VIDEO_RESULT_RE.search(video_path.name)
        if match is None or video_path.stat().st_size <= 0:
            continue
        trial_idx = int(match.group("trial_idx"))
        if 0 <= trial_idx < num_trials:
            candidates.setdefault(trial_idx, []).append(video_path)

    recovered: dict[int, dict[str, Any]] = {}
    for trial_idx, paths in candidates.items():
        latest = max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))
        match = _VIDEO_RESULT_RE.search(latest.name)
        recovered[trial_idx] = {
            "trial_idx": trial_idx,
            "success": match.group("success") == "True",
            "seed": base_seed + trial_idx,
            "video": str(latest.resolve()),
            "source": "video",
        }
    return recovered, candidates


def _protocol_matches(recorded: Any, required: dict[str, Any] | None) -> bool:
    """Require every field of *required* to be recorded with the same value."""
    if required is None:
        return True
    if not isinstance(recorded, dict):
        return False
    return all(recorded.get(key) == value for key, value in required.items())


def _identity_matches(payload: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    for key, value in expected.items():
        candidate = payload.get(key)
        if key == "checkpoint" and candidate is not None:
            try:
                candidate = _resolved_path(candidate)
            except (TypeError, ValueError):
                return False
        if candidate != value:
            return False
    return True


def load_recoverable_trials(
    task_output_dir: Path,
    *,
    task_suite_name: str,
    task_id: int,
    checkpoint: str | Path,
    num_trials: int,
    base_seed: int,
    required_protocol: dict[str, Any] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Load durable progress and fill missing IDs from existing rollout videos.

    When *required_protocol* is given, progress recorded under a different
    evaluation protocol is ignored, and rollout videos (which carry no protocol
    record) are imported only next to compatible progress.
    """
    task_output_dir = Path(task_output_dir)
    expected = _identity(task_suite_name, task_id, checkpoint, num_trials, base_seed)
    progress_path = task_output_dir / TRIAL_PROGRESS_FILENAME
    progress: dict[str, Any] | None = None
    if progress_path.is_file():
        try:
            candidate = json.loads(progress_path.read_text(encoding="utf-8"))
            if (
                candidate.get("schema_version") == TRIAL_PROGRESS_SCHEMA_VERSION
                and _identity_matches(candidate, expected)
                and _protocol_matches(candidate.get("evaluation_protocol"), required_protocol)
            ):
                progress = candidate
        except (OSError, json.JSONDecodeError, AttributeError):
            progress = None

    records = _records_by_id(
        progress.get("trials", []) if progress else [], num_trials, base_seed,
    )
    from_videos, duplicate_candidates = discover_video_results(
        task_output_dir / "videos", num_trials, base_seed,
    )
    videos_allowed = required_protocol is None or progress is not None
    if videos_allowed:
        for trial_idx, record in from_videos.items():
            records.setdefault(trial_idx, record)

    duplicate_ids = sorted(
        trial_idx for trial_idx, paths in duplicate_candidates.items() if len(paths) > 1
    )
    metadata = {
        "imported_video_trials": sum(
            record.get("source") == "video" for record in records.values()
        ),
        "duplicate_trial_ids": duplicate_ids,
        "ignored_incompatible_progress": progress is None and progress_path.exists(),
        "skipped_video_trials": 0 if videos_allowed else len(from_videos),
    }
    return records, metadata


def quarantine_superseded_results(task_output_dir: Path) -> Path | None:
    """Move results that will not be resumed out of the way before new progress is written.

    When a task folder holds progress or rollout videos from another evaluation
    protocol (or videos without any progress record), leaving them in place would
    let a later resume import those videos next to the new, compatible progress.
    The old ``videos/``, ``trial_progress.json`` and ``summary.json`` are moved
    into ``superseded_<timestamp>/`` inside the task folder, and an empty
    ``videos/`` is left for the new run's rollouts.
    """
    task_output_dir = Path(task_output_dir)
    candidates = [
        task_output_dir / "videos",
        task_output_dir / TRIAL_PROGRESS_FILENAME,
        task_output_dir / "summary.json",
    ]
    present = [path for path in candidates if path.exists()]
    if not present:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = task_output_dir / f"superseded_{stamp}"
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = task_output_dir / f"superseded_{stamp}-{suffix}"
    destination.mkdir(parents=True)
    for path in present:
        os.replace(path, destination / path.name)
    # The evaluator created videos/ before recovery and writes rollouts into it.
    (task_output_dir / "videos").mkdir(exist_ok=True)
    return destination


def write_trial_progress(
    task_output_dir: Path,
    records: dict[int, dict[str, Any]],
    *,
    task_suite_name: str,
    task_id: int,
    checkpoint: str | Path,
    num_trials: int,
    base_seed: int,
    evaluation_protocol: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "schema_version": TRIAL_PROGRESS_SCHEMA_VERSION,
        **_identity(task_suite_name, task_id, checkpoint, num_trials, base_seed),
        "updated_at": time.time(),
        "evaluation_protocol": dict(evaluation_protocol or {}),
        "trials": [records[index] for index in sorted(records)],
    }
    path = Path(task_output_dir) / TRIAL_PROGRESS_FILENAME
    atomic_write_json(path, payload)
    return path


def validate_complete_summary(
    summary_path: Path,
    *,
    task_suite_name: str,
    task_id: int,
    checkpoint: str | Path,
    num_trials: int,
    base_seed: int,
    required_protocol: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a summary only when it exactly covers the requested trial IDs."""
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        return None, "summary.json is missing"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"summary.json is unreadable: {exc}"
    expected = _identity(task_suite_name, task_id, checkpoint, num_trials, base_seed)
    if not _identity_matches(summary, {k: v for k, v in expected.items() if k != "base_seed"}):
        return None, "summary identity does not match this evaluation"
    if not _protocol_matches(summary.get("evaluation_protocol"), required_protocol):
        return None, "summary evaluation protocol does not match this evaluation"
    records = _records_by_id(summary.get("trials", []), num_trials, base_seed)
    expected_ids = set(range(num_trials))
    if set(records) != expected_ids:
        return None, f"summary covers {len(records)}/{num_trials} unique trials"
    success_count = sum(record["success"] for record in records.values())
    if summary.get("success_count") != success_count:
        return None, "summary success_count is inconsistent with trials"
    expected_rate = 100.0 * success_count / num_trials
    if abs(float(summary.get("success_rate", -1.0)) - expected_rate) > 1e-9:
        return None, "summary success_rate is inconsistent with trials"
    return summary, None


def summary_to_result(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite": summary["task_suite_name"],
        "task_id": int(summary["task_id"]),
        "description": summary.get("task_description", f"task_{summary['task_id']}"),
        "success_rate": float(summary["success_rate"]),
        "success_count": int(summary["success_count"]),
        "status": "success",
    }
