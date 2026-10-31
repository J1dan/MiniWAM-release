"""RoboTwin 2.0 data and action-schema contracts owned by MiniWAM."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .proprio import PROPRIO_ZSCORE, validate_proprio_statistics
from .statistics import ACTION_ZSCORE, get_action_zscore


ROBOTWIN_DATASET_KEY = "robotwin2_demo_clean_50x50"
ROBOTWIN_FULL_ACTION_CHUNK = "action_horizon"
ROBOTWIN_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)
ROBOTWIN_EPISODE_IDS = tuple(f"{index:07d}" for index in range(50))
ROBOTWIN_ACTION_DIM_INFO = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
ROBOTWIN_CAMERA_KEYS = (
    "cam_head",
    "cam_left_wrist",
    "cam_right_wrist",
)

_EPISODE_FILE = re.compile(r"episode_?(\d+)\.(?:hdf5|h5)$", re.IGNORECASE)


@dataclass(frozen=True)
class RoboTwinEpisode:
    task: str
    episode_id: str
    path: Path


class RoboTwinSplitError(ValueError):
    """Fixed-split validation failure carrying the audit built before abort."""

    def __init__(self, message: str, audit: Mapping[str, Any]):
        super().__init__(message)
        self.audit = dict(audit)


def resolve_robotwin_replan_steps(action_horizon: int, requested: Any) -> int:
    """Resolve RoboTwin chunk execution without duplicating the model horizon.

    ``None`` and the deployment sentinel ``"action_horizon"`` both execute the
    complete checkpoint-defined action chunk. A positive integer remains an
    explicit ablation override.
    """
    horizon = int(action_horizon)
    if horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    if requested is None or requested == ROBOTWIN_FULL_ACTION_CHUNK:
        return horizon
    if isinstance(requested, bool):
        raise ValueError("replan_steps must be an integer or 'action_horizon'.")
    try:
        resolved = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "replan_steps must be an integer or 'action_horizon', got "
            f"{requested!r}."
        ) from exc
    if not 1 <= resolved <= horizon:
        raise ValueError(f"replan_steps must be in [1, {horizon}], got {resolved}.")
    return resolved


def assert_robotwin_joint_contract(
    *,
    action_type: str,
    robot_action_dim_info: Mapping[str, Any],
    action_dim: int,
    proprio_dim: int,
    action_representation: str,
    action_transform: Any = None,
    delta_action_mask: Any = None,
) -> None:
    """Assert provenance/schema fields instead of guessing deltas numerically."""
    def is_unconfigured(value: Any) -> bool:
        return value is None or (isinstance(value, str) and value == "none") or (
            isinstance(value, (list, tuple)) and len(value) == 0
        )

    errors = []
    if action_type != "joint":
        errors.append(f"action_type={action_type!r}, expected 'joint'")
    arm_dim = list(robot_action_dim_info.get("arm_dim", ()))
    ee_dim = list(robot_action_dim_info.get("ee_dim", ()))
    if arm_dim != [6, 6] or ee_dim != [1, 1]:
        errors.append(f"robot_action_dim_info={{'arm_dim': {arm_dim}, 'ee_dim': {ee_dim}}}")
    if int(action_dim) != 14 or int(proprio_dim) != 14:
        errors.append(f"action_dim/proprio_dim={action_dim}/{proprio_dim}, expected 14/14")
    if action_representation != "absolute_qpos":
        errors.append(
            f"action_representation={action_representation!r}, expected 'absolute_qpos'"
        )
    if not is_unconfigured(action_transform):
        errors.append(f"action_transform={action_transform!r}, expected none")
    if not is_unconfigured(delta_action_mask):
        errors.append(f"delta_action_mask={delta_action_mask!r}, expected none")
    if errors:
        raise ValueError("Invalid RoboTwin absolute joint-qpos contract: " + "; ".join(errors))


def validate_robotwin_training_statistics(stats: Mapping[str, Any]) -> None:
    """Require global per-dimension z-score statistics for 14-D joint actions."""
    assert_robotwin_joint_contract(
        action_type=stats.get("action_type"),
        robot_action_dim_info=stats.get("robot_action_dim_info", {}),
        action_dim=14,
        proprio_dim=stats.get("proprio", {}).get("dim", -1),
        action_representation=stats.get("action_representation"),
        action_transform=stats.get("action_transform"),
        delta_action_mask=stats.get("delta_action_mask"),
    )
    mean, std, _constant, epsilon, clip, mode = get_action_zscore(stats)
    if mode != ACTION_ZSCORE or mean.shape != (14,) or std.shape != (14,):
        raise ValueError("RoboTwin action mean/std must be 14-D zscore_clip statistics.")
    if epsilon != 1e-8 or clip != 5.0:
        raise ValueError(f"RoboTwin action epsilon/clip must be 1e-8/5, got {epsilon}/{clip}.")
    proprio = validate_proprio_statistics(stats.get("proprio", {}), expected_dim=14)
    if proprio["normalization"] != PROPRIO_ZSCORE:
        raise ValueError("RoboTwin proprio statistics must use zscore_clip.")
    if proprio.get("epsilon") != 1e-8 or proprio["clip"] != 5.0:
        raise ValueError(
            "RoboTwin proprio epsilon/clip must be 1e-8/5, got "
            f"{proprio.get('epsilon')}/{proprio['clip']}."
        )
    absolute_mask = np.asarray(stats.get("absolute_action_mask"), dtype=bool)
    if absolute_mask.shape != (14,) or not absolute_mask.all():
        raise ValueError("Every RoboTwin joint-qpos action dimension must be marked absolute.")
    aggregation = stats.get("aggregation", {})
    if int(aggregation.get("tasks", -1)) != 50 or int(aggregation.get("episodes", -1)) != 2500:
        raise ValueError("RoboTwin statistics must aggregate exactly 50 tasks and 2,500 episodes.")
    source_hash = str(stats.get("source_manifest_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("RoboTwin statistics require a 64-character source manifest SHA256.")


def assert_robotwin_data_config(data_config: Mapping[str, Any]) -> None:
    """Validate the declared split/normalization provenance before training or eval."""
    if data_config.get("dataset_name") != ROBOTWIN_DATASET_KEY:
        raise ValueError(f"RoboTwin dataset_name must be {ROBOTWIN_DATASET_KEY!r}.")
    normalization = data_config.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("RoboTwin data config requires an action/proprio normalization mapping.")
    for field in ("action", "proprio"):
        block = normalization.get(field)
        expected = {
            "type": ACTION_ZSCORE,
            "scope": "global_exact_training_split",
            "clip": 5.0,
            "epsilon": 1e-8,
        }
        if not isinstance(block, Mapping):
            raise ValueError(f"RoboTwin normalization.{field} must be a mapping.")
        mismatches = {
            key: (block.get(key), value)
            for key, value in expected.items()
            if block.get(key) != value
        }
        if mismatches:
            raise ValueError(f"RoboTwin normalization.{field} mismatch: {mismatches}")
    split = data_config.get("split")
    expected_split = {
        "tasks": 50,
        "episodes_per_task": 50,
        "episode_ids": "0000000-0000049",
        "source": "demo_clean",
    }
    if not isinstance(split, Mapping) or any(
        split.get(key) != value for key, value in expected_split.items()
    ):
        raise ValueError(f"RoboTwin split must be exactly {expected_split}, got {split!r}.")


def _task_data_dir(data_root: Path, task: str) -> Path:
    candidates = (
        data_root / "demo_clean" / task / "aloha_agilex" / "data",
        data_root / task / "aloha_agilex" / "data",
    )
    existing = [path for path in candidates if path.is_dir()]
    if len(existing) > 1:
        raise ValueError(f"Ambiguous RoboTwin data roots for {task}: {existing}")
    return existing[0] if existing else candidates[0]


def select_training_episodes(
    data_root: str | Path,
    tasks: Sequence[str] = ROBOTWIN_TASKS,
) -> tuple[list[RoboTwinEpisode], dict[str, Any]]:
    """Select exactly IDs 0000000..0000049 for every requested task.

    Missing selected IDs and multiple files resolving to a selected ID are
    fatal. IDs outside the fixed split are omitted and preserved in the audit.
    """
    root = Path(data_root).expanduser().resolve()
    tasks = tuple(str(task) for task in tasks)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("RoboTwin task selection must be non-empty and duplicate-free.")
    unknown = sorted(set(tasks) - set(ROBOTWIN_TASKS))
    if unknown:
        raise ValueError(f"Unknown RoboTwin 2.0 leaderboard tasks: {unknown}")

    selected: list[RoboTwinEpisode] = []
    task_audits: dict[str, Any] = {}
    expected = set(ROBOTWIN_EPISODE_IDS)
    failures = []
    manifest_entries = []
    for task in tasks:
        data_dir = _task_data_dir(root, task)
        by_id: dict[str, list[Path]] = {}
        ignored_files = []
        if data_dir.is_dir():
            for path in sorted(data_dir.iterdir()):
                if not path.is_file():
                    continue
                match = _EPISODE_FILE.fullmatch(path.name)
                if match is None:
                    ignored_files.append(path.name)
                    continue
                episode_id = f"{int(match.group(1)):07d}"
                by_id.setdefault(episode_id, []).append(path)
        missing = sorted(expected - set(by_id))
        duplicates = {
            episode_id: [str(path) for path in paths]
            for episode_id, paths in sorted(by_id.items())
            if episode_id in expected and len(paths) != 1
        }
        excluded = {
            episode_id: [str(path) for path in paths]
            for episode_id, paths in sorted(by_id.items())
            if episode_id not in expected
        }
        if missing:
            failures.append(f"{task}: missing selected IDs {missing}")
        if duplicates:
            failures.append(f"{task}: duplicate selected IDs {duplicates}")
        task_audits[task] = {
            "data_dir": str(data_dir),
            "selected_episode_ids": sorted(
                episode_id
                for episode_id, paths in by_id.items()
                if episode_id in expected and len(paths) == 1
            ),
            "missing_selected_episode_ids": missing,
            "duplicate_selected_episode_ids": duplicates,
            "excluded_additional_episode_ids": excluded,
            "ignored_non_episode_files": ignored_files,
        }
        if not missing and not duplicates:
            for episode_id in ROBOTWIN_EPISODE_IDS:
                path = by_id[episode_id][0]
                selected.append(RoboTwinEpisode(task, episode_id, path))
                stat = path.stat()
                manifest_entries.append({
                    "task": task,
                    "episode_id": episode_id,
                    "relative_path": str(path.relative_to(root)),
                    "size": int(stat.st_size),
                })

    manifest_json = json.dumps(manifest_entries, sort_keys=True, separators=(",", ":"))
    audit = {
        "schema_version": 1,
        "dataset_key": ROBOTWIN_DATASET_KEY,
        "source_root": str(root),
        "requested_tasks": list(tasks),
        "required_episode_ids": list(ROBOTWIN_EPISODE_IDS),
        "selected_episode_count": len(selected),
        "source_manifest_sha256": hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
        "selection_note": (
            "The official downloader selects task archives, not episodes inside an archive; "
            "additional episode IDs are therefore downloaded, excluded here, and audited."
        ),
        "tasks": task_audits,
    }
    if failures:
        raise RoboTwinSplitError(
            "Invalid fixed RoboTwin training split:\n" + "\n".join(failures), audit,
        )
    expected_count = len(tasks) * len(ROBOTWIN_EPISODE_IDS)
    if len(selected) != expected_count:
        raise RuntimeError(f"Selected {len(selected)} episodes, expected exactly {expected_count}.")
    return selected, audit
