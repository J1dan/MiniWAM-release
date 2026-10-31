#!/usr/bin/env python3
"""Extract the fixed RoboTwin 2.0 50x50 co-training split for MiniWAM.

Images are decoded only with XPolicyLab's ``decode_image_bit`` and passed as
HWC uint8 RGB to the selected MiniWAM DINOv3 or Wan-VAE preprocessing boundary.
Robot state/action packing is delegated to XPolicyLab for both offline
``source_type='dataset'`` modes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from tqdm import tqdm


MINIWAM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MINIWAM_ROOT))

from scripts.data.extraction_utils import (  # noqa: E402
    VAE_TEMPORAL_ENCODING,
    encode_dinov3_frames,
    encode_rolling_vae_latents,
    encode_text,
    load_vae,
    load_text_encoder,
    stack_slot_latents,
    stack_slot_dino_features,
    validate_multiview_episode,
)
from miniwam.action_normalization import normalize_actions  # noqa: E402
from miniwam.image_preprocessing import (  # noqa: E402
    dino_rgb_uint8_to_bchw,
    wan_vae_rgb_uint8_to_bchw,
)
from miniwam.models.dinov3 import DINOv3FrameEncoder  # noqa: E402
from miniwam.proprio import (  # noqa: E402
    PROPRIO_ZSCORE,
    compute_proprio_statistics,
    make_stats_id,
)
from miniwam.robotwin import (  # noqa: E402
    ROBOTWIN_ACTION_DIM_INFO,
    ROBOTWIN_CAMERA_KEYS,
    ROBOTWIN_DATASET_KEY,
    ROBOTWIN_TASKS,
    RoboTwinEpisode,
    RoboTwinSplitError,
    assert_robotwin_joint_contract,
    select_training_episodes,
    validate_robotwin_training_statistics,
)
from miniwam.statistics import ACTION_ZSCORE, compute_action_statistics  # noqa: E402
from miniwam.robotwin_features import (  # noqa: E402
    json_sha256,
    text_feature_contract,
    wan_vae_feature_contract,
)


LOGGER = logging.getLogger("extract_robotwin")
ACTION_DIM = 14
PROPRIO_DIM = 14
ACTION_TYPE = "joint"
ACTION_REPRESENTATION = "absolute_qpos"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotwin-root", default="RoboTwin",
        help="RoboTwin checkout containing data/ and its XPolicyLab submodule.",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Override the downloader output root (default: <robotwin-root>/data).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--visual-encoder", choices=("dinov3", "wan_vae"), default="dinov3",
        help="Offline visual feature backend. Use a separate output root per backend.",
    )
    parser.add_argument(
        "--vae-path", default="pretrained/vision/Wan2.1_VAE.pth",
        help="Wan 2.1 VAE checkpoint used only with --visual-encoder wan_vae.",
    )
    parser.add_argument(
        "--future-rgb-horizon", type=int, default=16,
        help="Wan-VAE joint current/future window length; must be divisible by four.",
    )
    parser.add_argument(
        "--per-view-resolution", type=int, nargs=2, default=(224, 224), metavar=("H", "W"),
        help="Wan-VAE resize resolution. DINOv3 uses its frozen processor instead.",
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Validate IDs and write training_split_audit.json without loading HDF5/model assets.",
    )
    parser.add_argument(
        "--dinov3-model-id", default="facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    parser.add_argument(
        "--dinov3-revision", default="5931719e67bbdb9737e363e781fb0c67687896bc",
    )
    parser.add_argument(
        "--text-encoder-path",
        default="pretrained/language/models_t5_umt5-xxl-enc-bf16.pth",
    )
    parser.add_argument("--tokenizer-id", default="google/umt5-xxl")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Encoding microbatch (default: DINOv3=64, Wan-VAE=8).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--shard-id", "--shard_id", type=int, default=None,
        help=(
            "Extraction shard in [0, num_shards). Defaults to RANK under "
            "torchrun, otherwise 0."
        ),
    )
    parser.add_argument(
        "--num-shards", "--num_shards", type=int, default=None,
        help=(
            "Number of deterministic episode shards. Defaults to WORLD_SIZE under "
            "torchrun, otherwise 1."
        ),
    )
    parser.add_argument(
        "--stats-wait-timeout", type=float, default=1800.0,
        help="Seconds nonzero shards wait for shard 0 to publish global statistics.",
    )
    parser.add_argument("--text-dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument('--verify-existing-text', action='store_true',
                        help='Re-encode and compare all resumed text tensors before binding their provenance.')
    return parser.parse_args()


def _install_xpolicylab_import(robotwin_root: Path):
    xpl_root = robotwin_root / "XPolicyLab"
    if not (xpl_root / "utils" / "process_data.py").is_file():
        raise FileNotFoundError(
            f"XPolicyLab submodule is unavailable at {xpl_root}. "
            "Clone RoboTwin with --recurse-submodules or initialize the submodule."
        )
    sys.path.insert(0, str(robotwin_root))
    from XPolicyLab.utils.data_loader import load_xspark_v1
    from XPolicyLab.utils.process_data import decode_image_bit, pack_robot_state

    return load_xspark_v1, decode_image_bit, pack_robot_state


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _pickle_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def _resolve_sharding(args: argparse.Namespace) -> tuple[int, int]:
    shard_id = int(
        args.shard_id
        if args.shard_id is not None
        else os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))
    )
    num_shards = int(
        args.num_shards if args.num_shards is not None else os.environ.get("WORLD_SIZE", 1)
    )
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= shard_id < num_shards:
        raise ValueError(
            f"--shard-id must be in [0, {num_shards}), got {shard_id}."
        )
    return shard_id, num_shards


def _shard_bounds(total: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    chunk_size = math.ceil(total / num_shards)
    start = min(shard_id * chunk_size, total)
    end = min(start + chunk_size, total)
    return start, end


def _shard_audit_path(output_dir: Path, shard_id: int, num_shards: int) -> Path:
    return output_dir / f"extraction_audit.shard_{shard_id:05d}_of_{num_shards:05d}.json"


def _first_instruction(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped[:1] in "[{":
            try:
                nested = _first_instruction(json.loads(stripped))
                if nested:
                    return nested
            except json.JSONDecodeError:
                pass
        return stripped
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Mapping):
        for key in ("instruction", "instructions", "seen", "unseen", "task_instruction"):
            if key in value:
                result = _first_instruction(value[key])
                if result:
                    return result
        for nested in value.values():
            result = _first_instruction(nested)
            if result:
                return result
        return None
    if isinstance(value, (list, tuple)):
        for nested in value:
            result = _first_instruction(nested)
            if result:
                return result
    return None


def _load_instruction(raw_episode: Mapping[str, Any], episode: RoboTwinEpisode) -> str:
    for key in ("instruction", "instructions", "task_instruction"):
        if key in raw_episode:
            instruction = _first_instruction(raw_episode[key])
            if instruction:
                return instruction
    instruction_path = (
        episode.path.parent.parent / "instruction" / f"episode_{episode.episode_id}.json"
    )
    if instruction_path.is_file():
        instruction = _first_instruction(json.loads(instruction_path.read_text(encoding="utf-8")))
        if instruction:
            return instruction
    raise KeyError(
        f"{episode.path} has no usable HDF5 instruction and {instruction_path} is absent/empty."
    )


def _pack_episode(raw_episode, pack_robot_state, source: Path) -> tuple[np.ndarray, np.ndarray]:
    # XPolicyLab deliberately uses plural dataset keys in both calls.
    proprio = pack_robot_state(
        raw_episode,
        action_type=ACTION_TYPE,
        robot_action_dim_info=ROBOTWIN_ACTION_DIM_INFO,
        source_type="dataset",
        state_type="state",
    )
    actions = pack_robot_state(
        raw_episode,
        action_type=ACTION_TYPE,
        robot_action_dim_info=ROBOTWIN_ACTION_DIM_INFO,
        source_type="dataset",
        state_type="action",
    )
    proprio = np.asarray(proprio, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    for name, array, dim in (
        ("state", proprio, PROPRIO_DIM), ("action", actions, ACTION_DIM),
    ):
        if array.ndim != 2 or array.shape[1] != dim or not np.isfinite(array).all():
            raise ValueError(f"{source}: packed {name} must be finite [T,{dim}], got {array.shape}.")
    if len(proprio) != len(actions) or len(actions) == 0:
        raise ValueError(
            f"{source}: state/action temporal mismatch {len(proprio)} vs {len(actions)}."
        )
    return proprio, actions


def _camera_rgb(raw_episode, camera: str, decode_image_bit, source: Path) -> np.ndarray:
    try:
        encoded = raw_episode["vision"][camera]["colors"]
    except KeyError as exc:
        raise KeyError(f"{source} is missing vision/{camera}/colors.") from exc
    # This is the only offline image decoder. Its return is already RGB.
    rgb = np.asarray(decode_image_bit(encoded))
    dino_rgb_uint8_to_bchw(rgb, source=f"{source}:vision/{camera}/colors")
    return rgb


def compute_statistics(episodes, load_xspark_v1, pack_robot_state, audit):
    all_actions, all_proprio = [], []
    for episode in tqdm(episodes, desc="RoboTwin global action/state statistics"):
        raw = load_xspark_v1(str(episode.path), decode_images=False)
        proprio, actions = _pack_episode(raw, pack_robot_state, episode.path)
        all_actions.append(actions)
        all_proprio.append(proprio)
    action_values = np.concatenate(all_actions, axis=0)
    proprio_values = np.concatenate(all_proprio, axis=0)
    action_stats = compute_action_statistics(action_values, normalization=ACTION_ZSCORE, clip=5.0)
    proprio_stats = compute_proprio_statistics(
        proprio_values,
        representation="left6_left_gripper_right6_right_gripper",
        coordinate_frame="robot_joint_space",
        normalization=PROPRIO_ZSCORE,
        clip=5.0,
    )
    stats_id = make_stats_id(
        ROBOTWIN_DATASET_KEY,
        action_stats["mean"],
        action_stats["std"],
        proprio_stats,
        action_normalization=ACTION_ZSCORE,
        source_manifest_hash=audit["source_manifest_sha256"],
    )
    stats = {
        "action": action_stats,
        "proprio": proprio_stats,
        "absolute_action_mask": [True] * ACTION_DIM,
        "schema_version": 3,
        "stats_id": stats_id,
        "action_type": ACTION_TYPE,
        "action_representation": ACTION_REPRESENTATION,
        "action_transform": None,
        "delta_action_mask": None,
        "robot_action_dim_info": ROBOTWIN_ACTION_DIM_INFO,
        "source_manifest_sha256": audit["source_manifest_sha256"],
        "aggregation": {
            "scope": "global_exact_50_tasks_episode_ids_0000000_through_0000049",
            "tasks": len(ROBOTWIN_TASKS),
            "episodes": len(episodes),
            "transitions": int(len(action_values)),
        },
    }
    validate_robotwin_training_statistics(stats)
    return stats


def _load_matching_statistics(path: Path, audit: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        stats = payload[ROBOTWIN_DATASET_KEY]
        validate_robotwin_training_statistics(stats)
    except (EOFError, KeyError, OSError, pickle.UnpicklingError, ValueError, TypeError) as exc:
        raise ValueError(f'Invalid existing statistics at {path}; preserve and inspect this file.') from exc
    if stats.get("source_manifest_sha256") != audit["source_manifest_sha256"]:
        raise ValueError(f'Existing statistics at {path} belong to a different training split.')
    aggregation = stats.get("aggregation", {})
    if (
        int(aggregation.get("tasks", -1)) != len(ROBOTWIN_TASKS)
        or int(aggregation.get("episodes", -1)) != int(audit["selected_episode_count"])
    ):
        raise ValueError(f'Existing statistics at {path} have incorrect task/episode counts.')
    return dict(stats)


def _wait_for_matching_statistics(
    path: Path,
    audit: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("--stats-wait-timeout must be positive.")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        stats = _load_matching_statistics(path, audit)
        if stats is not None:
            return stats
        time.sleep(1.0)
    raise TimeoutError(
        f"Timed out after {timeout_seconds:g}s waiting for shard 0 to publish "
        f"matching global statistics at {path}."
    )


def _finalize_parallel_audit(
    output_dir: Path,
    audit: Mapping[str, Any],
    *,
    num_shards: int,
    stats_id: str,
    visual_encoder_type: str,
    visual_contract_sha256: str,
) -> bool:
    expected_episodes = int(audit["selected_episode_count"])
    for index in range(expected_episodes):
        metadata_path = output_dir / f"episode_{index:06d}" / "metadata.json"
        if not metadata_path.is_file():
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded_type = metadata.get(
            "visual_encoder_type", "dinov3" if "dinov3" in metadata else None,
        )
        recorded_contract = metadata.get(visual_encoder_type)
        if recorded_type != visual_encoder_type or not isinstance(recorded_contract, Mapping):
            raise ValueError(
                f"{metadata_path} does not match extraction visual backend "
                f"{visual_encoder_type!r}."
            )
        if json_sha256(recorded_contract) != visual_contract_sha256:
            raise ValueError(f"{metadata_path} has a different visual feature contract.")
    reports = []
    for shard_id in range(num_shards):
        path = _shard_audit_path(output_dir, shard_id, num_shards)
        if not path.is_file():
            return False
        report = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "shard_id": shard_id,
            "num_shards": num_shards,
            "stats_id": stats_id,
            "source_manifest_sha256": audit["source_manifest_sha256"],
        }
        mismatches = {
            key: (report.get(key), value)
            for key, value in expected.items()
            if report.get(key) != value
        }
        # Shard reports may omit visual provenance; the per-episode checks above
        # cover it, and any fields that are present remain strict.
        for key, value in (
            ("visual_encoder_type", visual_encoder_type),
            ("visual_contract_sha256", visual_contract_sha256),
        ):
            if report.get(key) not in (None, value):
                mismatches[key] = (report.get(key), value)
        if mismatches or not report.get("complete"):
            return False
        reports.append(report)
    assigned = sum(int(report["assigned"]) for report in reports)
    processed = sum(int(report["processed"]) for report in reports)
    resumed = sum(int(report["resumed"]) for report in reports)
    complete = (
        assigned == expected_episodes
        and processed + resumed == assigned
    )
    final_audit = dict(audit)
    final_audit["extraction"] = {
        "processed": processed,
        "resumed": resumed,
        "complete": complete,
        "stats_id": stats_id,
        "num_shards": num_shards,
        "visual_encoder_type": visual_encoder_type,
        "visual_contract_sha256": visual_contract_sha256,
        "shard_reports": [
            _shard_audit_path(output_dir, shard_id, num_shards).name
            for shard_id in range(num_shards)
        ],
    }
    _json_dump(output_dir / "extraction_audit.json", final_audit)
    return complete


def process_episode(
    index: int,
    episode: RoboTwinEpisode,
    *,
    output_dir: Path,
    load_xspark_v1,
    decode_image_bit,
    pack_robot_state,
    visual_encoder_type: str,
    dino_encoder,
    vae,
    visual_contract: Mapping[str, Any],
    future_rgb_horizon: int,
    per_view_resolution: tuple[int, int],
    text_encoder,
    tokenizer,
    device,
    text_dtype,
    batch_size: int,
    stats: Mapping[str, Any],
    text_contract: Mapping[str, Any],
    verify_existing_text: bool = False,
) -> bool:
    episode_dir = output_dir / f"episode_{index:06d}"
    metadata_path = episode_dir / "metadata.json"
    if metadata_path.is_file():
        validate_multiview_episode(
            episode_dir,
            ROBOTWIN_DATASET_KEY,
            ["cam_head"],
            ["cam_left_wrist", "cam_right_wrist"],
            proprio_dim=PROPRIO_DIM,
            require_vae=visual_encoder_type == "wan_vae",
            require_dino=visual_encoder_type == "dinov3",
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "task_name": episode.task,
            "source_episode_id": episode.episode_id,
            "stats_id": stats["stats_id"],
            "image_decoder": "XPolicyLab.utils.process_data.decode_image_bit",
            "image_color_order": "RGB",
            "action_dim": ACTION_DIM,
            "proprio_dim": PROPRIO_DIM,
            "action_type": ACTION_TYPE,
            "action_representation": ACTION_REPRESENTATION,
            "action_transform": None,
            "action_normalization": ACTION_ZSCORE,
            "proprio_normalization": PROPRIO_ZSCORE,
            visual_encoder_type: dict(visual_contract),
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing {episode_dir} provenance mismatch: {mismatches}")
        recorded_visual_type = metadata.get(
            "visual_encoder_type", "dinov3" if "dinov3" in metadata else None,
        )
        if recorded_visual_type != visual_encoder_type:
            raise ValueError(
                f"Existing {episode_dir} visual type is {recorded_visual_type!r}, "
                f"expected {visual_encoder_type!r}."
            )
        actions = np.load(episode_dir / "actions.npy", mmap_mode="r")
        proprio = np.load(episode_dir / "proprio.npy", mmap_mode="r")
        total = int(metadata["num_raw_frames"])
        if actions.shape != (total, ACTION_DIM) or proprio.shape != (total, PROPRIO_DIM):
            raise ValueError(
                f"Existing {episode_dir} must contain actions/proprio with exact shapes "
                f"({total},14), got {actions.shape}/{proprio.shape}."
            )
        if not np.isfinite(actions).all() or not np.isfinite(proprio).all():
            raise ValueError(f"Existing {episode_dir} action/proprio contains non-finite values.")
        recorded_text = metadata.get('text_contract')
        if recorded_text is not None and recorded_text != text_contract:
            raise ValueError(f'Existing {episode_dir} text encoder/tokenizer contract differs.')
        if verify_existing_text:
            raw = load_xspark_v1(str(episode.path), decode_images=False)
            if _load_instruction(raw, episode) != metadata['language_instruction']:
                raise ValueError(f'Existing {episode_dir} instruction differs from source HDF5/JSON.')
            context, mask = encode_text(text_encoder, tokenizer, metadata['language_instruction'], device, text_dtype)
            saved = torch.load(episode_dir/'text_embed.pt', map_location='cpu', weights_only=True)
            if not torch.equal(saved['context'], context) or not torch.equal(saved['mask'], mask):
                raise ValueError(f'Existing {episode_dir} text embedding differs from current encoder.')
            metadata['text_contract'] = dict(text_contract)
            _json_dump(metadata_path, metadata)
        elif recorded_text is None:
            raise ValueError(f'Existing {episode_dir} lacks text provenance; use --verify-existing-text.')
        return False

    raw = load_xspark_v1(str(episode.path), decode_images=False)
    proprio, actions = _pack_episode(raw, pack_robot_state, episode.path)
    frames = {
        camera: _camera_rgb(raw, camera, decode_image_bit, episode.path)
        for camera in ROBOTWIN_CAMERA_KEYS
    }
    lengths = [len(actions), len(proprio), *(len(frames[key]) for key in ROBOTWIN_CAMERA_KEYS)]
    if len(set(lengths)) != 1:
        raise ValueError(f"{episode.path} has inconsistent temporal lengths {lengths}.")
    instruction = _load_instruction(raw, episode)
    total = len(actions)

    main_frames = [frames["cam_head"]]
    wrist_frames = [frames["cam_left_wrist"], frames["cam_right_wrist"]]
    context, mask = encode_text(
        text_encoder, tokenizer, instruction, device, text_dtype,
    )

    episode_dir.mkdir(parents=True, exist_ok=True)
    if visual_encoder_type == "dinov3":
        main = [
            encode_dinov3_frames(
                dino_rgb_uint8_to_bchw(image, source=f"{episode.path}:cam_head"),
                dino_encoder, device, batch_size,
            )
            for image in main_frames
        ]
        wrist = [
            encode_dinov3_frames(
                dino_rgb_uint8_to_bchw(image, source=f"{episode.path}:{camera}"),
                dino_encoder, device, batch_size,
            )
            for image, camera in zip(
                wrist_frames, ("cam_left_wrist", "cam_right_wrist"), strict=True,
            )
        ]
        torch.save(
            {
                "dino_main": stack_slot_dino_features(main, total),
                "dino_wrist": stack_slot_dino_features(wrist, total),
                "main_present": torch.ones((total, 1), dtype=torch.bool),
                "wrist_present": torch.ones((total, 2), dtype=torch.bool),
            },
            episode_dir / "dino_features.pt",
        )
    else:
        encoded_main, encoded_wrist = [], []
        for image, camera, target in (
            (main_frames[0], "cam_head", encoded_main),
            (wrist_frames[0], "cam_left_wrist", encoded_wrist),
            (wrist_frames[1], "cam_right_wrist", encoded_wrist),
        ):
            normalized = wan_vae_rgb_uint8_to_bchw(
                image, resolution=per_view_resolution, source=f"{episode.path}:{camera}",
            )
            target.append(encode_rolling_vae_latents(
                normalized, vae, device, text_dtype, future_rgb_horizon, batch_size,
            ))
        num_fwd = future_rgb_horizon // 4
        latent_h, latent_w = encoded_main[0][0].shape[-2:]
        z_t_main, z_fwd_main = stack_slot_latents(
            [pair[0] for pair in encoded_main], [pair[1] for pair in encoded_main],
            total, num_fwd, latent_h, latent_w,
        )
        z_t_wrist, z_fwd_wrist = stack_slot_latents(
            [pair[0] for pair in encoded_wrist], [pair[1] for pair in encoded_wrist],
            total, num_fwd, latent_h, latent_w,
        )
        torch.save(
            {
                "z_t_main": z_t_main,
                "z_fwd_main": z_fwd_main,
                "main_present": torch.ones((total, 1), dtype=torch.bool),
                "z_t_wrist": z_t_wrist,
                "z_fwd_wrist": z_fwd_wrist,
                "wrist_present": torch.ones((total, 2), dtype=torch.bool),
            },
            episode_dir / "vae_latent.pt",
        )
    np.save(episode_dir / "actions.npy", normalize_actions(actions, stats))
    np.save(episode_dir / "proprio.npy", proprio)
    torch.save({"context": context, "mask": mask}, episode_dir / "text_embed.pt")
    metadata = {
        "dataset_name": ROBOTWIN_DATASET_KEY,
        "task_name": episode.task,
        "source_episode_id": episode.episode_id,
        "source_hdf5": str(episode.path),
        "language_instruction": instruction,
        "num_raw_frames": total,
        "feature_format_version": "multiview_v1",
        "visual_encoder_type": visual_encoder_type,
        "num_main_slots": 1,
        "num_wrist_slots": 2,
        "main_camera_keys": ["cam_head"],
        "wrist_camera_keys": ["cam_left_wrist", "cam_right_wrist"],
        "camera_slot_assignment": {
            "main[0]": "cam_head",
            "wrist[0]": "cam_left_wrist",
            "wrist[1]": "cam_right_wrist",
        },
        "image_decoder": "XPolicyLab.utils.process_data.decode_image_bit",
        "image_color_order": "RGB",
        "extra_manual_resize": visual_encoder_type == "wan_vae",
        "action_dim": ACTION_DIM,
        "proprio_dim": PROPRIO_DIM,
        "action_type": ACTION_TYPE,
        "action_representation": ACTION_REPRESENTATION,
        "action_transform": None,
        "action_normalization": ACTION_ZSCORE,
        "proprio_normalization": PROPRIO_ZSCORE,
        "normalization_clip": 5.0,
        "normalization_epsilon": 1e-8,
        "stats_id": stats["stats_id"],
        visual_encoder_type: dict(visual_contract),
        "text_contract": dict(text_contract),
        "text_embed_shape": list(context.shape),
        "context_len": int(context.shape[0]),
        "schema_version": 3,
    }
    if visual_encoder_type == "dinov3":
        metadata["visual_window_contract"] = {
            "storage": "per_frame_v1",
            "future_horizon_source": "runtime_protocol",
            "endpoint_padding": "repeat_last_frame",
        }
    else:
        metadata.update({
            "future_rgb_horizon": future_rgb_horizon,
            "num_fwd_latents": future_rgb_horizon // 4,
            "vae_temporal_encoding": VAE_TEMPORAL_ENCODING,
            "per_view_resolution": list(per_view_resolution),
            "z_t_shape": list(z_t_main.shape),
            "z_fwd_shape": list(z_fwd_main.shape),
        })
    _json_dump(metadata_path, metadata)
    return True


def main() -> None:
    args = parse_args()
    if args.batch_size is None:
        args.batch_size = 64 if args.visual_encoder == "dinov3" else 8
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not args.device.startswith('cuda') or args.text_dtype != 'bf16':
        raise ValueError('Canonical RoboTwin extraction requires CUDA and BF16 text encoding.')
    if args.future_rgb_horizon <= 0 or args.future_rgb_horizon % 4:
        raise ValueError("--future-rgb-horizon must be positive and divisible by four.")
    if len(args.per_view_resolution) != 2 or any(x <= 0 for x in args.per_view_resolution):
        raise ValueError("--per-view-resolution must contain two positive integers.")
    shard_id, num_shards = _resolve_sharding(args)
    robotwin_root = Path(args.robotwin_root).expanduser().resolve()
    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root else robotwin_root / "data"
    )
    default_output = (
        "datasets/dino_features_robotwin2_demo_clean_50x50"
        if args.visual_encoder == "dinov3"
        else "datasets/vae_features_robotwin2_demo_clean_50x50"
    )
    output_dir = Path(args.output_dir or default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_robotwin_joint_contract(
        action_type=ACTION_TYPE,
        robot_action_dim_info=ROBOTWIN_ACTION_DIM_INFO,
        action_dim=ACTION_DIM,
        proprio_dim=PROPRIO_DIM,
        action_representation=ACTION_REPRESENTATION,
        action_transform=None,
        delta_action_mask=None,
    )
    try:
        episodes, audit = select_training_episodes(data_root)
    except RoboTwinSplitError as exc:
        if shard_id == 0:
            _json_dump(output_dir / "training_split_audit.json", exc.audit)
        raise
    if args.audit_only:
        if shard_id == 0:
            _json_dump(output_dir / "training_split_audit.json", audit)
        LOGGER.info(
            "Selected exactly %d audited episodes; shard=%d/%d (audit only).",
            len(episodes), shard_id, num_shards,
        )
        return
    if shard_id == 0:
        initial_audit = dict(audit)
        initial_audit["extraction"] = {
            "complete": False,
            "num_shards": num_shards,
            "status": "in_progress",
        }
        _json_dump(output_dir / "extraction_audit.json", initial_audit)
    LOGGER.info(
        "Selected exactly %d audited episodes; shard=%d/%d.",
        len(episodes), shard_id, num_shards,
    )

    load_xspark_v1, decode_image_bit, pack_robot_state = _install_xpolicylab_import(
        robotwin_root,
    )
    stats_path = output_dir / "dataset_statistics.pkl"
    if shard_id == 0:
        stats = _load_matching_statistics(stats_path, audit)
        if stats is None:
            LOGGER.info("Shard 0 is computing global statistics across all selected episodes.")
            stats = compute_statistics(episodes, load_xspark_v1, pack_robot_state, audit)
            _pickle_dump(stats_path, {ROBOTWIN_DATASET_KEY: stats})
        else:
            LOGGER.info("Shard 0 is reusing matching audited statistics at %s.", stats_path)
    else:
        LOGGER.info("Shard %d is waiting for shard 0 statistics at %s.", shard_id, stats_path)
        stats = _wait_for_matching_statistics(
            stats_path, audit, timeout_seconds=args.stats_wait_timeout,
        )

    device_name = args.device
    if device_name == "cuda" and "LOCAL_RANK" in os.environ:
        device_name = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    device = torch.device(device_name)
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Resolved device {device} but only {torch.cuda.device_count()} CUDA devices "
                "are visible. Check CUDA_VISIBLE_DEVICES/LOCAL_RANK."
            )
        torch.cuda.set_device(device)
    text_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.text_dtype]
    dino_encoder = vae = None
    if args.visual_encoder == "dinov3":
        dino_encoder = DINOv3FrameEncoder(
            architecture="vitb16",
            pretraining="lvd1689m",
            model_id=args.dinov3_model_id,
            revision=args.dinov3_revision,
            input_size=224,
            frozen=True,
        ).to(device=device, dtype=torch.float32).eval()
        visual_contract = dino_encoder.feature_metadata(storage_dtype="bfloat16")
    else:
        vae_path = Path(args.vae_path).expanduser().resolve()
        vae = load_vae(vae_path, device, text_dtype)
        visual_contract = wan_vae_feature_contract(
            vae_path, args.per_view_resolution, text_dtype, args.future_rgb_horizon,
        )
    text_encoder, tokenizer = load_text_encoder(
        args.text_encoder_path,
        args.tokenizer_id,
        args.context_len,
        device,
        text_dtype,
    )
    text_contract = text_feature_contract(args.text_encoder_path, args.tokenizer_id, tokenizer,
        args.context_len, text_dtype, output_dir/'.encoder_fingerprints')
    visual_contract_sha256 = json_sha256(visual_contract)

    start, end = _shard_bounds(len(episodes), shard_id, num_shards)
    indexed_episodes = list(enumerate(episodes))[start:end]
    LOGGER.info(
        "Shard %d/%d owns global episode indices [%d, %d) (%d episodes) on %s.",
        shard_id, num_shards, start, end, len(indexed_episodes), device,
    )
    processed = skipped = 0
    for index, episode in tqdm(
        indexed_episodes,
        desc=f"RoboTwin {args.visual_encoder} shard {shard_id}/{num_shards}",
    ):
        changed = process_episode(
            index,
            episode,
            output_dir=output_dir,
            load_xspark_v1=load_xspark_v1,
            decode_image_bit=decode_image_bit,
            pack_robot_state=pack_robot_state,
            visual_encoder_type=args.visual_encoder,
            dino_encoder=dino_encoder,
            vae=vae,
            visual_contract=visual_contract,
            future_rgb_horizon=args.future_rgb_horizon,
            per_view_resolution=tuple(args.per_view_resolution),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            device=device,
            text_dtype=text_dtype,
            batch_size=args.batch_size,
            stats=stats,
            text_contract=text_contract,
            verify_existing_text=args.verify_existing_text,
        )
        processed += int(changed)
        skipped += int(not changed)
    shard_report = {
        "shard_id": shard_id,
        "num_shards": num_shards,
        "global_index_start": start,
        "global_index_end": end,
        "assigned": len(indexed_episodes),
        "processed": processed,
        "resumed": skipped,
        "complete": processed + skipped == len(indexed_episodes),
        "stats_id": stats["stats_id"],
        "source_manifest_sha256": audit["source_manifest_sha256"],
        "visual_encoder_type": args.visual_encoder,
        "visual_contract_sha256": visual_contract_sha256,
        "device": str(device),
    }
    _json_dump(_shard_audit_path(output_dir, shard_id, num_shards), shard_report)
    finalized = _finalize_parallel_audit(
        output_dir,
        audit,
        num_shards=num_shards,
        stats_id=str(stats["stats_id"]),
        visual_encoder_type=args.visual_encoder,
        visual_contract_sha256=visual_contract_sha256,
    )
    LOGGER.info(
        "Shard complete: shard=%d/%d processed=%d resumed=%d global_audit_complete=%s",
        shard_id, num_shards, processed, skipped, finalized,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
