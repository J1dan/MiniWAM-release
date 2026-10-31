"""
dataset.py

Dataset for MiniWAM training on pre-extracted features:
  - Multiview DINOv3 patch tokens or Wan-VAE latents instead of raw images
  - Pre-computed UMT5 instruction embeddings instead of tokenized text
  - Action chunks with end-of-episode padding and masks

End-of-episode handling:
  - Actions: padded with neutral actions (the normalized zero action for
             relative dims, the last valid action for absolute dims such as
             the gripper). This padding rule is adapted from OpenVLA
             (prismatic/vla/datasets/rlds/traj_transforms.py, MIT).
  - Future frames: padded by repeating the last frame.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .protocol import MiniWAMRuntimeProtocol

from miniwam.proprio import normalize_proprio
from miniwam.action_normalization import normalize_actions
from miniwam.statistics import ACTION_ZSCORE, get_action_bounds, load_statistics_mapping
from miniwam.robotwin import ROBOTWIN_DATASET_KEY, validate_robotwin_training_statistics

logger = logging.getLogger(__name__)

VAE_LATENT_KEYS = {
    "z_t_main",
    "z_fwd_main",
    "main_present",
    "z_t_wrist",
    "z_fwd_wrist",
    "wrist_present",
}
DINO_FEATURE_KEYS = {"dino_main", "dino_wrist", "main_present", "wrist_present"}
BASE_SAMPLE_KEYS = {
    "actions",
    "action_is_pad",
    "text_embeds",
    "text_mask",
    *VAE_LATENT_KEYS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Neutral action computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_neutral_action(dataset_statistics: dict, action_dim: int) -> np.ndarray:
    """
    Compute the normalized neutral (do-nothing) action used to pad action
    chunks past the end of an episode.

    For bounds normalization:
        neutral = 2 * (0 - lower) / max(upper - lower, 1e-8) - 1

    This is the value that maps to 0 in the unnormalized action space.
    For absolute action dimensions (e.g. gripper), the neutral action is
    handled per-sample by repeating the last valid action.

    Returns:
        neutral_action: [action_dim] numpy array
    """
    if "action" not in dataset_statistics:
        raise KeyError("Dataset statistics are missing the required action mapping.")
    action_stats = dataset_statistics["action"]
    if not isinstance(action_stats, dict):
        raise TypeError("Dataset action statistics must be a mapping.")
    parameter = action_stats.get("mean")
    if parameter is None:
        parameter = action_stats.get("min", action_stats.get("q01"))
    if parameter is None or np.asarray(parameter).shape[0] < action_dim:
        raise ValueError(f"Action statistics must have at least {action_dim} dimensions.")
    if action_stats.get("normalization") != ACTION_ZSCORE:
        lower, upper, constant, _ = get_action_bounds(dataset_statistics)
        neutral = 2.0 * (0.0 - lower) / np.maximum(upper - lower, 1e-8) - 1.0
        neutral[constant] = 0.0
        return neutral[:action_dim]
    zeros = np.zeros((np.asarray(parameter).shape[0],), dtype=np.float32)
    return np.asarray(normalize_actions(zeros, dataset_statistics), dtype=np.float32)[:action_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Pre-extracted feature dataset (recommended for training)
# ─────────────────────────────────────────────────────────────────────────────

class MiniWAMPreextractedDataset(Dataset):
    """
    Loads pre-extracted visual features and text embeddings from disk.

    Uses the multiview **rolling per-frame** extraction format where role/slot
    features are pre-computed for every raw frame:
        features_root/
            dataset_statistics.pkl
            episode_000000/
                dino_features.pt    # DINOv3: {dino_main, main_present,
                                    #          dino_wrist, wrist_present}
                vae_latent.pt       # Wan-VAE: {z_t_main, z_fwd_main, main_present,
                                    #           z_t_wrist, z_fwd_wrist, wrist_present}
                text_embed.pt       # {context: [L, D], mask: [L]}
                actions.npy         # [T, action_dim] flat per-timestep actions
                proprio.npy         # [T, proprio_dim]
                metadata.json       # {num_raw_frames, ...}

    End-of-episode padding:
        - Actions beyond episode end are replaced with neutral actions
          (normalized zero for relative dims, the last valid action for
          absolute dims like the gripper).
        - action_is_pad mask marks which action positions are padding.
    """

    def __init__(
        self,
        features_root: str,
        runtime_protocol: MiniWAMRuntimeProtocol,
        action_dim: int = 7,
        proprio_dim: Optional[int] = None,
        visual_encoder_config: Optional[dict] = None,
    ):
        self.features_root = features_root
        self.runtime_protocol = runtime_protocol
        self.action_horizon = runtime_protocol.action_horizon
        self.visual_encoder_type = runtime_protocol.visual_encoder_type
        self.temporal_downsample_factor = runtime_protocol.temporal_downsample_factor
        self.future_rgb_horizon = runtime_protocol.future_rgb_horizon
        self.num_future_frames = runtime_protocol.future_feature_horizon
        self.visual_encoder_config = dict(visual_encoder_config or {})
        self.action_dim = action_dim
        self.proprio_dim = None if proprio_dim in (None, 0) else int(proprio_dim)

        # Support comma-separated feature roots for training on multiple datasets
        roots = [Path(r.strip()) for r in features_root.split(",") if r.strip()]

        self.episode_dirs = []
        self.episode_neutral_action = []
        self.episode_absolute_mask = []
        self.episode_slot_counts = []
        self.episode_dataset_names = []
        self.episode_proprio_stats = []
        self.statistics = {}  # dataset name -> the statistics used to normalize it
        self.stats_sources = {}  # dataset name -> resolved path of its statistics file
        self.num_main_slots = 0
        self.num_wrist_slots = 0

        for root in roots:
            stats_path = root / "dataset_statistics.pkl"
            if not stats_path.exists():
                raise FileNotFoundError(f"Missing required dataset statistics: {stats_path}")
            all_stats = load_statistics_mapping(stats_path)
            stats_by_dataset = self._prepare_dataset_statistics(all_stats, stats_path, action_dim)
            logger.info(
                "Loaded dataset statistics from %s for datasets: %s",
                stats_path, sorted(stats_by_dataset),
            )

            # Find episode directories in this root
            ep_dirs = sorted([
                d for d in root.iterdir()
                if d.is_dir() and (d / "metadata.json").exists()
            ])

            for ep_dir in ep_dirs:
                slot_counts, dataset_name = self._inspect_multiview_episode(
                    ep_dir, self.action_dim, self.proprio_dim, self.num_future_frames,
                    visual_encoder_type=self.visual_encoder_type,
                    future_rgb_horizon=self.future_rgb_horizon,
                    temporal_downsample_factor=self.temporal_downsample_factor,
                    visual_encoder_config=self.visual_encoder_config,
                )
                stats_entry = self._stats_for_episode_dataset(stats_by_dataset, dataset_name, stats_path)
                with (ep_dir / "metadata.json").open("r", encoding="utf-8") as handle:
                    episode_metadata = json.load(handle)
                episode_stats_id = episode_metadata.get("stats_id")
                if episode_stats_id is None:
                    raise KeyError(f"{ep_dir / 'metadata.json'} is missing stats_id.")
                if str(episode_stats_id) != stats_entry["stats_id"]:
                    raise ValueError(
                        f"Statistics mismatch in {ep_dir}: episode={episode_stats_id!r}, "
                        f"dataset_statistics={stats_entry['stats_id']!r}."
                    )
                previous_source = self.stats_sources.setdefault(dataset_name, str(stats_path.resolve()))
                if previous_source != str(stats_path.resolve()):
                    raise ValueError(f"Dataset {dataset_name!r} resolves to multiple statistics sources.")
                previous = self.statistics.setdefault(dataset_name, all_stats[dataset_name])
                if str(previous["stats_id"]) != stats_entry["stats_id"]:
                    raise ValueError(f"Conflicting statistics IDs for {dataset_name!r}.")
                self.episode_dirs.append(ep_dir)
                self.episode_neutral_action.append(stats_entry["neutral_action"])
                self.episode_absolute_mask.append(stats_entry["absolute_action_mask"])
                self.episode_slot_counts.append(slot_counts)
                self.episode_dataset_names.append(dataset_name)
                self.episode_proprio_stats.append(stats_entry.get("proprio"))
                self.num_main_slots = max(self.num_main_slots, slot_counts[0])
                self.num_wrist_slots = max(self.num_wrist_slots, slot_counts[1])

        if not self.episode_dirs:
            raise FileNotFoundError(
                f"No complete episodes found in {self.features_root}. "
                "Each episode must have a metadata.json file."
            )

        # ── Build frame index: Iterate over RAW frames ──
        # With rolling per-frame extraction, every raw frame has its own
        # z_t and z_fwd pre-computed. No temporal downsampling needed.
        self.frame_index = []
        self.episode_frame_offsets = []
        for ep_idx, ep_dir in enumerate(self.episode_dirs):
            with open(ep_dir / "metadata.json", "r") as f:
                meta = json.load(f)
            num_raw_frames = meta["num_raw_frames"]
            self.episode_frame_offsets.append(len(self.frame_index))

            for frame_idx in range(num_raw_frames):
                self.frame_index.append((ep_idx, frame_idx))

        logger.info(
            f"MiniWAMPreextractedDataset: {len(self.episode_dirs)} episodes, "
            f"{len(self.frame_index)} valid frames (raw, 1:1 mapping), "
            f"slot schema main={self.num_main_slots}, wrist={self.num_wrist_slots}"
        )

    def __len__(self) -> int:
        return len(self.frame_index)

    def _resolve_sample_index(self, idx: int) -> Tuple[int, int, int]:
        normalized = int(idx)
        if normalized < 0:
            normalized += len(self.frame_index)
        if normalized < 0 or normalized >= len(self.frame_index):
            raise IndexError(f"MiniWAM sample index out of range: {idx}")
        ep_idx, frame_idx = self.frame_index[normalized]
        return normalized, ep_idx, frame_idx

    def episode_metadata(self, episode_index: int) -> dict:
        """Load a fresh copy of the exact source episode metadata."""
        ep_idx = int(episode_index)
        if ep_idx < 0:
            ep_idx += len(self.episode_dirs)
        if ep_idx < 0 or ep_idx >= len(self.episode_dirs):
            raise IndexError(f"MiniWAM episode index out of range: {episode_index}")
        path = self.episode_dirs[ep_idx] / "metadata.json"
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise TypeError(f"{path} must contain a JSON object.")
        return metadata

    def load_visual_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load only current/future visual features and camera presence masks."""
        _, ep_idx, frame_idx = self._resolve_sample_index(idx)
        ep_dir = self.episode_dirs[ep_idx]

        if self.visual_encoder_type == "wan_vae":
            data = torch.load(
                str(ep_dir / "vae_latent.pt"),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            self._validate_vae_schema(data, ep_dir)
            z_t_main = data["z_t_main"][frame_idx]
            z_fwd_main = data["z_fwd_main"][frame_idx, :, :, :self.num_future_frames]
            main_present = data["main_present"][frame_idx].bool()
            z_t_wrist = data["z_t_wrist"][frame_idx]
            z_fwd_wrist = data["z_fwd_wrist"][frame_idx, :, :, :self.num_future_frames]
            wrist_present = data["wrist_present"][frame_idx].bool()
        else:
            data = torch.load(
                str(ep_dir / "dino_features.pt"),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            self._validate_dino_schema(data, ep_dir)
            total = data["dino_main"].shape[0]
            indices = torch.arange(
                frame_idx + 1, frame_idx + 1 + self.future_rgb_horizon
            ).clamp(max=total - 1)
            z_t_main = data["dino_main"][frame_idx]
            z_fwd_main = data["dino_main"].index_select(0, indices).permute(1, 0, 2, 3)
            z_t_wrist = data["dino_wrist"][frame_idx]
            z_fwd_wrist = data["dino_wrist"].index_select(0, indices).permute(1, 0, 2, 3)
            main_present = data["main_present"][frame_idx].bool()
            wrist_present = data["wrist_present"][frame_idx].bool()

        z_t_main, z_fwd_main, main_present = self._pad_role_slots(
            z_t_main, z_fwd_main, main_present, self.num_main_slots, "main", ep_dir,
        )
        z_t_wrist, z_fwd_wrist, wrist_present = self._pad_role_slots(
            z_t_wrist, z_fwd_wrist, wrist_present, self.num_wrist_slots, "wrist", ep_dir,
        )
        return {
            "z_t_main": z_t_main,
            "z_fwd_main": z_fwd_main,
            "main_present": main_present,
            "z_t_wrist": z_t_wrist,
            "z_fwd_wrist": z_fwd_wrist,
            "wrist_present": wrist_present,
        }

    def _pad_actions(
        self,
        actions_full: np.ndarray,
        act_start: int,
        neutral_action: np.ndarray,
        absolute_action_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract an action chunk of length `action_horizon` starting at `act_start`.
        Pad beyond-end positions with neutral actions:
          - Relative dims (e.g. delta-xyz): neutral = normalized zero
          - Absolute dims (e.g. gripper):   neutral = repeat last valid action

        Returns:
            actions: [action_horizon, action_dim]
            action_is_pad: [action_horizon] boolean mask (True = padding)
        """
        T_total = len(actions_full)
        num_valid = min(self.action_horizon, T_total - act_start)

        # Extract valid portion
        valid_actions = actions_full[act_start:act_start + num_valid]  # [num_valid, D]

        if num_valid >= self.action_horizon:
            # No padding needed
            return valid_actions.copy(), np.zeros(self.action_horizon, dtype=bool)

        # Build padded action array
        pad_count = self.action_horizon - num_valid
        actions = np.empty((self.action_horizon, self.action_dim), dtype=np.float32)
        actions[:num_valid] = valid_actions

        # Pad with neutral actions
        #   - Relative dims: use normalized-zero action
        #   - Absolute dims: repeat the last valid action
        last_valid = valid_actions[-1] if num_valid > 0 else neutral_action
        for t in range(num_valid, self.action_horizon):
            for d in range(self.action_dim):
                if absolute_action_mask[d]:
                    actions[t, d] = last_valid[d]
                else:
                    actions[t, d] = neutral_action[d]

        action_is_pad = np.concatenate([
            np.zeros(num_valid, dtype=bool),
            np.ones(pad_count, dtype=bool),
        ])

        return actions, action_is_pad

    def _prepare_dataset_statistics(
        self,
        all_stats: dict,
        stats_path: Path,
        action_dim: int,
    ) -> Dict[str, dict]:
        stats_by_dataset = {}
        for ds_name, ds_stats in all_stats.items():
            if ds_name == ROBOTWIN_DATASET_KEY:
                validate_robotwin_training_statistics(ds_stats)
            if not ds_stats.get("stats_id"):
                raise KeyError(f"{stats_path} dataset {ds_name!r} is missing stats_id.")
            neutral_action = compute_neutral_action(ds_stats, action_dim)
            if "absolute_action_mask" not in ds_stats:
                raise KeyError(f"{stats_path} dataset {ds_name!r} is missing absolute_action_mask.")
            absolute_action_mask = np.array(ds_stats["absolute_action_mask"], dtype=bool)
            if absolute_action_mask.shape[0] < action_dim:
                raise ValueError(
                    f"{stats_path} dataset {ds_name!r} absolute_action_mask must have at least "
                    f"{action_dim} dims, got {absolute_action_mask.shape}."
                )
            prepared = {
                "neutral_action": neutral_action,
                "absolute_action_mask": absolute_action_mask[:action_dim],
                "stats_id": str(ds_stats["stats_id"]),
            }
            if self.proprio_dim is not None:
                if int(ds_stats.get("schema_version", 0)) not in (2, 3):
                    raise ValueError(
                        f"{stats_path} dataset {ds_name!r} must use schema_version=2 or 3 for proprio."
                    )
                if "proprio" not in ds_stats:
                    raise KeyError(f"{stats_path} dataset {ds_name!r} is missing proprio statistics.")
                # Validation and normalization are centralized in normalize_proprio.
                probe = np.zeros((self.proprio_dim,), dtype=np.float32)
                normalize_proprio(probe, ds_stats["proprio"])
                prepared["proprio"] = ds_stats["proprio"]
                prepared["stats_id"] = str(ds_stats["stats_id"])
            stats_by_dataset[str(ds_name)] = prepared
        return stats_by_dataset

    @staticmethod
    def _stats_for_episode_dataset(
        stats_by_dataset: Dict[str, dict],
        dataset_name: str,
        stats_path: Path,
    ) -> dict:
        if dataset_name not in stats_by_dataset:
            raise KeyError(
                f"Episode dataset_name={dataset_name!r} is not present in {stats_path}. "
                f"Available datasets: {sorted(stats_by_dataset)}"
            )
        return stats_by_dataset[dataset_name]

    @staticmethod
    def _validate_vae_schema(vae_data: dict, ep_dir: Path) -> None:
        keys = set(vae_data)
        if keys != VAE_LATENT_KEYS:
            raise ValueError(
                f"Unsupported VAE latent schema in {ep_dir / 'vae_latent.pt'}: "
                f"expected keys={sorted(VAE_LATENT_KEYS)}, got keys={sorted(keys)}."
            )

    @staticmethod
    def _validate_dino_schema(dino_data: dict, ep_dir: Path) -> None:
        keys = set(dino_data)
        if keys != DINO_FEATURE_KEYS:
            raise ValueError(
                f"Unsupported DINO feature schema in {ep_dir / 'dino_features.pt'}: "
                f"expected keys={sorted(DINO_FEATURE_KEYS)}, got keys={sorted(keys)}."
            )
        for key in ("dino_main", "dino_wrist"):
            if dino_data[key].dtype != torch.bfloat16:
                raise ValueError(
                    f"{ep_dir / 'dino_features.pt'} field {key!r} must store the BF16 "
                    f"MiniWAM feature boundary, got {dino_data[key].dtype}."
                )

    @staticmethod
    def _validate_text_embed(ep_dir: Path) -> None:
        text_path = ep_dir / "text_embed.pt"
        if not text_path.exists():
            raise FileNotFoundError(f"Missing required text embedding file: {text_path}")
        text_data = torch.load(str(text_path), map_location="cpu", weights_only=True, mmap=True)
        keys = set(text_data)
        expected = {"context", "mask"}
        if keys != expected:
            raise ValueError(
                f"Unsupported text embedding schema in {text_path}: "
                f"expected keys={sorted(expected)}, got keys={sorted(keys)}."
            )
        context = text_data["context"]
        mask = text_data["mask"]
        if context.ndim != 2 or mask.ndim != 1 or context.shape[0] != mask.shape[0]:
            raise ValueError(
                f"{text_path} must contain context [L,D] and mask [L], got "
                f"context={tuple(context.shape)}, mask={tuple(mask.shape)}."
            )
        if mask.to(torch.bool).sum().item() <= 1 or context.abs().sum().item() == 0:
            raise ValueError(f"{text_path} appears to contain dummy text embeddings.")

    @staticmethod
    def _validate_actions(ep_dir: Path, num_raw_frames: int, action_dim: int) -> None:
        actions_path = ep_dir / "actions.npy"
        if not actions_path.exists():
            raise FileNotFoundError(f"Missing required action file: {actions_path}")
        actions = np.load(actions_path, mmap_mode="r")
        if actions.ndim != 2 or actions.shape[0] != num_raw_frames or actions.shape[1] < action_dim:
            raise ValueError(
                f"{actions_path} must be [num_raw_frames, action_dim>= {action_dim}], got {actions.shape} "
                f"for num_raw_frames={num_raw_frames}."
            )

    @classmethod
    def _inspect_multiview_episode(
        cls,
        ep_dir: Path,
        action_dim: int,
        proprio_dim: Optional[int],
        num_future_frames: int,
        visual_encoder_type: str = "wan_vae",
        future_rgb_horizon: Optional[int] = None,
        temporal_downsample_factor: int = 4,
        visual_encoder_config: Optional[dict] = None,
    ) -> Tuple[Tuple[int, int], str]:
        if visual_encoder_type == "dinov3":
            return cls._inspect_dino_episode(
                ep_dir, action_dim, proprio_dim,
                int(future_rgb_horizon or num_future_frames * temporal_downsample_factor),
                visual_encoder_config,
            )
        vae_path = ep_dir / "vae_latent.pt"
        if not vae_path.exists():
            raise FileNotFoundError(f"Missing required multiview latent file: {vae_path}")

        vae_data = torch.load(str(vae_path), map_location="cpu", weights_only=True, mmap=True)
        cls._validate_vae_schema(vae_data, ep_dir)
        z_t_main, z_fwd_main = vae_data["z_t_main"], vae_data["z_fwd_main"]
        z_t_wrist, z_fwd_wrist = vae_data["z_t_wrist"], vae_data["z_fwd_wrist"]
        main_present, wrist_present = vae_data["main_present"], vae_data["wrist_present"]
        if z_t_main.ndim != 6 or z_fwd_main.ndim != 6:
            raise ValueError(f"Main-view latents in {vae_path} must be [T,M,C,T_lat,H,W].")
        if z_t_wrist.ndim != 6 or z_fwd_wrist.ndim != 6:
            raise ValueError(f"Wrist-view latents in {vae_path} must be [T,W,C,T_lat,H,W].")
        if z_t_main.shape[0] != z_fwd_main.shape[0] or z_t_wrist.shape[0] != z_fwd_wrist.shape[0]:
            raise ValueError(f"Mismatched temporal lengths in {vae_path}.")
        if main_present.shape != z_t_main.shape[:2]:
            raise ValueError(
                f"main_present in {vae_path} must be {tuple(z_t_main.shape[:2])}, got {tuple(main_present.shape)}."
            )
        if wrist_present.shape != z_t_wrist.shape[:2]:
            raise ValueError(
                f"wrist_present in {vae_path} must be {tuple(z_t_wrist.shape[:2])}, got {tuple(wrist_present.shape)}."
            )

        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing required metadata file: {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        feature_format = meta.get("feature_format_version")
        if feature_format != "multiview_v1":
            raise ValueError(
                f"Unsupported feature_format_version={feature_format!r} in {meta_path}. "
                "Expected feature_format_version='multiview_v1'."
            )
        for key in (
            "dataset_name", "num_main_slots", "num_wrist_slots", "num_raw_frames",
            "per_view_resolution", "main_camera_keys", "wrist_camera_keys",
        ):
            if key not in meta:
                raise KeyError(f"{meta_path} is missing required metadata key {key!r}.")
        dataset_name = str(meta["dataset_name"])
        num_main = int(meta["num_main_slots"])
        num_wrist = int(meta["num_wrist_slots"])
        num_raw = int(meta["num_raw_frames"])
        num_future_raw = meta.get("future_rgb_horizon", meta.get("num_future_raw_frames"))
        if num_future_raw is None:
            if "num_fwd_latents" in meta:
                num_future_raw = int(meta["num_fwd_latents"]) * int(temporal_downsample_factor)
            else:
                raise KeyError(f"{meta_path} is missing required metadata key 'num_future_raw_frames' or 'num_fwd_latents'.")
        else:
            num_future_raw = int(num_future_raw)
        if num_future_raw // int(temporal_downsample_factor) < num_future_frames:
            raise ValueError(
                f"{meta_path} provides too few future frames for {num_future_frames} latent frames."
            )
        if len(meta["main_camera_keys"]) != num_main or len(meta["wrist_camera_keys"]) != num_wrist:
            raise ValueError(
                f"{meta_path} camera-key counts do not match its declared role-slot counts."
            )
        if num_main < 1:
            raise ValueError(f"{meta_path} reports num_main_slots={num_main}; at least one main slot is required.")
        if num_wrist < 0:
            raise ValueError(f"{meta_path} reports invalid num_wrist_slots={num_wrist}.")
        if num_raw != int(z_t_main.shape[0]):
            raise ValueError(
                f"{meta_path} num_raw_frames={num_raw} does not match {vae_path} T={z_t_main.shape[0]}."
            )
        if num_main != int(z_t_main.shape[1]) or num_wrist != int(z_t_wrist.shape[1]):
            raise ValueError(
                f"{meta_path} slot counts main={num_main}, wrist={num_wrist} do not match "
                f"{vae_path} shapes main={z_t_main.shape[1]}, wrist={z_t_wrist.shape[1]}."
            )
        cls._validate_text_embed(ep_dir)
        cls._validate_actions(ep_dir, num_raw, action_dim=action_dim)
        if proprio_dim is not None:
            if int(meta.get("schema_version", 0)) not in (2, 3):
                raise ValueError(
                    f"{meta_path} must use schema_version=2 or 3 for proprio training."
                )
            if int(meta.get("proprio_dim", -1)) != proprio_dim:
                raise ValueError(
                    f"{meta_path} proprio_dim={meta.get('proprio_dim')}, expected {proprio_dim}."
                )
            proprio_path = ep_dir / "proprio.npy"
            if not proprio_path.exists():
                raise FileNotFoundError(f"Missing required raw proprio file: {proprio_path}")
            proprio = np.load(proprio_path, mmap_mode="r")
            if proprio.shape != (num_raw, proprio_dim):
                raise ValueError(
                    f"{proprio_path} must have shape ({num_raw},{proprio_dim}), got {proprio.shape}."
                )
            if not np.isfinite(proprio).all():
                raise ValueError(f"{proprio_path} contains non-finite values.")
        return (num_main, num_wrist), dataset_name

    @classmethod
    def _inspect_dino_episode(
        cls, ep_dir: Path, action_dim: int, proprio_dim: Optional[int],
        future_rgb_horizon: int,
        visual_encoder_config: Optional[dict],
    ) -> Tuple[Tuple[int, int], str]:
        path = ep_dir / "dino_features.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing required framewise DINO cache: {path}")
        data = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        cls._validate_dino_schema(data, ep_dir)
        main, wrist = data["dino_main"], data["dino_wrist"]
        if main.ndim != 4 or wrist.ndim != 4:
            raise ValueError("DINO caches must be [T,V,N,D] for each camera role.")
        if data["main_present"].shape != main.shape[:2] or data["wrist_present"].shape != wrist.shape[:2]:
            raise ValueError("DINO presence masks must match [T,V].")
        meta_path = ep_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("dataset_name", "num_main_slots", "num_wrist_slots", "num_raw_frames"):
            if key not in meta:
                raise KeyError(f"{meta_path} is missing required metadata key {key!r}.")
        total = int(meta["num_raw_frames"])
        if main.shape[0] != total or wrist.shape[0] != total:
            raise ValueError(f"DINO cache frame count does not match {meta_path}.")
        num_main, num_wrist = int(meta["num_main_slots"]), int(meta["num_wrist_slots"])
        if main.shape[1] != num_main or wrist.shape[1] != num_wrist:
            raise ValueError("DINO cache slot counts do not match metadata.")
        dino_meta = meta.get("dinov3", {})
        if dino_meta and dino_meta.get("cache_layout") != "per_frame_v1":
            raise ValueError(f"Unsupported DINO cache layout in {meta_path}: {dino_meta!r}.")
        window_contract = meta.get("visual_window_contract")
        if window_contract is not None:
            expected_window_contract = {
                "storage": "per_frame_v1",
                "future_horizon_source": "runtime_protocol",
                "endpoint_padding": "repeat_last_frame",
            }
            if window_contract != expected_window_contract:
                raise ValueError(
                    f"Unsupported visual_window_contract in {meta_path}: "
                    f"{window_contract!r}; expected {expected_window_contract!r}."
                )
        if visual_encoder_config:
            feature = visual_encoder_config.get("feature", {}) or {}
            layers = feature.get("layers", {}) or {}
            expected = {
                "architecture": visual_encoder_config.get("architecture"),
                "pretraining": visual_encoder_config.get("pretraining"),
                "model_id": visual_encoder_config.get("model_id"),
                "revision": visual_encoder_config.get("revision"),
                "resolved_revision": visual_encoder_config.get(
                    "resolved_revision", visual_encoder_config.get("revision")
                ),
                "input_size": visual_encoder_config.get("input_size"),
                "feature_type": feature.get("type"),
                "feature_normalization": feature.get("normalization"),
                "l2_normalize": feature.get("l2_normalize"),
                "include_cls": feature.get("include_cls"),
                "include_registers": feature.get("include_registers"),
                "feature_layers": layers.get("mode"),
                "compute_dtype": visual_encoder_config.get("compute_dtype"),
                "boundary_dtype": feature.get("output_dtype"),
                "storage_dtype": (visual_encoder_config.get("cache", {}) or {}).get(
                    "storage_dtype"
                ),
            }
            mismatches = {
                key: (dino_meta.get(key), value)
                for key, value in expected.items()
                if value is not None and dino_meta.get(key) != value
            }
            if mismatches:
                raise ValueError(f"DINO cache metadata mismatch in {meta_path}: {mismatches}.")
        cls._validate_text_embed(ep_dir)
        cls._validate_actions(ep_dir, total, action_dim)
        if proprio_dim is not None:
            proprio = np.load(ep_dir / "proprio.npy", mmap_mode="r")
            if proprio.shape != (total, proprio_dim):
                raise ValueError(f"Invalid proprio shape {proprio.shape} in {ep_dir}.")
        return (num_main, num_wrist), str(meta["dataset_name"])

    @staticmethod
    def _pad_role_slots(
        z_t_slots: torch.Tensor,
        z_fwd_slots: torch.Tensor,
        present: torch.Tensor,
        target_slots: int,
        role: str,
        ep_dir: Path,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_t_slots.shape[0] > target_slots:
            raise ValueError(
                f"{role} slots in {ep_dir} exceed dataset schema: "
                f"{z_t_slots.shape[0]} > {target_slots}."
            )
        if present.shape != (z_t_slots.shape[0],):
            raise ValueError(
                f"{role}_present for one frame in {ep_dir} must be [{z_t_slots.shape[0]}], got {tuple(present.shape)}."
            )
        pad_slots = target_slots - z_t_slots.shape[0]
        if pad_slots == 0:
            return z_t_slots, z_fwd_slots, present.bool()
        z_t_pad = torch.zeros(
            (pad_slots, *z_t_slots.shape[1:]),
            dtype=z_t_slots.dtype,
            device=z_t_slots.device,
        )
        z_fwd_pad = torch.zeros(
            (pad_slots, *z_fwd_slots.shape[1:]),
            dtype=z_fwd_slots.dtype,
            device=z_fwd_slots.device,
        )
        present_pad = torch.zeros((pad_slots,), dtype=torch.bool, device=present.device)
        return (
            torch.cat([z_t_slots, z_t_pad], dim=0),
            torch.cat([z_fwd_slots, z_fwd_pad], dim=0),
            torch.cat([present.bool(), present_pad], dim=0),
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        _, ep_idx, frame_idx = self._resolve_sample_index(idx)
        ep_dir = self.episode_dirs[ep_idx]
        neutral_action = self.episode_neutral_action[ep_idx]
        absolute_action_mask = self.episode_absolute_mask[ep_idx]
        proprio_stats = self.episode_proprio_stats[ep_idx]
        visual_sample = self.load_visual_sample(idx)

        # Text embeddings using memory mapping
        text_data = torch.load(str(ep_dir / "text_embed.pt"), map_location="cpu", weights_only=True, mmap=True)
        text_embeds = text_data["context"]  # [L, D]
        text_mask = text_data["mask"]       # [L]

        # Actions, with neutral padding past the episode end
        # With rolling extraction, frame_idx IS the raw action index (1:1 mapping)
        actions_full = np.load(ep_dir / "actions.npy")  # [T_raw, action_dim]
        actions, action_is_pad = self._pad_actions(
            actions_full, frame_idx, neutral_action, absolute_action_mask
        )

        sample = {
            "actions": torch.tensor(actions, dtype=torch.float32), # [Ta, action_dim]
            "action_is_pad": torch.tensor(action_is_pad),          # [Ta]
            "text_embeds": text_embeds,                            # [L, D]
            "text_mask": text_mask,                                # [L]
        }
        sample.update(visual_sample)
        if self.proprio_dim is not None:
            proprio_raw = np.load(ep_dir / "proprio.npy", mmap_mode="r")[frame_idx]
            sample["proprio"] = torch.from_numpy(
                normalize_proprio(proprio_raw, proprio_stats).copy()
            )
        return sample


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────

class MiniWAMCollator:
    """Collates multiview MiniWAM samples into batched tensors."""

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        has_proprio = "proprio" in batch[0]
        expected_keys = set(batch[0])
        if not BASE_SAMPLE_KEYS.issubset(expected_keys):
            raise ValueError(
                f"MiniWAM sample is missing required keys: {sorted(BASE_SAMPLE_KEYS - expected_keys)}."
            )
        for i, sample in enumerate(batch):
            keys = set(sample)
            if keys != expected_keys:
                raise ValueError(
                    f"Unsupported MiniWAM sample schema at batch index {i}: "
                    f"expected keys={sorted(expected_keys)}, got keys={sorted(keys)}."
                )

        z_t_main = torch.stack([b["z_t_main"] for b in batch])
        z_fwd_main = torch.stack([b["z_fwd_main"] for b in batch])
        main_present = torch.stack([b["main_present"] for b in batch])
        z_t_wrist = torch.stack([b["z_t_wrist"] for b in batch])
        z_fwd_wrist = torch.stack([b["z_fwd_wrist"] for b in batch])
        wrist_present = torch.stack([b["wrist_present"] for b in batch])
        actions = torch.stack([b["actions"] for b in batch])       # [B, Ta, D]
        action_is_pad = torch.stack([b["action_is_pad"] for b in batch])  # [B, Ta]

        # Pad text to max length in batch
        text_embeds_list = [b["text_embeds"] for b in batch]
        text_mask_list = [b["text_mask"] for b in batch]
        max_text_len = max(t.shape[0] for t in text_embeds_list)
        text_dim = text_embeds_list[0].shape[1]

        padded_embeds = torch.zeros(len(batch), max_text_len, text_dim)
        padded_masks = torch.zeros(len(batch), max_text_len, dtype=torch.bool)

        for i, (emb, mask) in enumerate(zip(text_embeds_list, text_mask_list)):
            L = emb.shape[0]
            padded_embeds[i, :L] = emb
            padded_masks[i, :L] = mask

        out = {
            "actions": actions,
            "action_is_pad": action_is_pad,
            "text_embeds": padded_embeds,
            "text_mask": padded_masks,
        }
        out.update({
            "z_t_main": z_t_main,
            "z_fwd_main": z_fwd_main,
            "main_present": main_present,
            "z_t_wrist": z_t_wrist,
            "z_fwd_wrist": z_fwd_wrist,
            "wrist_present": wrist_present,
        })
        if has_proprio:
            out["proprio"] = torch.stack([b["proprio"] for b in batch])
        return out
