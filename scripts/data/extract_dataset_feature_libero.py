#!/usr/bin/env python
"""
extract_dataset_feature_libero.py

Pre-extract DINOv3 or Wan-VAE visual features and UMT5 text embeddings from LeRobot formatted datasets
(like LIBERO-fastwam). scripts/data/extract_dataset_feature_libero.sh runs this for all four
LIBERO suites with shared statistics.
The future RGB horizon defaults to 16 and is resolved by the shared protocol.

Requirements:
    - the miniwam environment ([data] extras: pandas and PyAV for parquet and video files)
    - --visual_encoder dinov3: DINOv3 ViT-B/16 in the Hugging Face cache
    - --visual_encoder wan_vae (the default here): pretrained/vision/Wan2.1_VAE.pth
    - UMT5-XXL text encoder: pretrained/language/models_t5_umt5-xxl-enc-bf16.pth
    - UMT5-XXL tokenizer: google/umt5-xxl

Usage:
    python scripts/data/extract_dataset_feature_libero.py \
        --data_root datasets/LIBERO-fastwam \
        --data_mix libero_10_no_noops_lerobot \
        --visual_encoder dinov3 \
        --output_dir datasets/dino_features_libero_aggregate_minmax/libero_10_no_noops_lerobot \
        --future_rgb_horizon 16
"""

import argparse
import json
import logging
import sys
import pickle
from pathlib import Path

# Prefer this checkout over any separately installed editable MiniWAM package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import av
import math
from tqdm import tqdm

from extraction_utils import (
    VAE_TEMPORAL_ENCODING,
    encode_rolling_vae_latents,
    encode_dinov3_frames,
    encode_text,
    load_text_encoder,
    load_vae,
    make_text_provenance,
    resize_and_normalize_frames,
    stack_slot_latents,
    stack_slot_dino_features,
    validate_multiview_episode,
)
from miniwam.models.dinov3 import DINOv3FrameEncoder
from miniwam.instruction_contract import (
    require_exact_instruction_match,
    require_instruction_string,
)
from miniwam.protocol import resolve_miniwam_protocol
from miniwam.proprio import compute_proprio_statistics, make_stats_id
from miniwam.statistics import (
    ACTION_MINMAX,
    ACTION_Q01_Q99,
    compute_action_statistics,
    get_action_bounds,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LIBERO_DATASETS = (
    "libero_10_no_noops_lerobot",
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
)


def read_raw_proprio(df: pd.DataFrame) -> np.ndarray:
    if "observation.state" in df.columns:
        proprio = np.stack(df["observation.state"].tolist()).astype(np.float32)
    else:
        state_cols = sorted(
            [column for column in df.columns if column.startswith("observation.state_")],
            key=lambda name: int(name.rsplit("_", 1)[-1]),
        )
        if not state_cols:
            raise KeyError("LeRobot parquet has no observation.state column.")
        proprio = df[state_cols].to_numpy(dtype=np.float32)
    if proprio.ndim == 3 and proprio.shape[1] == 1:
        proprio = proprio[:, 0]
    if proprio.shape[1:] != (8,):
        raise ValueError(f"LIBERO proprio must be [T,8], got {proprio.shape}.")
    return proprio


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-extract DINOv3 or Wan-VAE features and text embeddings from a LeRobot dataset")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to LeRobot dataset root (e.g. datasets/LIBERO-fastwam)")
    parser.add_argument("--data_mix", type=str, required=True,
                        help="Dataset mixture names, comma separated (e.g. libero_10_no_noops_lerobot)")
    parser.add_argument("--stats_data_mix", type=str, default=",".join(LIBERO_DATASETS),
                        help="Datasets used to compute shared action and proprio statistics.")
    parser.add_argument(
        "--action_normalization", choices=["minmax", "q01_q99"], default="minmax",
        help="Action bounds policy. New LIBERO extraction defaults to aggregate min/max.",
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for extracted features")
    parser.add_argument("--future_rgb_horizon", type=int, default=None,
                        help="Canonical future RGB horizon. Defaults to 16.")
    parser.add_argument("--num_future_raw_frames", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--visual_encoder", choices=["wan_vae", "dinov3"], default="wan_vae")
    parser.add_argument("--dinov3_model_id", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--dinov3_revision", default="5931719e67bbdb9737e363e781fb0c67687896bc")

    # Model paths
    parser.add_argument("--vae_path", type=str, default="pretrained/vision/Wan2.1_VAE.pth",
                        help="Path to Wan 2.1 VAE checkpoint (.pth)")
    parser.add_argument("--text_encoder_path", type=str,
                        default="pretrained/language/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to UMT5-XXL text encoder checkpoint (.pth or .safetensors)")
    parser.add_argument(
        "--text_features_source_root", type=str, default=None,
        help=(
            "Optional existing LIBERO feature root whose per-episode BF16 text_embed.pt "
            "and text provenance are reused. This avoids loading UMT5 during DINO extraction."
        ),
    )
    parser.add_argument("--tokenizer_id", type=str, default="google/umt5-xxl",
                        help="HuggingFace tokenizer ID for UMT5-XXL")
    parser.add_argument("--context_len", type=int, default=128,
                        help="Maximum context length for text tokenization")

    # Data options
    parser.add_argument("--max_episodes", type=int, default=-1,
                        help="Maximum episodes to extract (-1 for all); statistics always use the full stats_data_mix")
    parser.add_argument("--per_view_resolution", type=int, nargs=2, default=[224, 224],
                        help="Per-camera encoder input resolution for multiview features (H, W)")
    parser.add_argument("--num_main_slots", type=int, default=1,
                        help="Maximum number of configured main camera keys to encode")
    parser.add_argument("--num_wrist_slots", type=int, default=1,
                        help="Maximum number of configured wrist camera keys to encode")
    parser.add_argument("--main_camera_keys", type=str, default="image",
                        help="Comma-separated LeRobot image keys for main slots, without 'observation.images.' prefix")
    parser.add_argument("--wrist_camera_keys", type=str, default="wrist_image",
                        help="Comma-separated LeRobot image keys for wrist slots, without 'observation.images.' prefix")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for visual encoding")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Data type for inference")
    parser.add_argument("--action_dim", type=int, default=7,
                        help="Dimension of action space")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard ID (0 to num_shards-1) for parallel extraction")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards for parallel extraction")
    return parser.parse_args()





# ── Episode Processing ───────────────────────────────────────────────────────
def process_lerobot_episode(
    data_root, dataset_name, episode_info, info_json, vae, text_encoder, tokenizer,
    output_dir, device, dtype, action_dim, context_len, num_future_raw_frames,
    action_lower, action_upper, action_normalization, stats_id, per_view_resolution, num_main_slots,
    num_wrist_slots, main_camera_keys, wrist_camera_keys, batch_size,
    text_provenance, dino_encoder=None, text_features_source_root=None,
):
    episode_idx = episode_info["episode_index"]
    lang = require_instruction_string(
        episode_info["tasks"][0], f"episode {episode_idx}",
    )
    expected_length = episode_info["length"]

    ep_dir = output_dir / f"episode_{episode_idx:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    if (ep_dir / "metadata.json").exists():
        main_keys = [key.strip() for key in main_camera_keys.split(",") if key.strip()][:num_main_slots]
        wrist_keys = [key.strip() for key in wrist_camera_keys.split(",") if key.strip()][:num_wrist_slots]
        validate_multiview_episode(
            ep_dir,
            dataset_name,
            main_keys,
            wrist_keys,
            proprio_dim=8,
            require_vae=vae is not None,
            require_dino=dino_encoder is not None,
        )
        with (ep_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
        if existing_metadata.get("stats_id") != stats_id:
            raise ValueError(
                f"{ep_dir} was extracted with stats_id={existing_metadata.get('stats_id')!r}, "
                f"but this run uses {stats_id!r}. Use a new output directory or re-extract "
                "the episode so actions and statistics cannot be mixed."
            )
        return None  # Already processed

    chunks_size = info_json.get("chunks_size", 1000)
    episode_chunk = episode_idx // chunks_size

    data_path_template = info_json["data_path"]
    video_path_template = info_json["video_path"]

    data_file = data_root / dataset_name / data_path_template.format(episode_chunk=episode_chunk, episode_index=episode_idx)

    def read_video(vfile):
        container = av.open(str(vfile))
        frames_list = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')
            img_t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(*img.shape)
            frames_list.append(img_t)
        container.close()
        return torch.stack(frames_list)[:expected_length].permute(0, 3, 1, 2).float() / 255.0

    # 1. Read Video
    def read_configured_camera(camera_key):
        video_key = f"observation.images.{camera_key}"
        video_file = data_root / dataset_name / video_path_template.format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_idx,
        )
        if not video_file.exists():
            raise FileNotFoundError(f"Missing configured camera video {video_file} for episode {episode_idx}")
        return resize_and_normalize_frames(read_video(video_file), tuple(per_view_resolution))

    main_keys = [k.strip() for k in main_camera_keys.split(",") if k.strip()]
    wrist_keys = [k.strip() for k in wrist_camera_keys.split(",") if k.strip()]
    if len(main_keys) < min(num_main_slots, 1):
        raise ValueError("At least one main camera key is required for LeRobot extraction.")
    main_frames = [read_configured_camera(k) for k in main_keys[:num_main_slots]]
    wrist_frames = [read_configured_camera(k) for k in wrist_keys[:num_wrist_slots]]

    present_frames = main_frames + wrist_frames
    if not present_frames:
        raise FileNotFoundError(f"No configured camera videos found for episode {episode_idx}")
    frames_norm = present_frames[0]
    T_raw = frames_norm.shape[0]
    H, W = per_view_resolution

    num_fwd_latents = num_future_raw_frames // 4

    if num_main_slots < 1:
        raise ValueError("LeRobot extraction requires at least one main-view slot.")
    if not main_frames:
        raise ValueError(f"Episode {episode_idx} has no configured main-view camera.")

    main_present_slots = [torch.ones(T_raw, dtype=torch.bool) for _ in main_frames]
    wrist_present_slots = [torch.ones(T_raw, dtype=torch.bool) for _ in wrist_frames]
    main_present = torch.stack(main_present_slots, dim=1)
    wrist_present = (
        torch.stack(wrist_present_slots, dim=1)
        if wrist_present_slots else torch.zeros((T_raw, 0), dtype=torch.bool)
    )

    z_t_main_slots, z_fwd_main_slots = [], []
    latent_h = latent_w = 28
    z_t_wrist_slots, z_fwd_wrist_slots = [], []
    if vae is not None:
        for frames_slot in main_frames:
            z_t_slot, z_fwd_slot = encode_rolling_vae_latents(
                frames_slot, vae, device, dtype, num_future_raw_frames, batch_size,
            )
            latent_h, latent_w = z_t_slot.shape[-2:]
            z_t_main_slots.append(z_t_slot)
            z_fwd_main_slots.append(z_fwd_slot)
        for frames_slot in wrist_frames:
            z_t_slot, z_fwd_slot = encode_rolling_vae_latents(
                frames_slot, vae, device, dtype, num_future_raw_frames, batch_size,
            )
            z_t_wrist_slots.append(z_t_slot)
            z_fwd_wrist_slots.append(z_fwd_slot)

        z_t_main_stacked, z_fwd_main_stacked = stack_slot_latents(
            z_t_main_slots, z_fwd_main_slots, T_raw, num_fwd_latents, latent_h, latent_w,
        )
        z_t_wrist_stacked, z_fwd_wrist_stacked = stack_slot_latents(
            z_t_wrist_slots, z_fwd_wrist_slots, T_raw, num_fwd_latents, latent_h, latent_w,
        )
        torch.save({
            "z_t_main": z_t_main_stacked,
            "z_fwd_main": z_fwd_main_stacked,
            "main_present": main_present,
            "z_t_wrist": z_t_wrist_stacked,
            "z_fwd_wrist": z_fwd_wrist_stacked,
            "wrist_present": wrist_present,
        }, ep_dir / "vae_latent.pt")
    else:
        z_t_main_stacked = z_fwd_main_stacked = None
        z_t_wrist_stacked = z_fwd_wrist_stacked = None

    if dino_encoder is not None:
        # Encode all camera streams together so large batches can saturate the
        # GPU even when an individual episode/view contains relatively few frames.
        camera_frames = main_frames + wrist_frames
        combined_features = encode_dinov3_frames(
            torch.cat(camera_frames, dim=0), dino_encoder, device, batch_size,
        )
        per_camera_features = list(combined_features.split(T_raw, dim=0))
        dino_main = per_camera_features[:len(main_frames)]
        dino_wrist = per_camera_features[len(main_frames):]
        torch.save({
            "dino_main": stack_slot_dino_features(dino_main, T_raw),
            "dino_wrist": stack_slot_dino_features(dino_wrist, T_raw),
            "main_present": main_present,
            "wrist_present": wrist_present,
        }, ep_dir / "dino_features.pt")

    # Read Actions
    df = pd.read_parquet(str(data_file))
    proprio = read_raw_proprio(df)[:T_raw]
    if "action" in df.columns:
        actions = np.stack(df["action"].tolist()).astype(np.float32)
    else:
        action_cols = sorted([c for c in df.columns if c.startswith("action_")], key=lambda x: int(x.split("_")[-1]))
        actions = df[action_cols].values.astype(np.float32)

    # Ensure correct shape if chunked
    if actions.ndim == 3 and actions.shape[1] == 1:
        actions = actions[:, 0]
    if actions.shape != (T_raw, action_dim) or proprio.shape != (T_raw, 8):
        raise ValueError(
            f"Episode {episode_idx} temporal/schema mismatch: images={T_raw}, "
            f"actions={actions.shape}, proprio={proprio.shape}."
        )

    actions = 2.0 * (actions - action_lower) / (
        action_upper - action_lower + 1e-8
    ) - 1.0
    actions[:, (action_upper - action_lower) <= 1e-6] = 0.0
    actions = np.clip(actions, -1.0, 1.0)

    np.save(ep_dir / "actions.npy", actions)
    np.save(ep_dir / "proprio.npy", proprio)

    # Reuse the canonical per-episode text feature when available. DINO-only
    # extraction should not keep an 11 GB UMT5 encoder resident on every GPU.
    if text_features_source_root is not None:
        source_ep = (
            Path(text_features_source_root)
            / dataset_name
            / f"episode_{episode_idx:06d}"
        )
        source_text_path = source_ep / "text_embed.pt"
        source_metadata_path = source_ep / "metadata.json"
        if not source_text_path.is_file() or not source_metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing reusable text feature or metadata under {source_ep}."
            )
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata.get("dataset_name") != dataset_name:
            raise ValueError(f"Reusable text dataset mismatch in {source_metadata_path}.")
        require_exact_instruction_match(
            source_metadata.get("language_instruction"),
            lang,
            source=str(source_metadata_path),
            current=f"episode {episode_idx}",
        )
        text_payload = torch.load(
            str(source_text_path), map_location="cpu", weights_only=True, mmap=True,
        )
        if set(text_payload) != {"context", "mask"}:
            raise KeyError(f"Invalid reusable text payload in {source_text_path}.")
        context = text_payload["context"].clone().contiguous()
        mask = text_payload["mask"].clone().contiguous()
        if context.shape != (context_len, 4096) or context.dtype != torch.bfloat16:
            raise ValueError(
                f"Reusable text context in {source_text_path} must be "
                f"[{context_len},4096] BF16, got {tuple(context.shape)} {context.dtype}."
            )
        if mask.shape != (context_len,) or mask.dtype != torch.bool:
            raise ValueError(f"Invalid reusable text mask in {source_text_path}.")
        text_provenance = source_metadata.get("text_provenance")
        if not isinstance(text_provenance, dict):
            raise ValueError(f"Missing reusable text provenance in {source_metadata_path}.")
    else:
        context, mask = encode_text(text_encoder, tokenizer, lang, device, dtype)
    torch.save({"context": context, "mask": mask}, ep_dir / "text_embed.pt")

    z_t_shape = list(z_t_main_stacked.shape) if z_t_main_stacked is not None else None
    z_fwd_shape = list(z_fwd_main_stacked.shape) if z_fwd_main_stacked is not None else None
    latent_label = "main_slot_latents"

    if episode_idx % 100 == 0:
        logger.info(
            f"\n=== Episode {episode_idx} Check ===\n"
            f"  Instruction: '{lang}'\n"
            f"  Raw frames:     [{T_raw}, 3, {H}, {W}]\n"
            f"  feature_format: multiview_v1\n"
            f"  {latent_label}:  {z_t_shape}\n"
            f"  z_fwd:          {z_fwd_shape}\n"
            f"  actions:        {list(actions.shape)}\n"
            f"=============================="
        )

    metadata = {
        "dataset_name": dataset_name,
        "language_instruction": lang,
        "num_raw_frames": T_raw,
        "num_fwd_latents": num_fwd_latents,
        "future_rgb_horizon": num_future_raw_frames,
        "vae_temporal_encoding": VAE_TEMPORAL_ENCODING if vae is not None else None,
        "feature_format_version": "multiview_v1",
        "per_view_resolution": list(per_view_resolution),
        "num_main_slots": len(main_frames),
        "num_wrist_slots": len(wrist_frames),
        "main_camera_keys": [k.strip() for k in main_camera_keys.split(",") if k.strip()][:len(main_frames)],
        "wrist_camera_keys": [k.strip() for k in wrist_camera_keys.split(",") if k.strip()][:len(wrist_frames)],
        "z_t_shape": z_t_shape,
        "z_fwd_shape": z_fwd_shape,
        "text_embed_shape": list(context.shape),
        "context_len": int(context.shape[0]),
        "schema_version": 2,
        "proprio_dim": 8,
        "proprio_representation": "eef_xyz_axis_angle_gripper_qpos",
        "proprio_coordinate_frame": "simulator_world",
        "proprio_source": "observation.state",
        "proprio_timestep": "pre_action_observation_t",
        "action_stats_policy": action_normalization,
        "stats_id": stats_id,
        "text_provenance": dict(text_provenance),
        "dinov3": (
            dino_encoder.feature_metadata(storage_dtype="bfloat16")
            if dino_encoder is not None else None
        ),
    }
    with open(ep_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return actions

def main():
    args = parse_args()
    raw_protocol_cfg = {
        "data": {
            **({"future_rgb_horizon": args.future_rgb_horizon} if args.future_rgb_horizon is not None else {}),
            **({"num_future_raw_frames": args.num_future_raw_frames} if args.num_future_raw_frames is not None else {}),
        },
        "model": {
            "visual_encoder": {
                "type": args.visual_encoder,
                "future": {"temporal_downsample_factor": 4},
            },
        },
    }
    _, runtime_protocol = resolve_miniwam_protocol(raw_protocol_cfg)
    args.num_future_raw_frames = runtime_protocol.future_rgb_horizon
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)

    logger.info("Extraction pipeline dtype: %s", args.dtype)
    if args.visual_encoder == "dinov3":
        logger.info("DINO compute dtype:        float32")
        logger.info("DINO feature boundary:     bfloat16")
        logger.info("DINO cache storage dtype:  bfloat16")

    need_vae = args.visual_encoder == "wan_vae"
    vae = load_vae(args.vae_path, device, dtype) if need_vae else None
    dino_encoder = None
    if args.visual_encoder == "dinov3":
        dino_encoder = DINOv3FrameEncoder(
            architecture="vitb16", pretraining="lvd1689m", input_size=224, frozen=True,
            model_id=args.dinov3_model_id,
            revision=args.dinov3_revision,
        ).to(device=device, dtype=torch.float32).eval()
    if args.text_features_source_root:
        text_features_source_root = Path(args.text_features_source_root)
        if not text_features_source_root.is_dir():
            raise FileNotFoundError(
                f"Reusable text feature root does not exist: {text_features_source_root}"
            )
        text_encoder = tokenizer = text_provenance = None
        logger.info("Reusing BF16 text features from: %s", text_features_source_root)
    else:
        text_features_source_root = None
        text_encoder, tokenizer = load_text_encoder(
            args.text_encoder_path, args.tokenizer_id, args.context_len, device, dtype,
        )
        text_provenance = make_text_provenance(
            args.text_encoder_path, args.tokenizer_id, args.context_len,
        )

    all_dataset_statistics = {}
    total_processed = 0
    total_skipped = 0

    datasets = [d.strip() for d in args.data_mix.split(",") if d.strip()]
    stats_datasets = [d.strip() for d in args.stats_data_mix.split(",") if d.strip()]
    if len(datasets) != 1:
        raise ValueError(
            "One LIBERO output directory must contain exactly one dataset; "
            f"got data_mix={datasets}. Run one extraction per suite."
        )
    if not stats_datasets:
        raise ValueError("--stats_data_mix must contain at least one dataset.")

    logger.info("Pass 1: Computing aggregate action and proprio statistics across %s", stats_datasets)
    all_actions_raw_global = []
    all_proprio_raw_global = []

    for dataset_name in stats_datasets:
        info_path = data_root / dataset_name / "meta" / "info.json"
        episodes_path = data_root / dataset_name / "meta" / "episodes.jsonl"

        if not info_path.exists() or not episodes_path.exists():
            raise FileNotFoundError(f"Dataset missing metadata files in {data_root / dataset_name}")

        with open(info_path, "r") as f:
            info_json = json.load(f)
        with open(episodes_path, "r") as f:
            episodes = [json.loads(line) for line in f]

        chunks_size = info_json.get("chunks_size", 1000)
        data_path_template = info_json["data_path"]

        for ep in tqdm(episodes, desc=f"Stats: {dataset_name}"):
            ep_idx = ep["episode_index"]
            ep_chunk = ep_idx // chunks_size
            data_file = data_root / dataset_name / data_path_template.format(episode_chunk=ep_chunk, episode_index=ep_idx)
            try:
                df = pd.read_parquet(str(data_file))
                if "action" in df.columns:
                    acts = np.stack(df["action"].tolist()).astype(np.float32)
                else:
                    action_cols = sorted([c for c in df.columns if c.startswith("action_")], key=lambda x: int(x.split("_")[-1]))
                    acts = df[action_cols].values.astype(np.float32)
                if acts.ndim == 3 and acts.shape[1] == 1:
                    acts = acts[:, 0]
                all_actions_raw_global.append(acts)
                all_proprio_raw_global.append(read_raw_proprio(df))
            except Exception as e:
                raise RuntimeError(f"Failed to read {data_file} for stats") from e

    if not all_actions_raw_global:
        raise RuntimeError("No actions found for any dataset.")

    all_actions_cat = np.concatenate(all_actions_raw_global, axis=0)
    requested_mode = ACTION_MINMAX if args.action_normalization == "minmax" else ACTION_Q01_Q99
    action_stats = compute_action_statistics(all_actions_cat, requested_mode)
    action_lower, action_upper, _, action_normalization = get_action_bounds(
        {"action": action_stats}, "computed LIBERO statistics",
    )
    proprio_stats = compute_proprio_statistics(
        np.concatenate(all_proprio_raw_global, axis=0),
        representation="eef_xyz_axis_angle_gripper_qpos",
        coordinate_frame="simulator_world",
    )

    logger.info("Aggregate action mode: %s", action_normalization)
    logger.info("Aggregate action lower: %s", action_lower)
    logger.info("Aggregate action upper: %s", action_upper)
    logger.info("Aggregate proprio min: %s", proprio_stats["min"])
    logger.info("Aggregate proprio max: %s", proprio_stats["max"])

    # Use LIBERO specific absolute action mask
    absolute_action_mask = [False, False, False, False, False, False, True]

    # 2. Second pass: extract everything and save normalized actions per suite
    for dataset_name in datasets:
        logger.info(f"Processing dataset: {dataset_name}")

        info_path = data_root / dataset_name / "meta" / "info.json"
        episodes_path = data_root / dataset_name / "meta" / "episodes.jsonl"

        if not info_path.exists() or not episodes_path.exists():
            raise FileNotFoundError(f"Dataset missing metadata files in {data_root / dataset_name}")

        with open(info_path, "r") as f:
            info_json = json.load(f)
        with open(episodes_path, "r") as f:
            episodes = [json.loads(line) for line in f]

        if args.max_episodes > 0:
            episodes = episodes[:args.max_episodes]

        stats_id = make_stats_id(
            dataset_name, action_lower, action_upper, proprio_stats, action_normalization,
        )
        all_dataset_statistics[dataset_name] = {
            "action": dict(action_stats),
            "absolute_action_mask": absolute_action_mask,
            "proprio": proprio_stats,
            "schema_version": 2,
            "stats_id": stats_id,
            "aggregation": {
                "scope": "libero_all_suites",
                "datasets": list(stats_datasets),
            },
        }


        if args.num_shards > 1:
            chunk_size = math.ceil(len(episodes) / args.num_shards)
            start_idx = args.shard_id * chunk_size
            end_idx = min(start_idx + chunk_size, len(episodes))
            episodes_to_process = episodes[start_idx:end_idx]
            logger.info(f"Shard {args.shard_id}/{args.num_shards}: Processing episodes {start_idx} to {end_idx-1} ({len(episodes_to_process)} total)")
        else:
            episodes_to_process = episodes

        # 2. Second pass: extract everything and save normalized actions per suite
        logger.info(f"Pass 2: Extracting visual and text features (Shard {args.shard_id})...")
        for ep in tqdm(episodes_to_process, desc=f"{dataset_name} shard {args.shard_id}"):
            acts = process_lerobot_episode(
                data_root, dataset_name, ep, info_json, vae, text_encoder, tokenizer,
                output_dir, device, dtype, args.action_dim,
                args.context_len, args.num_future_raw_frames,
                action_lower, action_upper, action_normalization, stats_id,
                args.per_view_resolution, args.num_main_slots,
                args.num_wrist_slots, args.main_camera_keys, args.wrist_camera_keys,
                args.batch_size, text_provenance, dino_encoder, text_features_source_root,
            )

            if acts is not None:
                total_processed += 1
            else:
                total_skipped += 1

    # Save dataset statistics (only if shard 0 to avoid race conditions)
    if args.shard_id == 0:
        with open(output_dir / "dataset_statistics.pkl", "wb") as f:
            pickle.dump(all_dataset_statistics, f)
        logger.info("Saved dataset statistics to %s", output_dir / "dataset_statistics.pkl")

    logger.info("Done! Processed=%d, Skipped=%d. Output in %s", total_processed, total_skipped, output_dir)

if __name__ == "__main__":
    main()
