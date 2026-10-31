"""Shared, source-controlled utilities for MiniWAM feature extraction."""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from miniwam.models.dinov3 import DINO_MINIWAM_DTYPE
from miniwam.vae_loading import load_wan_video_vae
from miniwam.prompting import (
    DEFAULT_PROMPT,
    format_instruction_prompt,
)
from miniwam.text_encoding import encode_text, load_text_encoder  # noqa: F401 (re-exported)


PROMPT_TEMPLATE = DEFAULT_PROMPT
VAE_TEMPORAL_ENCODING = "joint_current_future_v1"


def _validate_joint_temporal_latents(z_t, z_fwd, source):
    if z_t.numel() == 0 or z_fwd.numel() == 0 or z_t.shape[0] < 2 or z_fwd.shape[0] < 2:
        return
    first_future = z_fwd[:-1].select(-3, 0)
    next_current = z_t[1:].select(-3, 0)
    has_signal = next_current.float().std().item() > 1e-6
    if has_signal and first_future.shape == next_current.shape and torch.equal(first_future, next_current):
        raise ValueError(
            f"{source} has the signature of invalid future-only VAE encoding: the first future "
            "latent is exactly the next frame's standalone current latent. Re-extract with joint "
            "[current + future] causal windows."
        )


def validate_multiview_episode(
    ep_dir,
    dataset_name,
    main_camera_keys,
    wrist_camera_keys,
    proprio_dim,
    *,
    require_vae=True,
    require_dino=False,
):
    ep_dir = Path(ep_dir)
    required = ["text_embed.pt", "actions.npy", "proprio.npy", "metadata.json"]
    if require_vae:
        required.append("vae_latent.pt")
    if require_dino:
        required.append("dino_features.pt")
    missing = [name for name in required if not (ep_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete resumed episode {ep_dir}: missing {missing}.")
    metadata = json.loads((ep_dir / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "dataset_name": str(dataset_name),
        "feature_format_version": "multiview_v1",
        "main_camera_keys": list(main_camera_keys),
        "wrist_camera_keys": list(wrist_camera_keys),
        "proprio_dim": int(proprio_dim),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Existing {ep_dir} has {key}={metadata.get(key)!r}, expected {value!r}.")
    total = int(metadata["num_raw_frames"])
    if require_vae:
        latents = torch.load(ep_dir / "vae_latent.pt", map_location="cpu", weights_only=True, mmap=True)
        expected_keys = {"z_t_main", "z_fwd_main", "main_present", "z_t_wrist", "z_fwd_wrist", "wrist_present"}
        if set(latents) != expected_keys:
            raise KeyError(f"Existing {ep_dir} has an invalid multiview latent payload.")
        if any(latents[key].shape[0] != total for key in expected_keys):
            raise ValueError(f"Existing {ep_dir} latent lengths do not match metadata.")
        encoding = metadata.get("vae_temporal_encoding")
        if encoding not in (None, VAE_TEMPORAL_ENCODING):
            raise ValueError(f"Existing {ep_dir} has unsupported vae_temporal_encoding={encoding!r}.")
        _validate_joint_temporal_latents(latents["z_t_main"], latents["z_fwd_main"], ep_dir)
        _validate_joint_temporal_latents(latents["z_t_wrist"], latents["z_fwd_wrist"], ep_dir)
    if require_dino:
        dino = torch.load(ep_dir / "dino_features.pt", map_location="cpu", weights_only=True, mmap=True)
        expected_dino_keys = {"dino_main", "main_present", "dino_wrist", "wrist_present"}
        if set(dino) != expected_dino_keys:
            raise KeyError(f"Existing {ep_dir} has an invalid DINO payload.")
        if dino["dino_main"].shape[0] != total or dino["dino_wrist"].shape[0] != total:
            raise ValueError(f"Existing {ep_dir} DINO lengths do not match metadata.")
        if dino["dino_main"].dtype != DINO_MINIWAM_DTYPE or dino["dino_wrist"].dtype != DINO_MINIWAM_DTYPE:
            raise ValueError(f"Existing {ep_dir} DINO cache must use the BF16 MiniWAM feature boundary.")
        dino_metadata = metadata.get("dinov3") or {}
        expected_dtypes = {
            "compute_dtype": "float32",
            "boundary_dtype": "bfloat16",
            "storage_dtype": "bfloat16",
        }
        mismatches = {
            key: (dino_metadata.get(key), value)
            for key, value in expected_dtypes.items()
            if dino_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing {ep_dir} DINO precision metadata mismatch: {mismatches}.")
    actions = np.load(ep_dir / "actions.npy", mmap_mode="r")
    proprio = np.load(ep_dir / "proprio.npy", mmap_mode="r")
    if actions.shape[0] != total or proprio.shape != (total, proprio_dim):
        raise ValueError(f"Existing {ep_dir} action/proprio lengths do not match metadata.")


def format_prompt(instruction):
    return format_instruction_prompt(instruction)


def make_text_provenance(text_encoder_path, tokenizer_id, context_len):
    """Fingerprint the exact text assets used by a fresh extraction run."""
    path = Path(text_encoder_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "prompt_template": PROMPT_TEMPLATE,
        "text_encoder_sha256": digest.hexdigest(),
        "text_encoder_filename": path.name,
        "tokenizer_id": str(tokenizer_id),
        "context_len": int(context_len),
    }


def load_vae(path, device, dtype):
    return load_wan_video_vae(path, device=device, dtype=dtype)


def resize_and_normalize_frames(frames, resolution):
    frames = torch.as_tensor(frames)
    if frames.ndim != 4:
        raise ValueError(f"Frames must be [T,H,W,C] or [T,C,H,W], got {tuple(frames.shape)}.")
    if frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)
    elif frames.shape[1] != 3:
        raise ValueError(f"Frames must have three RGB channels, got {tuple(frames.shape)}.")
    if frames.shape[0] == 0:
        raise ValueError("Cannot extract features from an empty frame sequence.")
    frames = frames.float()
    if frames.max().item() > 1.0:
        frames = frames / 255.0
    frames = F.interpolate(frames, size=tuple(resolution), mode="bilinear", align_corners=False)
    return frames.mul(2).sub(1).contiguous()


@torch.no_grad()
def encode_vae_windows(windows, vae, device, dtype, num_future_raw_frames, batch_size):
    """Encode explicit joint current+future windows with the training cache contract."""
    if windows.ndim != 5 or windows.shape[2] != 3:
        raise ValueError(f"VAE windows must be [B,T,3,H,W], got {tuple(windows.shape)}.")
    if num_future_raw_frames <= 0 or num_future_raw_frames % 4:
        raise ValueError("num_future_raw_frames must be positive and divisible by four.")
    if windows.shape[1] != 1 + num_future_raw_frames:
        raise ValueError(
            f"VAE windows must contain {1 + num_future_raw_frames} frames, got {windows.shape[1]}."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    total = int(windows.shape[0])
    if total == 0:
        raise ValueError("Cannot encode an empty VAE window batch.")
    expected_future = num_future_raw_frames // 4
    expected_total = 1 + expected_future
    current_parts, future_parts = [], []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        latents = vae.single_encode(
            windows[start:end].permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype),
            device=device,
        )
        if latents.shape[2] != expected_total:
            raise ValueError(
                f"Joint VAE window must produce {expected_total} latent frames, got {tuple(latents.shape)}."
            )
        current_parts.append(latents[:, :, :1].detach().cpu().to(torch.bfloat16))
        future_parts.append(latents[:, :, 1:].detach().cpu().to(torch.bfloat16))
    return torch.cat(current_parts), torch.cat(future_parts)


@torch.no_grad()
def encode_rolling_vae_latents(frames_norm, vae, device, dtype, num_future_raw_frames, batch_size):
    if frames_norm.ndim != 4 or frames_norm.shape[1] != 3:
        raise ValueError(f"Normalized frames must be [T,3,H,W], got {tuple(frames_norm.shape)}.")
    total = int(frames_norm.shape[0])
    if total == 0:
        raise ValueError("Cannot encode an empty episode.")
    current_parts, future_parts = [], []
    for start in range(0, total, batch_size):
        windows = []
        for timestep in range(start, min(start + batch_size, total)):
            window = frames_norm[timestep:timestep + 1 + num_future_raw_frames]
            if window.shape[0] < 1 + num_future_raw_frames:
                window = torch.cat((
                    window,
                    frames_norm[-1:].expand(1 + num_future_raw_frames - len(window), -1, -1, -1),
                ))
            windows.append(window)
        z_t, z_fwd = encode_vae_windows(
            torch.stack(windows), vae, device, dtype, num_future_raw_frames, batch_size,
        )
        current_parts.append(z_t)
        future_parts.append(z_fwd)
    return torch.cat(current_parts), torch.cat(future_parts)


def stack_slot_latents(z_t_slots, z_fwd_slots, total, num_fwd, latent_h, latent_w):
    if len(z_t_slots) != len(z_fwd_slots):
        raise ValueError("Current and future slot lists must have the same length.")
    if not z_t_slots:
        return (
            torch.empty((total, 0, 16, 1, latent_h, latent_w), dtype=torch.bfloat16),
            torch.empty((total, 0, 16, num_fwd, latent_h, latent_w), dtype=torch.bfloat16),
        )
    z_t, z_fwd = torch.stack(z_t_slots, 1), torch.stack(z_fwd_slots, 1)
    if z_t.shape[:2] != (total, len(z_t_slots)) or z_fwd.shape[3] != num_fwd:
        raise ValueError(f"Invalid stacked slot shapes: current={tuple(z_t.shape)}, future={tuple(z_fwd.shape)}.")
    return z_t, z_fwd


@torch.no_grad()
def encode_dinov3_frames(frames, encoder, device, batch_size=64):
    """Encode physical frames in FP32 and store the BF16 MiniWAM boundary values."""
    frames = torch.as_tensor(frames)
    if frames.ndim != 4:
        raise ValueError(f"DINO frames must be [T,H,W,C] or [T,C,H,W], got {tuple(frames.shape)}.")
    if frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)
    if frames.shape[1] != 3 or frames.shape[0] == 0:
        raise ValueError(f"Invalid DINO frame tensor {tuple(frames.shape)}.")
    outputs = []
    for start in range(0, frames.shape[0], int(batch_size)):
        batch = frames[start:start + int(batch_size)].to(device=device)
        outputs.append(encoder.encode_for_miniwam(batch).detach().cpu().contiguous())
    result = torch.cat(outputs)
    if result.dtype != DINO_MINIWAM_DTYPE:
        raise RuntimeError(f"DINO cache boundary must be BF16, got {result.dtype}.")
    if not torch.isfinite(result).all():
        raise ValueError("DINO feature cache contains non-finite values.")
    return result


def stack_slot_dino_features(
    slot_features, total, num_patches=196, feature_dim=768,
    storage_dtype=DINO_MINIWAM_DTYPE,
):
    if not slot_features:
        return torch.empty((total, 0, num_patches, feature_dim), dtype=storage_dtype)
    result = torch.stack(slot_features, dim=1)
    expected = (total, len(slot_features), num_patches, feature_dim)
    if tuple(result.shape) != expected:
        raise ValueError(f"Invalid stacked DINO cache: expected {expected}, got {tuple(result.shape)}.")
    return result
