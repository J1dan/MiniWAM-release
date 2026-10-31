"""Proprioception schema, normalization, and statistics provenance helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch


DEFAULT_PROPRIO_CLIP = 5.0
PROPRIO_MINMAX = "minmax_to_minus1_plus1"
PROPRIO_ZSCORE = "zscore_clip"
DEFAULT_PROPRIO_EPSILON = 1e-8


def compute_proprio_statistics(
    values: np.ndarray,
    representation: str,
    coordinate_frame: str,
    clip: float = DEFAULT_PROPRIO_CLIP,
    normalization: str = PROPRIO_MINMAX,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Raw proprio must be a non-empty [T,D] array, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Raw proprio contains non-finite values.")
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    values64 = values.astype(np.float64)
    variance = values64.var(axis=0, ddof=0).astype(np.float32)
    constant = (variance <= 1e-12) | ((maximum - minimum) <= 1e-6)
    common = {
        "min": minimum,
        "max": maximum,
        "variance": variance,
        "constant_mask": constant,
        "dim": int(values.shape[1]),
        "representation": representation,
        "coordinate_frame": coordinate_frame,
        "normalization": normalization,
        "clip": float(clip),
        "num_samples": int(values.shape[0]),
    }
    if normalization == PROPRIO_MINMAX:
        return common
    if normalization == PROPRIO_ZSCORE:
        return {
            **common,
            "mean": values64.mean(axis=0).astype(np.float32),
            "std": values64.std(axis=0, ddof=0).astype(np.float32),
            "epsilon": float(DEFAULT_PROPRIO_EPSILON),
        }
    raise ValueError(f"Unsupported proprio normalization: {normalization!r}")


def make_stats_id(
    dataset_name: str,
    action_lower: np.ndarray,
    action_upper: np.ndarray,
    proprio_stats: Mapping[str, Any],
    action_normalization: str = "q01_q99_to_minus1_plus1",
    source_manifest_hash: str | None = None,
) -> str:
    proprio_mode = str(proprio_stats.get("normalization", PROPRIO_MINMAX))
    is_v3 = action_normalization == "zscore_clip" or proprio_mode == PROPRIO_ZSCORE
    digest = hashlib.sha256()
    digest.update(str(dataset_name).encode("utf-8"))
    if is_v3:
        digest.update(str(action_normalization).encode("utf-8"))
    digest.update(np.asarray(action_lower, dtype=np.float32).tobytes())
    digest.update(np.asarray(action_upper, dtype=np.float32).tobytes())
    proprio_keys = (
        ("mean", "std", "constant_mask")
        if proprio_mode == PROPRIO_ZSCORE
        else ("min", "max", "constant_mask")
    )
    for key in proprio_keys:
        digest.update(np.asarray(proprio_stats[key]).tobytes())
    if is_v3:
        digest.update(proprio_mode.encode("utf-8"))
        digest.update(str(proprio_stats.get("epsilon", "")).encode("utf-8"))
        digest.update(str(proprio_stats.get("clip", "")).encode("utf-8"))
    digest.update(str(proprio_stats["representation"]).encode("utf-8"))
    digest.update(str(proprio_stats["coordinate_frame"]).encode("utf-8"))
    if is_v3:
        digest.update(str(source_manifest_hash or "").encode("utf-8"))
    version = 3 if is_v3 else 2
    return f"{dataset_name}_v{version}_{digest.hexdigest()[:12]}"


def validate_proprio_statistics(stats: Mapping[str, Any], expected_dim: int) -> dict[str, Any]:
    """Validate and canonicalize one dataset's proprio statistics block."""
    required = {
        "constant_mask", "dim", "representation", "coordinate_frame", "normalization", "clip",
    }
    missing = required - set(stats)
    if missing:
        raise KeyError(f"Proprio statistics are missing fields: {sorted(missing)}")
    dim = int(stats["dim"])
    if dim != expected_dim:
        raise ValueError(f"Proprio statistics dim={dim}, expected {expected_dim}.")
    constant_mask = np.asarray(stats["constant_mask"], dtype=bool)
    if constant_mask.shape != (dim,):
        raise ValueError(f"Proprio constant_mask must have shape ({dim},), got {constant_mask.shape}.")
    mode = str(stats["normalization"])
    canonical = {**dict(stats), "constant_mask": constant_mask, "dim": dim}
    if mode == PROPRIO_MINMAX:
        missing_bounds = {"min", "max"} - set(stats)
        if missing_bounds:
            raise KeyError(f"Proprio statistics are missing fields: {sorted(missing_bounds)}")
        minimum = np.asarray(stats["min"], dtype=np.float32)
        maximum = np.asarray(stats["max"], dtype=np.float32)
        if minimum.shape != (dim,) or maximum.shape != (dim,):
            raise ValueError(f"Proprio min/max must have shape ({dim},).")
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("Proprio statistics contain non-finite bounds.")
        if np.any(maximum < minimum):
            raise ValueError("Proprio statistics contain max < min.")
        canonical.update({"min": minimum, "max": maximum})
    elif mode == PROPRIO_ZSCORE:
        missing_moments = {"mean", "std"} - set(stats)
        if missing_moments:
            raise KeyError(f"Proprio statistics are missing fields: {sorted(missing_moments)}")
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        if mean.shape != (dim,) or std.shape != (dim,):
            raise ValueError(f"Proprio mean/std must have shape ({dim},).")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError("Proprio mean/std must be finite with std >= 0.")
        epsilon = float(stats.get("epsilon", DEFAULT_PROPRIO_EPSILON))
        if epsilon <= 0:
            raise ValueError(f"Proprio epsilon must be positive, got {epsilon}.")
        canonical.update({"mean": mean, "std": std, "epsilon": epsilon})
    else:
        raise ValueError(f"Unsupported proprio normalization: {mode!r}")
    clip = float(stats["clip"])
    if clip <= 0:
        raise ValueError(f"Proprio clip must be positive, got {clip}.")
    canonical["clip"] = clip
    return canonical


def normalize_proprio(
    proprio: np.ndarray | torch.Tensor,
    stats: Mapping[str, Any],
) -> np.ndarray | torch.Tensor:
    """Normalize raw proprio according to its typed, dataset-owned statistics."""
    dim = int(stats["dim"])
    canonical = validate_proprio_statistics(stats, dim)
    if proprio.shape[-1] != dim:
        raise ValueError(f"Proprio last dim must be {dim}, got {proprio.shape[-1]}.")

    if isinstance(proprio, torch.Tensor):
        constant = torch.as_tensor(
            canonical["constant_mask"], device=proprio.device, dtype=torch.bool,
        )
        if canonical["normalization"] == PROPRIO_ZSCORE:
            mean = torch.as_tensor(canonical["mean"], device=proprio.device, dtype=proprio.dtype)
            std = torch.as_tensor(canonical["std"], device=proprio.device, dtype=proprio.dtype)
            normalized = (proprio - mean) / (std + canonical["epsilon"])
            return normalized.clamp(-canonical["clip"], canonical["clip"])
        else:
            minimum = torch.as_tensor(canonical["min"], device=proprio.device, dtype=proprio.dtype)
            maximum = torch.as_tensor(canonical["max"], device=proprio.device, dtype=proprio.dtype)
            normalized = 2.0 * (proprio - minimum) / (maximum - minimum).clamp_min(1e-8) - 1.0
        normalized = normalized.clamp(-canonical["clip"], canonical["clip"])
        return normalized.masked_fill(constant, 0.0)

    raw = np.asarray(proprio, dtype=np.float32)
    if canonical["normalization"] == PROPRIO_ZSCORE:
        normalized = (raw - canonical["mean"]) / (canonical["std"] + canonical["epsilon"])
        return np.clip(
            normalized, -canonical["clip"], canonical["clip"],
        ).astype(np.float32, copy=False)
    else:
        normalized = 2.0 * (raw - canonical["min"]) / np.maximum(
            canonical["max"] - canonical["min"], 1e-8,
        ) - 1.0
    normalized = np.clip(normalized, -canonical["clip"], canonical["clip"])
    normalized[..., canonical["constant_mask"]] = 0.0
    return normalized.astype(np.float32, copy=False)
