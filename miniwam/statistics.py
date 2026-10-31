"""Fail-fast loading and exact selection of dataset statistics."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np


ACTION_MINMAX = "minmax_to_minus1_plus1"
ACTION_Q01_Q99 = "q01_q99_to_minus1_plus1"
ACTION_ZSCORE = "zscore_clip"
DEFAULT_ZSCORE_EPSILON = 1e-8
DEFAULT_ZSCORE_CLIP = 5.0


def compute_action_statistics(
    values: np.ndarray,
    normalization: str = ACTION_MINMAX,
    epsilon: float = 1e-6,
    clip: float = DEFAULT_ZSCORE_CLIP,
) -> dict:
    """Compute typed action bounds without collapsing non-constant dimensions."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Raw actions must be a non-empty [T,D] array, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Raw actions contain non-finite values.")

    raw_min = values.min(axis=0)
    raw_max = values.max(axis=0)
    raw_constant = (raw_max - raw_min) <= epsilon
    if normalization == ACTION_ZSCORE:
        if clip <= 0:
            raise ValueError(f"Z-score clip must be positive, got {clip}.")
        # RoboTwin uses global, per-dimension statistics. Use
        # float64 population moments for deterministic accumulation and persist
        # float32 parameters, which are then shared by training and inference.
        values64 = values.astype(np.float64)
        mean = values64.mean(axis=0).astype(np.float32)
        std = values64.std(axis=0, ddof=0).astype(np.float32)
        return {
            "normalization": ACTION_ZSCORE,
            "mean": mean,
            "std": std,
            "min": raw_min,
            "max": raw_max,
            "constant_mask": raw_constant,
            "epsilon": float(DEFAULT_ZSCORE_EPSILON),
            "clip": float(clip),
            "num_samples": int(values.shape[0]),
        }
    if normalization == ACTION_MINMAX:
        return {
            "normalization": ACTION_MINMAX,
            "min": raw_min,
            "max": raw_max,
            "constant_mask": raw_constant,
        }
    if normalization != ACTION_Q01_Q99:
        raise ValueError(f"Unsupported action normalization {normalization!r}.")

    lower = np.percentile(values, 1, axis=0).astype(np.float32)
    upper = np.percentile(values, 99, axis=0).astype(np.float32)
    collapsed = ((upper - lower) <= epsilon) & ~raw_constant
    # Percentiles can erase rare states of binary/discrete controls. Preserve the
    # observed support for every affected dimension instead of making it constant.
    lower[collapsed] = raw_min[collapsed]
    upper[collapsed] = raw_max[collapsed]
    return {
        "normalization": ACTION_Q01_Q99,
        "q01": lower,
        "q99": upper,
        "constant_mask": raw_constant,
        "percentile_fallback_mask": collapsed,
    }


def get_action_zscore(
    stats: Mapping,
    source: str | Path = "statistics",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, str]:
    """Return validated global z-score parameters for an action statistics block."""
    action = stats.get("action") if isinstance(stats, Mapping) else None
    if not isinstance(action, Mapping):
        raise KeyError(f"{source} is missing an action statistics mapping.")
    mode = action.get("normalization")
    if mode != ACTION_ZSCORE:
        raise ValueError(f"Action normalization in {source} is {mode!r}, expected {ACTION_ZSCORE!r}.")
    mean = np.asarray(action.get("mean"), dtype=np.float32)
    std = np.asarray(action.get("std"), dtype=np.float32)
    if mean.ndim != 1 or mean.shape != std.shape:
        raise ValueError(f"Invalid action mean/std in {source}: {mean.shape} and {std.shape}.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError(f"Action mean/std in {source} must be finite with std >= 0.")
    declared_constant = action.get("constant_mask")
    constant = std <= 1e-6 if declared_constant is None else np.asarray(declared_constant, dtype=bool)
    if constant.shape != mean.shape:
        raise ValueError(
            f"Action constant_mask in {source} must have shape {mean.shape}, got {constant.shape}."
        )
    epsilon = float(action.get("epsilon", DEFAULT_ZSCORE_EPSILON))
    clip = float(action.get("clip", DEFAULT_ZSCORE_CLIP))
    if epsilon <= 0 or clip <= 0:
        raise ValueError(f"Action epsilon and clip in {source} must be positive.")
    return mean, std, constant, epsilon, clip, ACTION_ZSCORE


def validate_action_statistics(stats: Mapping, source: str | Path = "statistics") -> str:
    """Validate any supported action statistics mode and return its canonical mode."""
    action = stats.get("action") if isinstance(stats, Mapping) else None
    if not isinstance(action, Mapping):
        raise KeyError(f"{source} is missing an action statistics mapping.")
    mode = action.get("normalization")
    if mode == ACTION_ZSCORE:
        return get_action_zscore(stats, source)[-1]
    return get_action_bounds(stats, source)[-1]


def get_action_bounds(stats: Mapping, source: str | Path = "statistics") -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return validated action lower/upper bounds, constant mask, and mode."""
    action = stats.get("action") if isinstance(stats, Mapping) else None
    if not isinstance(action, Mapping):
        raise KeyError(f"{source} is missing an action statistics mapping.")
    mode = action.get("normalization")
    if mode == ACTION_MINMAX:
        lower_key, upper_key = "min", "max"
    elif mode == ACTION_Q01_Q99:
        lower_key, upper_key = "q01", "q99"
    else:
        raise ValueError(f"Unsupported action normalization {mode!r} in {source}.")
    if lower_key not in action or upper_key not in action:
        raise KeyError(
            f"Action normalization {mode!r} in {source} requires "
            f"action.{lower_key} and action.{upper_key}."
        )
    lower = np.asarray(action[lower_key], dtype=np.float32)
    upper = np.asarray(action[upper_key], dtype=np.float32)
    if lower.ndim != 1 or lower.shape != upper.shape:
        raise ValueError(f"Invalid action bounds in {source}: {lower.shape} and {upper.shape}.")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError(f"Action bounds in {source} contain non-finite values.")
    if np.any(upper < lower):
        raise ValueError(f"Action bounds in {source} contain upper < lower.")
    declared_constant = action.get("constant_mask")
    if declared_constant is None:
        constant = (upper - lower) <= 1e-6
    else:
        constant = np.asarray(declared_constant, dtype=bool)
        if constant.shape != lower.shape:
            raise ValueError(
                f"Action constant_mask in {source} must have shape {lower.shape}, got {constant.shape}."
            )
        if not np.array_equal(constant, (upper - lower) <= 1e-6):
            raise ValueError(f"Action constant_mask in {source} disagrees with its bounds.")
    return lower, upper, constant, str(mode)


def load_statistics_mapping(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset statistics file does not exist: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{path} must contain a non-empty dataset-keyed mapping.")
    if "action" in payload:
        raise ValueError(f"{path} uses an ambiguous unkeyed statistics schema; expected {{dataset_name: statistics}}.")
    return dict(payload)


def select_statistics(payload: Mapping, dataset_key: str, source: str | Path) -> dict:
    if not isinstance(dataset_key, str) or not dataset_key.strip():
        raise ValueError("A non-empty canonical dataset key is required for statistics selection.")
    if dataset_key not in payload:
        raise KeyError(f"Dataset {dataset_key!r} is absent from {source}; available={sorted(payload)}.")
    stats = payload[dataset_key]
    if not isinstance(stats, Mapping):
        raise TypeError(f"Statistics for {dataset_key!r} in {source} must be a mapping.")
    validate_action_statistics(stats, f"statistics for {dataset_key!r} in {source}")
    if not stats.get("stats_id"):
        raise KeyError(f"Statistics for {dataset_key!r} in {source} are missing stats_id.")
    return dict(stats)
