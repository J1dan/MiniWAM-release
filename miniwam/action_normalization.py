"""Action normalization helpers shared by training and deployment code."""

from collections.abc import Mapping

import numpy as np
import torch

from .statistics import ACTION_ZSCORE, get_action_bounds, get_action_zscore


def normalize_actions(
    actions: np.ndarray | torch.Tensor,
    stats: Mapping,
) -> np.ndarray | torch.Tensor:
    """Normalize actions with the typed statistics used by their dataset."""
    action_stats = stats.get("action") if isinstance(stats, Mapping) else None
    if not isinstance(action_stats, Mapping):
        raise KeyError("Statistics are missing an action mapping.")
    if action_stats.get("normalization") == ACTION_ZSCORE:
        mean, std, _constant, epsilon, clip, _ = get_action_zscore(stats)
        if actions.shape[-1] != len(mean):
            raise ValueError(f"Action/statistics shape mismatch: {actions.shape[-1]} vs {len(mean)}.")
        if isinstance(actions, torch.Tensor):
            mean_t = torch.as_tensor(mean, device=actions.device, dtype=actions.dtype)
            std_t = torch.as_tensor(std, device=actions.device, dtype=actions.dtype)
            return ((actions - mean_t) / (std_t + epsilon)).clamp(-clip, clip)
        result = (np.asarray(actions, dtype=np.float32) - mean) / (std + epsilon)
        return np.clip(result, -clip, clip).astype(np.float32, copy=False)

    lower, upper, constant, _ = get_action_bounds(stats)
    if actions.shape[-1] != len(lower):
        raise ValueError(f"Action/statistics shape mismatch: {actions.shape[-1]} vs {len(lower)}.")
    if isinstance(actions, torch.Tensor):
        lower_t = torch.as_tensor(lower, device=actions.device, dtype=actions.dtype)
        upper_t = torch.as_tensor(upper, device=actions.device, dtype=actions.dtype)
        constant_t = torch.as_tensor(constant, device=actions.device, dtype=torch.bool)
        result = (2.0 * (actions - lower_t) / (upper_t - lower_t).clamp_min(1e-8) - 1.0).clamp(-1, 1)
        return result.masked_fill(constant_t, 0.0)
    result = 2.0 * (np.asarray(actions, dtype=np.float32) - lower) / np.maximum(upper - lower, 1e-8) - 1.0
    result = np.clip(result, -1.0, 1.0)
    result[..., constant] = 0.0
    return result.astype(np.float32, copy=False)


def denormalize_actions(
    normalized_actions: np.ndarray | torch.Tensor,
    stats: Mapping,
) -> np.ndarray | torch.Tensor:
    """Invert typed action normalization, clipping in normalized space first."""
    action_stats = stats.get("action") if isinstance(stats, Mapping) else None
    if not isinstance(action_stats, Mapping):
        raise KeyError("Statistics are missing an action mapping.")
    if action_stats.get("normalization") == ACTION_ZSCORE:
        mean, std, _constant, epsilon, clip, _ = get_action_zscore(stats)
        if normalized_actions.shape[-1] != len(mean):
            raise ValueError(
                f"Action/statistics shape mismatch: {normalized_actions.shape[-1]} vs {len(mean)}."
            )
        if isinstance(normalized_actions, torch.Tensor):
            mean_t = torch.as_tensor(mean, device=normalized_actions.device, dtype=normalized_actions.dtype)
            std_t = torch.as_tensor(std, device=normalized_actions.device, dtype=normalized_actions.dtype)
            return normalized_actions.clamp(-clip, clip) * (std_t + epsilon) + mean_t
        return (
            np.clip(np.asarray(normalized_actions, dtype=np.float32), -clip, clip)
            * (std + epsilon)
            + mean
        ).astype(np.float32, copy=False)

    lower, upper, constant, _ = get_action_bounds(stats)
    if isinstance(normalized_actions, torch.Tensor):
        return denormalize_clipped_actions(
            normalized_actions,
            torch.as_tensor(lower, device=normalized_actions.device, dtype=normalized_actions.dtype),
            torch.as_tensor(upper, device=normalized_actions.device, dtype=normalized_actions.dtype),
            torch.as_tensor(constant, device=normalized_actions.device),
        )
    bounded = np.clip(np.asarray(normalized_actions, dtype=np.float32), -1.0, 1.0)
    result = (bounded + 1.0) * 0.5 * (upper - lower) + lower
    result[..., constant] = lower[constant]
    return result.astype(np.float32, copy=False)


def denormalize_clipped_actions(
    normalized_actions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    constant_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project predictions to the training support before bounds denormalization."""
    if normalized_actions.shape[-1] != lower.numel() or lower.shape != upper.shape:
        raise ValueError(
            "Action/statistics shape mismatch: "
            f"actions={tuple(normalized_actions.shape)}, lower={tuple(lower.shape)}, "
            f"upper={tuple(upper.shape)}."
        )
    bounded_actions = normalized_actions.clamp(-1.0, 1.0)
    result = (bounded_actions + 1.0) / 2.0 * (upper - lower) + lower
    if constant_mask is not None:
        if constant_mask.shape != lower.shape:
            raise ValueError(
                f"Action constant_mask must have shape {tuple(lower.shape)}, "
                f"got {tuple(constant_mask.shape)}."
            )
        result = torch.where(constant_mask.to(device=result.device), lower, result)
    return result
