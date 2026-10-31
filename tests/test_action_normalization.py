import numpy as np
import pytest
import torch

from miniwam.action_normalization import (
    denormalize_actions,
    denormalize_clipped_actions,
    normalize_actions,
)
from miniwam.statistics import ACTION_ZSCORE


def test_denormalization_clips_predictions_to_training_support():
    predictions = torch.tensor([[-1.5, 0.0, 1.5]])
    q01 = torch.tensor([-2.0, 10.0, 20.0])
    q99 = torch.tensor([2.0, 14.0, 24.0])

    result = denormalize_clipped_actions(predictions, q01, q99)

    torch.testing.assert_close(result, torch.tensor([[-2.0, 12.0, 24.0]]))


def test_denormalization_rejects_action_statistics_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        denormalize_clipped_actions(
            torch.zeros(2, 7),
            torch.zeros(6),
            torch.ones(6),
        )


def test_denormalization_preserves_constant_dimensions():
    predictions = torch.tensor([[-1.0, 1.0], [0.5, -0.5]])
    lower = torch.tensor([3.0, -2.0])
    upper = torch.tensor([3.0, 2.0])
    constant = torch.tensor([True, False])

    result = denormalize_clipped_actions(predictions, lower, upper, constant)

    torch.testing.assert_close(result, torch.tensor([[3.0, 2.0], [3.0, -1.0]]))


def test_denormalization_rejects_constant_mask_shape_mismatch():
    with pytest.raises(ValueError, match="constant_mask"):
        denormalize_clipped_actions(
            torch.zeros(1, 2),
            torch.zeros(2),
            torch.ones(2),
            torch.zeros(1, dtype=torch.bool),
        )


def test_zscore_is_shared_by_numpy_and_torch_and_clips_at_five():
    stats = {
        "action": {
            "normalization": ACTION_ZSCORE,
            "mean": np.array([1.0, -2.0], dtype=np.float32),
            "std": np.array([2.0, 4.0], dtype=np.float32),
            "constant_mask": np.array([False, False]),
            "epsilon": 1e-8,
            "clip": 5.0,
        }
    }
    raw = np.array([[3.0, 38.0], [-19.0, -42.0]], dtype=np.float32)

    normalized_numpy = normalize_actions(raw, stats)
    normalized_torch = normalize_actions(torch.from_numpy(raw), stats)

    np.testing.assert_allclose(normalized_numpy, [[1.0, 5.0], [-5.0, -5.0]])
    torch.testing.assert_close(normalized_torch, torch.from_numpy(normalized_numpy))
    np.testing.assert_allclose(
        denormalize_actions(normalized_numpy, stats),
        [[3.0, 18.0], [-9.0, -22.0]],
    )
