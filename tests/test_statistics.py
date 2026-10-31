import pickle
import numpy as np
import pytest

from miniwam.statistics import (
    ACTION_MINMAX,
    ACTION_Q01_Q99,
    ACTION_ZSCORE,
    compute_action_statistics,
    get_action_bounds,
    get_action_zscore,
    load_statistics_mapping,
    select_statistics,
)


def _entry(stats_id):
    return {
        "action": {"normalization": ACTION_Q01_Q99, "q01": np.zeros(7), "q99": np.ones(7)},
        "stats_id": stats_id,
    }


def test_selection_is_exact_and_order_independent(tmp_path):
    path = tmp_path / "stats.pkl"
    with path.open("wb") as handle:
        pickle.dump({"synthetic_shifted": _entry("g"), "synthetic_reference": _entry("b")}, handle)
    payload = load_statistics_mapping(path)
    assert select_statistics(payload, "synthetic_reference", path)["stats_id"] == "b"
    with pytest.raises(KeyError):
        select_statistics(payload, "missing", path)


def test_single_entry_is_not_a_name_fallback(tmp_path):
    path = tmp_path / "stats.pkl"
    with path.open("wb") as handle:
        pickle.dump({"wrong": _entry("id")}, handle)
    with pytest.raises(KeyError):
        select_statistics(load_statistics_mapping(path), "synthetic_reference", path)


def test_unkeyed_schema_is_rejected(tmp_path):
    path = tmp_path / "stats.pkl"
    with path.open("wb") as handle:
        pickle.dump(_entry("id"), handle)
    with pytest.raises(ValueError, match="unkeyed"):
        load_statistics_mapping(path)


def test_explicit_minmax_action_bounds_are_selected():
    stats = {
        "action": {
            "normalization": ACTION_MINMAX,
            "min": np.array([-2.0, 0.5, 1.0], dtype=np.float32),
            "max": np.array([2.0, 0.5, 3.0], dtype=np.float32),
            "constant_mask": np.array([False, True, False]),
        }
    }

    lower, upper, constant, mode = get_action_bounds(stats)

    np.testing.assert_array_equal(lower, [-2.0, 0.5, 1.0])
    np.testing.assert_array_equal(upper, [2.0, 0.5, 3.0])
    np.testing.assert_array_equal(constant, [False, True, False])
    assert mode == ACTION_MINMAX


def test_declared_constant_mask_must_match_bounds():
    stats = {
        "action": {
            "normalization": ACTION_MINMAX,
            "min": np.zeros(2),
            "max": np.ones(2),
            "constant_mask": np.array([True, False]),
        }
    }

    with pytest.raises(ValueError, match="constant_mask.*disagrees"):
        get_action_bounds(stats)


def test_action_statistics_default_to_minmax():
    values = np.array([[-2.0, -1.0], [1.0, 1.0]], dtype=np.float32)

    stats = compute_action_statistics(values)

    assert stats["normalization"] == ACTION_MINMAX
    np.testing.assert_array_equal(stats["min"], [-2.0, -1.0])
    np.testing.assert_array_equal(stats["max"], [1.0, 1.0])
    np.testing.assert_array_equal(stats["constant_mask"], [False, False])


def test_percentile_mode_preserves_rare_binary_action_state():
    values = np.zeros((1000, 2), dtype=np.float32)
    values[:, 0] = -1.0
    values[-5:, 0] = 1.0
    values[:, 1] = 3.0

    stats = compute_action_statistics(values, ACTION_Q01_Q99)

    np.testing.assert_array_equal(stats["q01"], [-1.0, 3.0])
    np.testing.assert_array_equal(stats["q99"], [1.0, 3.0])
    np.testing.assert_array_equal(stats["constant_mask"], [False, True])
    np.testing.assert_array_equal(stats["percentile_fallback_mask"], [True, False])


def test_global_zscore_uses_population_moments_and_fixed_constants():
    values = np.array([[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]], dtype=np.float32)

    stats = compute_action_statistics(values, ACTION_ZSCORE)
    mean, std, constant, epsilon, clip, mode = get_action_zscore({"action": stats})

    np.testing.assert_allclose(mean, [3.0, 4.0])
    np.testing.assert_allclose(std, [np.sqrt(8.0 / 3.0), 0.0])
    np.testing.assert_array_equal(constant, [False, True])
    assert epsilon == 1e-8
    assert clip == 5.0
    assert mode == ACTION_ZSCORE
