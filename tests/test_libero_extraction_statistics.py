import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_episode_limit_preserves_full_statistics_split(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/data"))
    extractor = importlib.import_module("scripts.data.extract_dataset_feature_libero")
    suites = ("libero_10_no_noops_lerobot", "libero_goal_no_noops_lerobot")
    for suite_index, suite in enumerate(suites):
        root = tmp_path / suite
        (root / "meta").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "meta/info.json").write_text(json.dumps({
            "chunks_size": 1000, "data_path": "data/episode_{episode_index:06d}.parquet"}))
        episodes = [{"episode_index": index, "tasks": ["test instruction"], "length": 2}
                    for index in range(2)]
        (root / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
        for index in range(2):
            value = float(20 * suite_index + 10 * index)
            pd.DataFrame({
                "action": [np.full(7, value, dtype=np.float32)] * 2,
                "observation.state": [np.full(8, value, dtype=np.float32)] * 2,
            }).to_parquet(root / f"data/episode_{index:06d}.parquet")

    class Encoder:
        def __init__(self, **kwargs):
            pass

        def to(self, **kwargs):
            return self

        def eval(self):
            return self

    extracted = []

    def extract_episode(*args):
        extracted.append((args[1], args[2]["episode_index"]))
        return np.zeros((2, 7), dtype=np.float32)

    monkeypatch.setattr(extractor, "DINOv3FrameEncoder", Encoder)
    monkeypatch.setattr(extractor, "process_lerobot_episode", extract_episode)
    output = tmp_path / "features"
    monkeypatch.setattr(sys, "argv", [
        "extract", "--data_root", str(tmp_path), "--data_mix", suites[0],
        "--stats_data_mix", ",".join(suites), "--output_dir", str(output),
        "--visual_encoder", "dinov3", "--device", "cpu", "--max_episodes", "1",
        "--text_features_source_root", str(tmp_path),
    ])
    extractor.main()

    assert extracted == [(suites[0], 0)]
    with (output / "dataset_statistics.pkl").open("rb") as stream:
        stats = pickle.load(stream)[suites[0]]
    np.testing.assert_array_equal(stats["action"]["min"], np.zeros(7))
    np.testing.assert_array_equal(stats["action"]["max"], np.full(7, 30.))
    np.testing.assert_array_equal(stats["proprio"]["max"], np.full(8, 30.))
    assert stats["proprio"]["num_samples"] == 8
