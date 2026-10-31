import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


EVAL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from libero_eval_recovery import (  # noqa: E402
    load_recoverable_trials,
    validate_complete_summary,
    write_trial_progress,
)
from gpu_environment import parse_gpu_capacity  # noqa: E402


def _video(path: Path, trial_idx: int, success: bool, timestamp_ns: int) -> Path:
    video_dir = path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video = video_dir / (
        f"2026_08_23-12_00_00--episode={trial_idx}--success={success}--task=test.mp4"
    )
    if video.exists():
        video = video.with_name(f"retry--episode={trial_idx}--success={success}--task=test.mp4")
    video.write_bytes(b"mp4")
    os.utime(video, ns=(timestamp_ns, timestamp_ns))
    return video


def test_videos_are_deduplicated_and_progress_fills_missing_trials(tmp_path):
    checkpoint = tmp_path / "step_000001" / "model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    task_dir = tmp_path / "results" / "libero_10" / "task_6"

    _video(task_dir, 0, False, 10)
    latest = _video(task_dir, 0, True, 20)
    _video(task_dir, 2, False, 10)

    records, metadata = load_recoverable_trials(
        task_dir,
        task_suite_name="libero_10",
        task_id=6,
        checkpoint=checkpoint,
        num_trials=3,
        base_seed=42,
    )
    assert records[0]["success"] is True
    assert records[0]["video"] == str(latest.resolve())
    assert records[2]["success"] is False
    assert metadata["duplicate_trial_ids"] == [0]

    records[1] = {"trial_idx": 1, "success": True, "seed": 43, "source": "evaluation"}
    write_trial_progress(
        task_dir,
        records,
        task_suite_name="libero_10",
        task_id=6,
        checkpoint=checkpoint,
        num_trials=3,
        base_seed=42,
    )
    recovered, _ = load_recoverable_trials(
        task_dir,
        task_suite_name="libero_10",
        task_id=6,
        checkpoint=checkpoint,
        num_trials=3,
        base_seed=42,
    )
    assert sorted(recovered) == [0, 1, 2]
    assert recovered[1]["source"] == "evaluation"


def test_complete_summary_requires_exact_trials_and_consistent_metrics(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    summary_path = tmp_path / "summary.json"
    summary = {
        "task_suite_name": "libero_goal",
        "task_id": 2,
        "checkpoint": str(checkpoint.resolve()),
        "num_trials": 3,
        "success_count": 2,
        "success_rate": 100.0 * 2 / 3,
        "trials": [
            {"trial_idx": 0, "success": True, "seed": 42},
            {"trial_idx": 1, "success": False, "seed": 43},
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    valid, reason = validate_complete_summary(
        summary_path,
        task_suite_name="libero_goal",
        task_id=2,
        checkpoint=checkpoint,
        num_trials=3,
        base_seed=42,
    )
    assert valid is None
    assert "2/3" in reason

    summary["trials"].append({"trial_idx": 2, "success": True, "seed": 44})
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    valid, reason = validate_complete_summary(
        summary_path,
        task_suite_name="libero_goal",
        task_id=2,
        checkpoint=checkpoint,
        num_trials=3,
        base_seed=42,
    )
    assert reason is None
    assert valid["success_count"] == 2


def test_summary_rejects_wrong_seed_or_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    other_checkpoint = tmp_path / "other.pt"
    checkpoint.write_bytes(b"checkpoint")
    other_checkpoint.write_bytes(b"other")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({
        "task_suite_name": "libero_spatial",
        "task_id": 0,
        "checkpoint": str(other_checkpoint.resolve()),
        "num_trials": 1,
        "success_count": 1,
        "success_rate": 100.0,
        "trials": [{"trial_idx": 0, "success": True, "seed": 99}],
    }), encoding="utf-8")
    valid, reason = validate_complete_summary(
        summary_path,
        task_suite_name="libero_spatial",
        task_id=0,
        checkpoint=checkpoint,
        num_trials=1,
        base_seed=42,
    )
    assert valid is None
    assert "identity" in reason


def test_progress_from_other_protocol_is_not_resumed(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    task_dir = tmp_path / "results" / "libero_10" / "task_0"
    other = {"protocol": "other", "max_steps": 600, "num_steps_wait": 20}
    common = {"protocol": "common_libero_v1", "max_steps": 520, "num_steps_wait": 10}
    write_trial_progress(
        task_dir,
        {0: {"trial_idx": 0, "success": True, "seed": 42, "source": "evaluation"}},
        task_suite_name="libero_10", task_id=0, checkpoint=checkpoint,
        num_trials=2, base_seed=42, evaluation_protocol=other,
    )
    _video(task_dir, 1, True, 10)

    records, metadata = load_recoverable_trials(
        task_dir, task_suite_name="libero_10", task_id=0, checkpoint=checkpoint,
        num_trials=2, base_seed=42, required_protocol=common,
    )
    assert records == {}
    assert metadata["ignored_incompatible_progress"] is True
    assert metadata["skipped_video_trials"] == 1

    records, metadata = load_recoverable_trials(
        task_dir, task_suite_name="libero_10", task_id=0, checkpoint=checkpoint,
        num_trials=2, base_seed=42, required_protocol=other,
    )
    assert sorted(records) == [0, 1]
    assert metadata["skipped_video_trials"] == 0

    records, _ = load_recoverable_trials(
        task_dir, task_suite_name="libero_10", task_id=0, checkpoint=checkpoint,
        num_trials=2, base_seed=42,
    )
    assert sorted(records) == [0, 1]


def test_summary_from_other_protocol_is_rejected(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    summary_path = tmp_path / "summary.json"
    summary = {
        "task_suite_name": "libero_spatial", "task_id": 0,
        "checkpoint": str(checkpoint.resolve()), "num_trials": 1,
        "success_count": 1, "success_rate": 100.0,
        "evaluation_protocol": {"protocol": "other", "max_steps": 600},
        "trials": [{"trial_idx": 0, "success": True, "seed": 42}],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    common = {"protocol": "common_libero_v1", "max_steps": 220}
    valid, reason = validate_complete_summary(
        summary_path, task_suite_name="libero_spatial", task_id=0,
        checkpoint=checkpoint, num_trials=1, base_seed=42, required_protocol=common,
    )
    assert valid is None and "protocol" in reason
    summary["evaluation_protocol"] = {**common, "dtype": "bf16"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    valid, reason = validate_complete_summary(
        summary_path, task_suite_name="libero_spatial", task_id=0,
        checkpoint=checkpoint, num_trials=1, base_seed=42, required_protocol=common,
    )
    assert reason is None and valid["success_count"] == 1


def test_superseded_results_are_quarantined_so_a_later_resume_cannot_import_them(tmp_path):
    from libero_eval_recovery import quarantine_superseded_results

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    task_dir = tmp_path / "results" / "libero_10" / "task_0"
    identity = dict(task_suite_name="libero_10", task_id=0, checkpoint=checkpoint,
                    num_trials=3, base_seed=42)
    other = {"protocol": "other", "max_steps": 600, "num_steps_wait": 20}
    common = {"protocol": "common_libero_v1", "max_steps": 520, "num_steps_wait": 10}
    write_trial_progress(
        task_dir, {0: {"trial_idx": 0, "success": True, "seed": 42, "source": "evaluation"}},
        **identity, evaluation_protocol=other,
    )
    for trial_idx in range(3):
        _video(task_dir, trial_idx, True, 10 + trial_idx)
    (task_dir / "summary.json").write_text("{}", encoding="utf-8")

    # First rerun under the new protocol: nothing is resumed and the old results move aside.
    records, metadata = load_recoverable_trials(task_dir, **identity, required_protocol=common)
    assert records == {} and metadata["skipped_video_trials"] == 3
    superseded = quarantine_superseded_results(task_dir)
    assert superseded is not None and superseded.parent == task_dir
    assert sorted(path.name for path in superseded.iterdir()) == [
        "summary.json", "trial_progress.json", "videos",
    ]
    # An empty videos/ stays in place: the evaluator writes rollouts into it
    # without creating it again.
    assert (task_dir / "videos").is_dir() and not any((task_dir / "videos").iterdir())

    # The rerun records one new trial and is interrupted; the next rerun must not
    # import the superseded videos for the remaining trials.
    (task_dir / "videos" / "2026_10_15-12_00_00--episode=0--success=False--task=test.mp4").write_bytes(b"mp4")
    write_trial_progress(
        task_dir, {0: {"trial_idx": 0, "success": False, "seed": 42, "source": "evaluation"}},
        **identity, evaluation_protocol=common,
    )
    records, metadata = load_recoverable_trials(task_dir, **identity, required_protocol=common)
    assert {index: record["source"] for index, record in records.items()} == {0: "evaluation"}
    assert metadata["imported_video_trials"] == 0
    assert quarantine_superseded_results(tmp_path / "empty") is None


def test_task_caps_take_one_value_or_one_per_gpu():
    flags = ("--gpu_ids", "--max_tasks_per_gpu")
    assert parse_gpu_capacity("0,1", "2", *flags) == {0: 2, 1: 2}
    assert parse_gpu_capacity("0, 1", "9, 4", *flags) == {0: 9, 1: 4}
    with pytest.raises(ValueError, match="one per entry of --gpu_ids"):
        parse_gpu_capacity("0,1", "1,2,3", *flags)
    with pytest.raises(ValueError, match="--max_tasks_per_gpu must be at least 1"):
        parse_gpu_capacity("0,1", "2,0", *flags)
    with pytest.raises(ValueError, match="must not repeat a GPU; raise --max_tasks_per_gpu"):
        parse_gpu_capacity("0,0", "2", *flags)
    with pytest.raises(ValueError, match="no GPUs"):
        parse_gpu_capacity(" , ", "1", *flags)
    with pytest.raises(ValueError, match="GPU indices"):
        parse_gpu_capacity("-1", "1", *flags)
    with pytest.raises(ValueError, match="GPU indices"):
        parse_gpu_capacity("0,a", "1", *flags)
    with pytest.raises(ValueError, match="one whole number, or one per entry of --gpu_ids"):
        parse_gpu_capacity("0", "1.5", *flags)


def test_parallel_launcher_refuses_an_unsupported_suite_before_any_work(tmp_path, monkeypatch):
    import run_libero_parallel_miniwam as launcher

    def launch(*args, **kwargs):
        raise AssertionError("an unsupported suite must be refused before any task starts")

    monkeypatch.setattr(launcher, "subprocess", SimpleNamespace(Popen=launch))
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_config"))
    monkeypatch.setattr(sys, "argv", [
        "run_libero_parallel_miniwam.py", "--checkpoint", str(tmp_path),
        "--task_suite_names", "libero_spatial,libero_spatail",
        "--output_dir", str(tmp_path / "results"),
    ])
    with pytest.raises(ValueError, match="libero_spatail"):
        launcher.main()
    assert not (tmp_path / "results").exists()
