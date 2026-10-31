import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from miniwam.models.dinov3 import DINO_MINIWAM_FEATURE_CONTRACT
from scripts.eval import gpu_environment
from scripts.eval.gpu_environment import assign_egl_device, parse_gpu_capacity, worker_gpus
from scripts.eval.libero_plus_protocol import (
    CLASSIFICATION_SHA256,
    LIBERO_PLUS_REV,
    LiberoPlusInstructions,
    LiberoPlusTask,
    aggregate_results,
    assert_libero_modules_from,
    load_task_manifest,
    resolve_libero_plus_instructions,
    select_stratified_tasks,
)
from scripts.eval.run_libero_plus_parallel_miniwam import (
    _add_visual_feature_contract,
    _latest_by_task,
    _next_attempt_index,
    _protocol_id,
    _write_reports,
)


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts/eval/manifests/libero_plus_ablation_2k_seed42.txt"


def _record(index, suite="libero_spatial", category="Sensor Noise", difficulty=1):
    return LiberoPlusTask(
        suite=suite,
        task_id=index,
        official_id=index + 1,
        task_name=f"task_{index}",
        category=category,
        difficulty=difficulty,
    )


def test_committed_ablation_manifest_is_exact_and_covers_every_stratum():
    records, metadata = load_task_manifest(MANIFEST)
    assert len(records) == 2000
    assert metadata["seed"] == "42"
    assert len({record.key for record in records}) == 2000
    strata = Counter((r.suite, r.category, r.difficulty) for r in records)
    assert len(strata) == 141
    assert min(strata.values()) >= 1
    assert any(record.difficulty is None for record in records)


def test_stratified_selection_is_order_independent_and_lower_bounded():
    records = []
    for index in range(20):
        records.append(_record(index, category="Sensor Noise", difficulty=1))
    records.append(_record(20, category="Sensor Noise", difficulty=5))
    records.extend(_record(index, category="Camera Viewpoints", difficulty=3) for index in range(21, 30))
    first, quotas = select_stratified_tasks(records, count=10, seed=42)
    second, _ = select_stratified_tasks(reversed(records), count=10, seed=42)
    assert first == second
    assert len(first) == 10
    assert all(quota >= 1 for quota in quotas.values())
    assert any(record.difficulty == 5 for record in first)


def test_import_origin_assertion_rejects_vanilla_checkout(monkeypatch, tmp_path):
    plus_root = tmp_path / "LIBERO-plus"
    plus_module = plus_root / "libero/__init__.py"
    plus_module.parent.mkdir(parents=True)
    plus_module.touch()
    module = types.ModuleType("libero")
    module.__file__ = str(plus_module)
    monkeypatch.setitem(sys.modules, "libero", module)
    assert_libero_modules_from(plus_root)

    vanilla = tmp_path / "LIBERO/libero/__init__.py"
    vanilla.parent.mkdir(parents=True)
    vanilla.touch()
    module.__file__ = str(vanilla)
    with pytest.raises(RuntimeError, match="not intended LIBERO-Plus"):
        assert_libero_modules_from(plus_root)


def test_import_origin_assertion_accepts_intended_namespace_package(monkeypatch, tmp_path):
    plus_root = tmp_path / "LIBERO-plus"
    namespace_root = plus_root / "libero"
    namespace_root.mkdir(parents=True)
    module = types.ModuleType("libero")
    module.__path__ = [str(namespace_root)]
    monkeypatch.setitem(sys.modules, "libero", module)
    assert_libero_modules_from(plus_root)


def test_aggregation_excludes_execution_failures_from_policy_denominator():
    results = [
        {"suite": "libero_spatial", "category": "Sensor Noise", "difficulty": 1,
         "status": "completed", "success": True},
        {"suite": "libero_spatial", "category": "Sensor Noise", "difficulty": None,
         "status": "completed", "success": False},
        {"suite": "libero_goal", "category": "Light Conditions", "difficulty": 2,
         "status": "execution_failed", "success": False},
    ]
    summary = aggregate_results(results)
    assert summary["overall"]["all"] == {
        "episodes": 2, "successes": 1, "success_rate": 0.5,
    }
    assert len(summary["execution_failures"]) == 1
    assert summary["by_difficulty"]["unclassified"]["episodes"] == 1


def test_markdown_report_breaks_results_down_by_suite_category_and_difficulty(tmp_path):
    results = [
        {"suite": "libero_spatial", "category": "Sensor Noise", "difficulty": 1,
         "status": "completed", "success": True},
        {"suite": "libero_goal", "category": "Light Conditions", "difficulty": None,
         "status": "completed", "success": False},
    ]
    report = _write_reports(tmp_path, results, 2, {"standard_protocol": True})
    assert report["complete"] is True
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    for heading in ("## By suite", "## By category", "## By difficulty"):
        assert heading in markdown
    assert "| libero_spatial | 1 | 1 | 100.00% |" in markdown
    assert "| unclassified | 0 | 1 | 0.00% |" in markdown


def test_resume_rejects_results_from_another_protocol(tmp_path):
    checkpoint = tmp_path / "model.pt"
    protocol_id = _protocol_id({"env_seed": 0, "policy_seed": 42})
    result = {
        "suite": "libero_spatial",
        "task_id": 0,
        "checkpoint": str(checkpoint),
        "benchmark_revision": LIBERO_PLUS_REV,
        "classification_sha256": CLASSIFICATION_SHA256,
        "selection_manifest_id": "selection",
        "protocol_id": protocol_id,
    }
    latest = _latest_by_task(
        [result], checkpoint=checkpoint, manifest_id="selection", protocol_id=protocol_id,
    )
    assert latest["libero_spatial:0"] == result
    with pytest.raises(ValueError, match="different evaluation protocol"):
        _latest_by_task(
            [result], checkpoint=checkpoint, manifest_id="selection", protocol_id="changed",
        )


def test_resume_uses_new_attempt_index_without_overwriting(tmp_path):
    assert _next_attempt_index(tmp_path) == 0
    (tmp_path / "gpu0_attempt0.log").touch()
    (tmp_path / "results_gpu1_attempt2.jsonl").touch()
    (tmp_path / "unrelated.txt").touch()
    assert _next_attempt_index(tmp_path) == 3


def test_dino_protocol_identity_is_versioned_while_wan_identity_is_unchanged(tmp_path):
    checkpoint = tmp_path / "latest"
    checkpoint.mkdir()
    config = checkpoint / "config.yaml"

    wan_protocol = {"checkpoint": str(checkpoint), "dtype": "bf16"}
    original_wan_id = _protocol_id(wan_protocol)
    config.write_text("model:\n  visual_encoder:\n    type: wan_vae\n", encoding="utf-8")
    _add_visual_feature_contract(wan_protocol, checkpoint)
    assert "visual_feature_contract" not in wan_protocol
    assert _protocol_id(wan_protocol) == original_wan_id

    dino_protocol = {"checkpoint": str(checkpoint), "dtype": "bf16"}
    original_dino_id = _protocol_id(dino_protocol)
    config.write_text("model:\n  visual_encoder:\n    type: dinov3\n", encoding="utf-8")
    _add_visual_feature_contract(dino_protocol, checkpoint)
    assert dino_protocol["visual_feature_contract"] == DINO_MINIWAM_FEATURE_CONTRACT
    assert _protocol_id(dino_protocol) != original_dino_id


def test_assign_egl_device_overrides_stale_renderer_selection(monkeypatch):
    monkeypatch.setattr(gpu_environment, "egl_device_for_cuda", lambda gpu_id: None)
    environment = {"MUJOCO_EGL_DEVICE_ID": "0", "EGL_DEVICE_ID": "0", "KEEP": "yes"}
    assign_egl_device(environment, 3)
    assert environment == {
        "MUJOCO_EGL_DEVICE_ID": "3",
        "EGL_DEVICE_ID": "3",
        "KEEP": "yes",
    }


def test_assign_egl_device_renders_on_the_cuda_device(monkeypatch):
    # EGL lists these two GPUs in the reverse of CUDA's order.
    monkeypatch.setattr(gpu_environment, "egl_device_for_cuda", lambda gpu_id: 1 - gpu_id)
    environment = {}
    assign_egl_device(environment, 0)
    assert environment == {"MUJOCO_EGL_DEVICE_ID": "1", "EGL_DEVICE_ID": "1"}


def test_egl_lookup_without_libegl_falls_back(monkeypatch):
    def missing_library(name):
        raise OSError(name)

    monkeypatch.setattr(gpu_environment.ctypes, "CDLL", missing_library)
    assert gpu_environment.egl_device_for_cuda(0) is None


def test_clean_instruction_uses_vanilla_style_but_keeps_raw_suffix():
    record = _record(
        0,
        category="Camera Viewpoints",
        difficulty=2,
    )
    base_name = "task_0"
    record = LiberoPlusTask(
        suite=record.suite,
        task_id=record.task_id,
        official_id=record.official_id,
        task_name=base_name + "_view_0_0_100_0_0_initstate_384",
        category=record.category,
        difficulty=record.difficulty,
    )
    task = types.SimpleNamespace(
        name=record.task_name,
        language="task 0 view 0 0 100 0 0 initstate 384",
    )
    instructions = resolve_libero_plus_instructions(
        record,
        task,
        {"libero_spatial": {base_name: "task zero"}},
    )
    assert instructions == LiberoPlusInstructions(
        raw=task.language,
        clean="task zero",
        base="task zero",
        source="vanilla_libero_task_language",
    )


def test_language_perturbation_remains_model_facing():
    record = LiberoPlusTask(
        suite="libero_spatial",
        task_id=0,
        official_id=1,
        task_name="task_0_language_7",
        category="Language Instructions",
        difficulty=3,
    )
    task = types.SimpleNamespace(name=record.task_name, language="deliberate paraphrase")
    instructions = resolve_libero_plus_instructions(
        record,
        task,
        {"libero_spatial": {"task_0": "task zero"}},
    )
    assert instructions.clean == "deliberate paraphrase"
    assert instructions.base == "task zero"
    assert instructions.source == "libero_plus_language_perturbation"


def test_workers_alternate_between_gpus_up_to_each_gpu_count():
    flags = ("--gpu_ids", "--max_tasks_per_gpu")
    assert worker_gpus(parse_gpu_capacity("0", "1", *flags)) == [0]
    assert worker_gpus(parse_gpu_capacity("0,1", "2", *flags)) == [0, 1, 0, 1]
    assert worker_gpus(parse_gpu_capacity("1,0", "3,1", *flags)) == [1, 0, 1, 1]
    with pytest.raises(ValueError, match="must not repeat"):
        parse_gpu_capacity("0,0,1", "1", *flags)
