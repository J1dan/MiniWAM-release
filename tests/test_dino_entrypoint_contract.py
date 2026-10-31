import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_online_dino_entrypoints_use_fp32_compute_and_bf16_boundary():
    plus = _source("scripts/eval/miniwam_libero_plus_runtime.py")

    assert ").to(device=self.device, dtype=torch.float32).eval()" in plus


def test_all_dino_extractors_use_current_helper_signature_and_metadata():
    paths = (
        "scripts/data/extract_dataset_feature_libero.py",
    )
    precision_messages = (
        "Extraction pipeline dtype:",
        "DINO compute dtype:",
        "DINO feature boundary:",
        "DINO cache storage dtype:",
    )
    for relative in paths:
        source = _source(relative)
        tree = ast.parse(source, filename=relative)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "encode_dinov3_frames"
        ]
        assert calls, relative
        assert all(len(call.args) == 4 for call in calls), relative
        assert 'feature_metadata(storage_dtype="bfloat16")' in source
        assert all(message in source for message in precision_messages)


def test_direct_and_parallel_simulator_entrypoints_assign_egl_devices():
    for relative in (
        "scripts/eval/eval_libero_miniwam.py",
        "scripts/eval/run_libero_parallel_miniwam.py",
        "scripts/eval/eval_libero_plus_miniwam.py",
        "scripts/eval/run_libero_plus_parallel_miniwam.py",
    ):
        assert "assign_egl_device(" in _source(relative), relative
