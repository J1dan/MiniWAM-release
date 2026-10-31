"""Catch broken package initializers on the supported deployment import path."""
import subprocess
import os
import pytest
import sys
from pathlib import Path


def test_robotwin_runtime_imports_in_a_fresh_process():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, '-c',
         'from deploy.robotwin_runtime import MiniWAMRoboTwinRuntime; '
         'assert callable(MiniWAMRoboTwinRuntime)'],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", [
    "scripts/data/build_libero_text_feature_cache.py",
])
def test_data_cli_resolves_checkout_without_pythonpath(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(root / script), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
