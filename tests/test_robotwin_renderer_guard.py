import subprocess
import sys
from pathlib import Path


WATCHDOG = Path(
    "integrations/xpolicylab/policy/MiniWAM/compat/run_with_inactivity_timeout.py"
)


def test_renderer_error_is_fatal_even_when_child_exits_successfully():
    result = subprocess.run(
        [
            sys.executable,
            str(WATCHDOG),
            "--timeout-seconds",
            "10",
            "--",
            sys.executable,
            "-c",
            "print('[svulkan2] [error] OIDN Error: unsupported device type: CUDA')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == 125
    assert "Fatal renderer output detected" in result.stdout


def test_healthy_child_output_is_not_reclassified():
    result = subprocess.run(
        [sys.executable, str(WATCHDOG), "--timeout-seconds", "10", "--",
         sys.executable, "-c", "print('renderer ok')"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
