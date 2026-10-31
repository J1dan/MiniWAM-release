"""Every import of a project module must point at a file in this repository."""
import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("miniwam", "scripts", "deploy")


def _tracked_python_files():
    # Vendored checkouts (LIBERO/, RoboTwin/, ...) are gitignored and not part of
    # this repository; `git ls-files` walks only what's actually tracked, so a
    # cloned third-party file can never be mistaken for project source here.
    output = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return sorted(ROOT / line for line in output.splitlines())


def _project_imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_project_imports_resolve():
    missing = []
    for path in _tracked_python_files():
        for module in _project_imports(path):
            parts = module.split(".")
            if parts[0] not in PACKAGES:
                continue
            target = ROOT.joinpath(*parts)
            if not (target.with_suffix(".py").is_file() or target.is_dir()):
                missing.append(f"{path.relative_to(ROOT)}: {module}")
    assert not missing, missing
