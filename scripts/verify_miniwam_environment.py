#!/usr/bin/env python3
"""Verify the runtime contracts required by MiniWAM."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSIONS = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "numpy": "1.26.4",
    "transformers": "4.57.6",
    "tokenizers": "0.22.1",
}
EXPECTED_DISTRIBUTIONS = {
    "protobuf": "4.25.3",
}
LIBERO_REV = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_PLUS_REV = "4976dc30028e805ff8094b55501d532c48fec182"
LIBERO_PLUS_CLASSIFICATION_SHA256 = "faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6"
REQUIRED_ASSETS = {
    ROOT / "pretrained/vision/Wan2.1_VAE.pth": 500_000_000,
    ROOT / "pretrained/language/models_t5_umt5-xxl-enc-bf16.pth": 11_000_000_000,
}


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-libero", action="store_true")
    parser.add_argument("--check-libero-plus", action="store_true")
    parser.add_argument("--check-assets", action="store_true")
    parser.add_argument("--visual-encoder", choices=["dinov3", "wan_vae"], default="dinov3")
    args = parser.parse_args()

    errors: list[str] = []
    if sys.version_info[:2] != (3, 10):
        errors.append(f"Python 3.10 is required, found {sys.version.split()[0]}")

    loaded = {}
    for module_name, expected in EXPECTED_VERSIONS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # Import failures need their original message.
            errors.append(f"Could not import {module_name}: {exc}")
            continue
        loaded[module_name] = module
        actual = public_version(module.__version__)
        if actual != expected:
            errors.append(f"{module_name}=={expected} is required, found {module.__version__}")

    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing required distribution: {distribution}=={expected}")
            continue
        if public_version(actual) != expected:
            errors.append(f"{distribution}=={expected} is required, found {actual}")

    torch = loaded.get("torch")
    if torch is not None:
        if torch.version.cuda != "12.8":
            errors.append(f"PyTorch CUDA runtime 12.8 is required, found {torch.version.cuda}")
        if args.require_cuda and not torch.cuda.is_available():
            errors.append(
                "CUDA is not available. Check the NVIDIA driver and whether this shell has GPU access."
            )
        try:
            import torch.utils.checkpoint  # noqa: F401
        except Exception as exc:
            errors.append(f"Could not import torch.utils.checkpoint: {exc}")

    try:
        import miniwam  # noqa: F401
        from miniwam.models.scheduler import FlowMatchScheduler

        FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    except Exception as exc:
        errors.append(f"Could not import MiniWAM: {exc}")

    if args.check_libero:
        config_path = ROOT / ".libero/config.yaml"
        libero_checkout = ROOT / "LIBERO"
        if not libero_checkout.is_dir():
            errors.append(f"Missing pinned LIBERO checkout: {libero_checkout}")
        elif (libero_checkout / ".git").is_dir():
            try:
                actual_rev = subprocess.check_output(
                    ["git", "-C", str(libero_checkout), "rev-parse", "HEAD"],
                    text=True,
                ).strip()
                if actual_rev != LIBERO_REV:
                    errors.append(f"LIBERO must be at {LIBERO_REV}, found {actual_rev}")
            except (OSError, subprocess.CalledProcessError) as exc:
                errors.append(f"Could not verify LIBERO revision: {exc}")
        if not config_path.is_file():
            errors.append(f"Missing non-interactive LIBERO config: {config_path}")
        else:
            os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_path.parent))
            try:
                sys.path.insert(0, str(libero_checkout))
                import libero  # noqa: F401
                from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
            except Exception as exc:
                errors.append(f"Could not import pinned LIBERO environment: {exc}")

    if args.check_libero_plus:
        plus_checkout = ROOT / "LIBERO-plus"
        plus_config = ROOT / ".libero_plus/config.yaml"
        for distribution, expected in {
            "Wand": "0.6.13",
            "scikit-image": "0.24.0",
            "usd-core": "25.5.1",
        }.items():
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Missing LIBERO-Plus distribution: {distribution}=={expected}")
            else:
                if public_version(actual) != expected:
                    errors.append(f"{distribution}=={expected} is required, found {actual}")
        if not plus_checkout.is_dir():
            errors.append(f"Missing pinned LIBERO-Plus checkout: {plus_checkout}")
        elif not (plus_checkout / ".git").is_dir():
            errors.append(f"LIBERO-Plus is not a Git checkout: {plus_checkout}")
        else:
            try:
                actual_rev = subprocess.check_output(
                    ["git", "-C", str(plus_checkout), "rev-parse", "HEAD"], text=True,
                ).strip()
                if actual_rev != LIBERO_PLUS_REV:
                    errors.append(f"LIBERO-Plus must be at {LIBERO_PLUS_REV}, found {actual_rev}")
            except (OSError, subprocess.CalledProcessError) as exc:
                errors.append(f"Could not verify LIBERO-Plus revision: {exc}")
        classification = plus_checkout / "libero/libero/benchmark/task_classification.json"
        if classification.is_file():
            import hashlib

            actual_sha = hashlib.sha256(classification.read_bytes()).hexdigest()
            if actual_sha != LIBERO_PLUS_CLASSIFICATION_SHA256:
                errors.append(
                    "LIBERO-Plus task_classification.json checksum mismatch: "
                    f"{actual_sha}"
                )
        else:
            errors.append(f"Missing LIBERO-Plus classification registry: {classification}")
        for relative in (
            "libero/libero/assets/textures",
            "libero/libero/assets/new_objects",
            "libero/libero/assets/scenes",
        ):
            if not (plus_checkout / relative).is_dir():
                errors.append(f"Missing LIBERO-Plus asset directory: {plus_checkout / relative}")
        if not plus_config.is_file():
            errors.append(f"Missing non-interactive LIBERO-Plus config: {plus_config}")
        if plus_checkout.is_dir() and plus_config.is_file():
            code = """
from pathlib import Path
from wand.image import Image
from scripts.eval.libero_plus_protocol import configure_libero_plus, instantiate_suite
root = Path(__import__('os').environ['MINIWAM_PLUS_VERIFY_ROOT'])
config = Path(__import__('os').environ['MINIWAM_PLUS_VERIFY_CONFIG'])
png = root / 'libero/libero/assets/articulated_objects/short_cabinet/base/dark-wood.png'
with Image(blob=png.read_bytes()) as image:
    assert image.format == 'PNG' and image.width > 0 and image.height > 0
benchmark = configure_libero_plus(root, config_dir=config, require_assets=True)
expected = {'libero_spatial': 2402, 'libero_object': 2518, 'libero_goal': 2591, 'libero_10': 2519}
for name, size in expected.items():
    assert instantiate_suite(benchmark, name).n_tasks == size
"""
            verify_env = os.environ.copy()
            verify_env["PYTHONPATH"] = str(ROOT)
            verify_env["MINIWAM_PLUS_VERIFY_ROOT"] = str(plus_checkout)
            verify_env["MINIWAM_PLUS_VERIFY_CONFIG"] = str(plus_config.parent)
            try:
                subprocess.run([sys.executable, "-c", code], env=verify_env, check=True)
            except subprocess.CalledProcessError as exc:
                errors.append(f"Could not import the intended LIBERO-Plus checkout: {exc}")

    if args.check_assets:
        if args.visual_encoder == "dinov3":
            try:
                from huggingface_hub import snapshot_download
                from miniwam.models.dinov3 import DINOv3FrameEncoder  # noqa: F401 (import check)
                snapshot = Path(snapshot_download(
                    "facebook/dinov3-vitb16-pretrain-lvd1689m",
                    revision="5931719e67bbdb9737e363e781fb0c67687896bc",
                    local_files_only=True,
                ))
                if not (snapshot / "config.json").is_file() or not list(snapshot.glob("*.safetensors")):
                    errors.append(f"Incomplete DINOv3 snapshot: {snapshot}")
            except Exception as exc:
                errors.append(f"DINOv3 asset unavailable; see docs/install.md: {exc}")
        for path, minimum_size in REQUIRED_ASSETS.items():
            if args.visual_encoder == "dinov3" and path.name == "Wan2.1_VAE.pth":
                continue
            if not path.is_file():
                errors.append(f"Missing model asset: {path}")
            elif path.stat().st_size < minimum_size:
                errors.append(
                    f"Model asset appears incomplete: {path} ({path.stat().st_size} bytes)"
                )

    if errors:
        print("MiniWAM environment verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    cuda_status = "unavailable"
    if torch is not None and torch.cuda.is_available():
        cuda_status = torch.cuda.get_device_name(0)
    print(f"MiniWAM environment verified (Python {sys.version.split()[0]}, CUDA: {cuda_status}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
