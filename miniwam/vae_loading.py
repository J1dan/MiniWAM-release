"""Shared, strict loading for the frozen Wan visual encoder."""

from __future__ import annotations

from pathlib import Path

import torch

from miniwam.models.wan_video_vae import WanVideoVAE


def load_wan_video_vae(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> WanVideoVAE:
    """Load the exact Wan VAE used by offline extraction and online inference."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Wan VAE checkpoint does not exist: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        raw_state = load_file(str(path), device="cpu")
    else:
        raw_state = torch.load(path, map_location="cpu", weights_only=True)
    model = WanVideoVAE(z_dim=16)
    converted = WanVideoVAE.state_dict_converter().from_civitai(raw_state)
    model.load_state_dict(converted, strict=True)
    return model.to(device=device, dtype=dtype).eval()
