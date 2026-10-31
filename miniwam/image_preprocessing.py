"""Shared RGB boundary validation for offline and online DINOv3 inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def dino_rgb_uint8_to_bchw(images: np.ndarray | torch.Tensor, *, source: str) -> torch.Tensor:
    """Validate decoded HWC uint8 RGB and change layout only.

    RoboTwin's offline ``decode_image_bit`` and XPolicyLab's policy server both
    expose decoded RGB at this boundary. Color conversion, resizing, rescaling,
    cropping, and normalization intentionally remain the responsibility of the
    frozen DINOv3 image processor.
    """
    if isinstance(images, torch.Tensor):
        array = images.detach().cpu().numpy()
    else:
        array = np.asarray(images)
    if array.dtype != np.uint8:
        raise TypeError(f"{source} must be HWC uint8 RGB, got dtype={array.dtype}.")
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{source} must be [H,W,3] or [B,H,W,3], got {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{source} contains an empty image dimension: {array.shape}.")
    # Copy protects against negative-stride views and mutable simulator buffers.
    return torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).contiguous()


def wan_vae_rgb_uint8_to_bchw(
    images: np.ndarray | torch.Tensor,
    *,
    resolution: tuple[int, int] | list[int],
    source: str,
) -> torch.Tensor:
    """Canonical RoboTwin/LIBERO Wan-VAE RGB preprocessing.

    The input boundary is decoded HWC uint8 RGB. Frames are resized directly
    with bilinear interpolation and mapped to ``[-1, 1]``. There is no channel
    swap, crop, or backend-specific image normalization.
    """
    bchw = dino_rgb_uint8_to_bchw(images, source=source).float().div_(255.0)
    size = tuple(int(value) for value in resolution)
    if len(size) != 2 or any(value <= 0 for value in size):
        raise ValueError(f"Wan-VAE resolution must be [H,W], got {resolution!r}.")
    return F.interpolate(
        bchw, size=size, mode="bilinear", align_corners=False,
    ).mul_(2.0).sub_(1.0).contiguous()
