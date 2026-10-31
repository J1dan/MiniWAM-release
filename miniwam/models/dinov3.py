"""Frozen Hugging Face DINOv3 framewise patch-token extraction."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_COMPUTE_DTYPE = torch.float32
DINO_MINIWAM_DTYPE = torch.bfloat16
DINO_MINIWAM_FEATURE_CONTRACT = "dinov3_fp32_compute_bf16_boundary_v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


class DINOv3FrameEncoder(nn.Module):
    """Return final LayerNorm-normalized patch tokens without L2 normalization."""

    def __init__(
        self,
        architecture: str = "vitb16",
        pretraining: str = "lvd1689m",
        model_id: str = DEFAULT_DINOV3_MODEL_ID,
        revision: Optional[str] = None,
        input_size: int = 224,
        frozen: bool = True,
        token: Optional[str] = None,
        backbone: Optional[nn.Module] = None,
        image_processor: Optional[Any] = None,
    ):
        super().__init__()
        architecture = str(architecture).lower()
        pretraining = str(pretraining).lower()
        if architecture != "vitb16":
            raise ValueError(f"Unsupported DINOv3 architecture {architecture!r}; expected 'vitb16'.")
        if pretraining != "lvd1689m":
            raise ValueError("The initial DINO ablation supports only lvd1689m pretraining.")
        if int(input_size) != 224:
            raise ValueError("The initial DINOv3 comparison requires input_size=224.")

        self.architecture = architecture
        self.pretraining = pretraining
        self.model_id = str(model_id)
        self.requested_revision = str(revision) if revision is not None else None
        self.input_size = int(input_size)
        self.patch_size = 16
        self.feature_dim = 768
        self.frozen = bool(frozen)

        if backbone is None:
            from transformers import AutoImageProcessor, AutoModel

            image_processor = AutoImageProcessor.from_pretrained(
                self.model_id,
                revision=self.requested_revision,
                token=token,
            )
            backbone = AutoModel.from_pretrained(
                self.model_id,
                revision=self.requested_revision,
                token=token,
            )
        self.backbone = backbone
        self.image_processor = image_processor

        config = getattr(self.backbone, "config", None)
        if config is not None:
            patch_size = getattr(config, "patch_size", self.patch_size)
            if isinstance(patch_size, (list, tuple)):
                if len(set(int(x) for x in patch_size)) != 1:
                    raise ValueError(f"DINO patch size must be square, got {patch_size!r}.")
                patch_size = patch_size[0]
            hidden_size = int(getattr(config, "hidden_size", self.feature_dim))
            if int(patch_size) != self.patch_size or hidden_size != self.feature_dim:
                raise ValueError(
                    "Expected DINOv3 ViT-B/16 with patch_size=16 and hidden_size=768, got "
                    f"patch_size={patch_size}, hidden_size={hidden_size}."
                )
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 4))
        resolved_revision = getattr(config, "_commit_hash", None)
        self.resolved_revision = str(resolved_revision or self.requested_revision or "injected")

        if self.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.register_buffer(
            "fallback_pixel_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "fallback_pixel_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    @property
    def processor_settings(self) -> dict[str, Any]:
        if self.image_processor is None:
            return {
                "implementation": "test_fallback",
                "size": [self.input_size, self.input_size],
                "image_mean": self.fallback_pixel_mean.flatten().tolist(),
                "image_std": self.fallback_pixel_std.flatten().tolist(),
                "do_rescale": False,
            }
        fields = (
            "do_resize", "size", "resample", "do_rescale", "rescale_factor",
            "do_normalize", "image_mean", "image_std", "do_center_crop", "crop_size",
        )
        return {
            "implementation": type(self.image_processor).__name__,
            **{
                field: _jsonable(getattr(self.image_processor, field, None))
                for field in fields
            },
        }

    def feature_metadata(self, storage_dtype: str = "bfloat16") -> dict[str, Any]:
        storage_dtype = str(storage_dtype).removeprefix("torch.")
        if storage_dtype != "bfloat16":
            raise ValueError(
                "DINOv3 cache storage must match the BF16 MiniWAM feature boundary."
            )
        return {
            "architecture": self.architecture,
            "pretraining": self.pretraining,
            "model_id": self.model_id,
            "revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "feature_type": "patch_tokens",
            "feature_normalization": "final_layernorm",
            "l2_normalize": False,
            "include_cls": False,
            "include_registers": False,
            "feature_layers": "final",
            "compute_dtype": "float32",
            "boundary_dtype": "bfloat16",
            "cache_layout": "per_frame_v1",
            "storage_dtype": storage_dtype,
            "processor": self.processor_settings,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"DINO RGB input must be [B,3,H,W], got {tuple(rgb.shape)}.")
        x = rgb.detach().float()
        # uint8 is always 0..255, even a dark frame whose max is 1. Float inputs
        # may be 0..255, 0..1 or -1..1.
        if rgb.dtype == torch.uint8 or x.max().item() > 1.0:
            x = x / 255.0
        if x.min().item() < 0.0:
            x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)

        if self.image_processor is not None:
            processed = self.image_processor(
                images=[frame.cpu() for frame in x],
                return_tensors="pt",
                do_rescale=False,
            )
            pixel_values = processed["pixel_values"]
        else:
            pixel_values = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )
            pixel_values = (
                pixel_values.to(self.fallback_pixel_mean.device)
                - self.fallback_pixel_mean
            ) / self.fallback_pixel_std

        parameter = next(self.backbone.parameters(), None)
        device = parameter.device if parameter is not None else self.fallback_pixel_mean.device
        dtype = parameter.dtype if parameter is not None else self.fallback_pixel_mean.dtype
        return pixel_values.to(device=device, dtype=dtype)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        floating_dtypes = {
            parameter.dtype
            for parameter in self.backbone.parameters()
            if parameter.is_floating_point()
        }
        if floating_dtypes and floating_dtypes != {DINO_COMPUTE_DTYPE}:
            raise RuntimeError(
                "DINOv3 must run entirely in FP32 before the MiniWAM BF16 feature boundary; "
                f"found backbone parameter dtypes {sorted(map(str, floating_dtypes))}."
            )

        # DINO is an explicit FP32 island. This nested disabled context prevents
        # a caller's MiniWAM BF16 autocast scope from changing backbone numerics.
        device_type = next(self.backbone.parameters(), self.fallback_pixel_mean).device.type
        with torch.autocast(device_type=device_type, enabled=False):
            pixel_values = self.preprocess(rgb)
            if pixel_values.dtype != DINO_COMPUTE_DTYPE:
                raise RuntimeError(
                    f"DINOv3 preprocessed pixels must be FP32, got {pixel_values.dtype}."
                )
            with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
                outputs = self.backbone(pixel_values=pixel_values)
        if isinstance(outputs, dict):
            last_hidden_state = outputs.get("last_hidden_state")
        else:
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise RuntimeError("Hugging Face DINOv3 did not return outputs.last_hidden_state.")
        if last_hidden_state.dtype != DINO_COMPUTE_DTYPE:
            raise RuntimeError(
                "DINOv3 must produce FP32 features before the MiniWAM BF16 boundary, "
                f"got {last_hidden_state.dtype}."
            )

        patch_tokens = last_hidden_state[:, 1 + self.num_register_tokens:]
        expected_patches = (self.input_size // self.patch_size) ** 2
        if patch_tokens.ndim != 3 or patch_tokens.shape[1:] != (
            expected_patches,
            self.feature_dim,
        ):
            raise ValueError(
                f"Expected final DINO patch tokens [B,{expected_patches},{self.feature_dim}], "
                f"got {tuple(patch_tokens.shape)}."
            )
        return patch_tokens

    def encode_for_miniwam(self, rgb: torch.Tensor) -> torch.Tensor:
        """Run DINO in FP32 and apply the single BF16 MiniWAM feature boundary."""
        return self(rgb).to(dtype=DINO_MINIWAM_DTYPE)
