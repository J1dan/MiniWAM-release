"""
MiniWAM: compact world-action model.

Self-attention runs over [z0_tokens, L_tokens, a_tokens]; text and proprioception
enter through cross-attention.

    z0_tokens: current-observation features (always clean, timestep=0)
    L_tokens:  PRISM world tokens from E_phi (Stage 1: clean / Stage 2: noised)
    a_tokens:  action chunk (always noised)

Stage 1 trains E_phi with inverse dynamics (and optional feature
reconstruction); Stage 2 freezes E_phi and jointly denoises L and actions.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.action_dit import ActionDiT
from .models.context_encoder import ContextEncoder
from .models.world_dit import WorldDiT
from .models.latent_encoder import (
    DINOv3LatentEncoder3D,
    FutureFeatureReconstructionDecoder,
    LatentEncoder3D,
)
from .models.mot import MoT
from .models.scheduler import FlowMatchScheduler
from .protocol import MiniWAMRuntimeProtocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FlattenedRoleSlotFeatures:
    batch_size: int
    num_slots: int
    current: torch.Tensor
    future: torch.Tensor
    concatenated: torch.Tensor


def _flatten_role_slot_features(
    z_t_slots: torch.Tensor,
    z_fwd_slots: torch.Tensor,
    present: torch.Tensor,
    visual_encoder_type: str,
) -> _FlattenedRoleSlotFeatures:
    """Validate, flatten, and concatenate native current/future slot features."""
    expected_dims = (6, 6) if visual_encoder_type == "wan_vae" else (4, 5)
    if (z_t_slots.ndim, z_fwd_slots.ndim) != expected_dims:
        expected = (
            "[B,V,C,1,H,W] and [B,V,C,T,H,W]"
            if visual_encoder_type == "wan_vae"
            else "[B,V,N,D] and [B,V,T,N,D]"
        )
        raise ValueError(
            f"Expected {visual_encoder_type} slot tensors {expected}, got "
            f"{tuple(z_t_slots.shape)} and {tuple(z_fwd_slots.shape)}"
        )
    batch_size, num_slots = z_t_slots.shape[:2]
    if z_fwd_slots.shape[:2] != (batch_size, num_slots):
        raise ValueError(
            "Current and future role-slot tensors must have matching [B,V] dimensions, "
            f"got {tuple(z_t_slots.shape[:2])} and {tuple(z_fwd_slots.shape[:2])}."
        )
    if present.shape != (batch_size, num_slots):
        raise ValueError(
            f"`present` must be [B,V]={batch_size,num_slots}, got {tuple(present.shape)}"
        )
    current = z_t_slots.reshape(batch_size * num_slots, *z_t_slots.shape[2:])
    future = z_fwd_slots.reshape(batch_size * num_slots, *z_fwd_slots.shape[2:])
    concatenated = (
        torch.cat([current, future], dim=2)
        if visual_encoder_type == "wan_vae"
        else torch.cat([current.unsqueeze(1), future], dim=1)
    )
    return _FlattenedRoleSlotFeatures(
        batch_size=batch_size,
        num_slots=num_slots,
        current=current,
        future=future,
        concatenated=concatenated,
    )


@dataclass(frozen=True)
class LatentEncoderPair:
    """The main and wrist PRISM encoders built from one model config."""

    latent_encoder: LatentEncoder3D
    wrist_latent_encoder: LatentEncoder3D
    visual_encoder_type: str
    visual_encoder_metadata: Dict[str, Any]
    future_config: Dict[str, Any]
    patch_grid: Optional[Tuple[int, int]]


def build_miniwam_latent_encoders(
    cfg: dict,
    runtime_protocol: MiniWAMRuntimeProtocol,
) -> LatentEncoderPair:
    """Build the exact role-specific E_phi modules declared by a model config."""
    enc_cfg = cfg.get("latent_encoder", {})
    visual_cfg = dict(cfg.get("visual_encoder", {}))
    visual_type = str(visual_cfg.get("type", "wan_vae")).lower()
    bottleneck_cfg = enc_cfg.get("bottleneck", {})
    bottleneck_type = bottleneck_cfg.get("type", "none")
    target_grid = bottleneck_cfg.get("target_grid")
    common_encoder_kwargs = {
        "latent_dim": enc_cfg.get("latent_dim", 64),
        "compression_stages": enc_cfg.get("compression_stages"),
        "use_kl": enc_cfg.get("use_kl", True),
        "bottleneck_type": bottleneck_type,
        "target_grid": target_grid,
    }
    if visual_type == "dinov3":
        visual_cfg.setdefault("architecture", "vitb16")
        visual_cfg.setdefault("pretraining", "lvd1689m")
        visual_cfg.setdefault("frozen", True)
        visual_cfg.setdefault("input_size", 224)
        visual_cfg.setdefault("patch_size", 16)
        if visual_cfg["architecture"] != "vitb16" or visual_cfg["pretraining"] != "lvd1689m":
            raise ValueError(
                "MiniWAM supports only DINOv3 vitb16 pretrained on lvd1689m."
            )
        if not visual_cfg.get("model_id"):
            raise ValueError("DINOv3 configs must declare visual_encoder.model_id.")
        feature_cfg = dict(visual_cfg.get("feature", {}))
        feature_cfg.setdefault("type", "patch_tokens")
        feature_cfg.setdefault("normalization", "final_layernorm")
        feature_cfg.setdefault("l2_normalize", False)
        feature_cfg.setdefault("include_cls", False)
        feature_cfg.setdefault("include_registers", False)
        layer_cfg = dict(feature_cfg.get("layers", {}))
        layer_cfg.setdefault("mode", "final")
        feature_cfg["layers"] = layer_cfg
        visual_cfg["feature"] = feature_cfg
        future_cfg = dict(visual_cfg.get("future", {}))
        future_cfg.setdefault("encode_all_frames", True)
        future_cfg.setdefault("temporal_downsample_factor", 4)
        visual_cfg["future"] = future_cfg
        expected_feature = (
            feature_cfg["type"] == "patch_tokens"
            and feature_cfg["normalization"] == "final_layernorm"
            and feature_cfg["l2_normalize"] is False
            and feature_cfg["include_cls"] is False
            and feature_cfg["include_registers"] is False
            and feature_cfg.get("output_dtype") == "bfloat16"
            and layer_cfg["mode"] == "final"
            and future_cfg["encode_all_frames"] is True
            and visual_cfg["frozen"] is True
            and visual_cfg.get("compute_dtype") == "float32"
            and (visual_cfg.get("cache", {}) or {}).get("storage_dtype") == "bfloat16"
        )
        if not expected_feature:
            raise ValueError(
                "MiniWAM's DINOv3 features require a frozen backbone, final LayerNorm-normalized "
                "patch tokens without L2 normalization, no CLS/register tokens, and "
                "encode_all_frames=true, with FP32 DINO compute and a BF16 "
                "MiniWAM/cache boundary."
            )
        adapter_cfg = visual_cfg.get("adapter", {})
        input_size = int(visual_cfg.get("input_size", 224))
        patch_size = int(visual_cfg.get("patch_size", 16))
        input_dim = int(adapter_cfg.get("input_dim", 768))
        if (input_size, patch_size, input_dim) != (224, 16, 768):
            raise ValueError(
                "MiniWAM's DINOv3 features require input_size=224, patch_size=16, "
                f"and adapter.input_dim=768; got {(input_size, patch_size, input_dim)}."
            )
        patch_grid = (input_size // patch_size, input_size // patch_size)
        if common_encoder_kwargs["compression_stages"] is None:
            common_encoder_kwargs["compression_stages"] = [
                (64, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                (128, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                (128, (3, 3, 3), (2, 1, 1), (1, 1, 1)),
            ]
        encoder_cls = DINOv3LatentEncoder3D
        encoder_extra = {
            "input_dim": input_dim,
            "adapter_dim": adapter_cfg.get("hidden_dim", 64),
            "temporal_downsample_factor": future_cfg.get(
                "temporal_downsample_factor", 4
            ),
            "patch_grid": patch_grid,
        }
    elif visual_type == "wan_vae":
        encoder_cls = LatentEncoder3D
        encoder_extra = {"in_channels": enc_cfg.get("in_channels", 16)}
        future_cfg = dict(visual_cfg.get("future", {}))
        patch_grid = None
    else:
        raise ValueError(
            f"visual_encoder_type must be 'wan_vae' or 'dinov3', got {visual_type!r}."
        )

    temporal_factor = int(future_cfg.get("temporal_downsample_factor", 4))
    if temporal_factor != runtime_protocol.temporal_downsample_factor:
        raise ValueError(
            "Resolved temporal_downsample_factor does not match model configuration: "
            f"{runtime_protocol.temporal_downsample_factor} vs {temporal_factor}."
        )
    if visual_type != runtime_protocol.visual_encoder_type:
        raise ValueError(
            "Resolved visual encoder does not match model configuration: "
            f"{runtime_protocol.visual_encoder_type!r} vs {visual_type!r}."
        )

    # Main and wrist cameras get separate encoders with the same architecture.
    latent_encoder = encoder_cls(**encoder_extra, **common_encoder_kwargs)
    wrist_latent_encoder = encoder_cls(**encoder_extra, **common_encoder_kwargs)
    return LatentEncoderPair(
        latent_encoder=latent_encoder,
        wrist_latent_encoder=wrist_latent_encoder,
        visual_encoder_type=visual_type,
        visual_encoder_metadata=visual_cfg,
        future_config=future_cfg,
        patch_grid=patch_grid,
    )


class MiniWAM(nn.Module):

    def __init__(
        self,
        latent_encoder: LatentEncoder3D,
        world_dit: WorldDiT,
        action_dit: ActionDiT,
        context_encoder: ContextEncoder,
        mot: MoT,
        world_scheduler: FlowMatchScheduler,
        action_scheduler: FlowMatchScheduler,
        stage: int = 1,
        independent_timesteps: bool = True,
        lambda_latent: float = 0.0,
        lambda_action: float = 1.0,
        kl_beta: float = 1e-4,
        kl_warmup_steps: int = 1000,
        wrist_latent_encoder: Optional[LatentEncoder3D] = None,
        max_main_slots: int = 2,
        max_wrist_slots: int = 1,
        camera_rope_slot_stride: int = 32,
        visual_encoder_type: str = "wan_vae",
        visual_encoder_metadata: Optional[Dict[str, Any]] = None,
        future_rgb_horizon: int = 16,
        temporal_downsample_factor: int = 4,
        reconstruction_decoder_enabled: bool = False,
        lambda_rec: float = 0.0,
        reconstruction_hidden_dim: int = 64,
        stage1_objective: str = "inverse_dynamics",
    ):
        super().__init__()
        if int(max_main_slots) < 1:
            raise ValueError(f"max_main_slots must be positive, got {max_main_slots}.")
        if int(max_wrist_slots) < 0:
            raise ValueError(f"max_wrist_slots must be non-negative, got {max_wrist_slots}.")
        if int(camera_rope_slot_stride) < 1:
            raise ValueError(
                f"camera_rope_slot_stride must be positive, got {camera_rope_slot_stride}."
            )
        if not world_dit.use_3d_rope:
            raise ValueError("Tiled camera RoPE requires world_dit.use_3d_rope=true.")
        self.visual_encoder_type = str(visual_encoder_type).lower()
        if self.visual_encoder_type not in ("wan_vae", "dinov3"):
            raise ValueError(
                f"visual_encoder_type must be 'wan_vae' or 'dinov3', got {visual_encoder_type!r}."
            )
        self.visual_encoder_metadata = dict(visual_encoder_metadata or {})
        self.future_rgb_horizon = int(future_rgb_horizon)
        self.temporal_downsample_factor = int(temporal_downsample_factor)
        if self.future_rgb_horizon <= 0:
            raise ValueError("future_rgb_horizon must be positive.")
        if self.temporal_downsample_factor <= 0:
            raise ValueError("temporal_downsample_factor must be positive.")
        self.stage1_objective = str(stage1_objective).lower()
        if self.stage1_objective not in ("inverse_dynamics", "reconstruction_only"):
            raise ValueError(
                "stage1_objective must be 'inverse_dynamics' or 'reconstruction_only', "
                f"got {stage1_objective!r}."
            )
        self.latent_encoder = latent_encoder
        if wrist_latent_encoder is None:
            raise ValueError("MiniWAM is multiview-only and requires `wrist_latent_encoder`.")
        self.wrist_latent_encoder = wrist_latent_encoder
        self.world_dit = world_dit
        self.action_dit = action_dit
        self.context_encoder = context_encoder
        self.mot = mot
        self.world_scheduler = world_scheduler
        self.action_scheduler = action_scheduler

        self.stage = stage
        self.independent_timesteps = independent_timesteps
        self.lambda_latent = lambda_latent
        self.lambda_action = lambda_action
        self.kl_beta = kl_beta
        self.kl_warmup_steps = kl_warmup_steps
        self.max_main_slots = int(max_main_slots)
        self.max_wrist_slots = int(max_wrist_slots)
        self.camera_rope_slot_stride = int(camera_rope_slot_stride)
        encoder_dims = {
            int(self.latent_encoder.latent_dim),
            int(self.wrist_latent_encoder.latent_dim),
            int(self.world_dit.latent_dim),
        }
        if len(encoder_dims) != 1:
            raise ValueError(
                "Latent dimension mismatch between main E_phi, wrist E_phi, and WorldDiT: "
                f"{sorted(encoder_dims)}."
            )
        self.latent_dim = encoder_dims.pop()
        if self.latent_encoder.bottleneck_type != self.wrist_latent_encoder.bottleneck_type:
            raise ValueError("Main and wrist latent bottleneck types must match.")
        if self.latent_encoder.target_grid != self.wrist_latent_encoder.target_grid:
            raise ValueError("Main and wrist latent target grids must match.")
        self.latent_bottleneck_type = self.latent_encoder.bottleneck_type
        self.reconstruction_decoder_enabled = bool(reconstruction_decoder_enabled)
        self.reconstruction_hidden_dim = int(reconstruction_hidden_dim)
        self.reconstruction_target_type = (
            "dinov3_patch_features"
            if self.visual_encoder_type == "dinov3"
            else "wan_vae_features"
        )
        self.reconstruction_out_channels = (
            int((self.visual_encoder_metadata.get("adapter", {}) or {}).get("input_dim", 768))
            if self.visual_encoder_type == "dinov3"
            else 16
        )
        self.lambda_rec = float(lambda_rec)
        if self.lambda_rec < 0:
            raise ValueError("lambda_rec must be non-negative.")
        if self.lambda_rec > 0 and not self.reconstruction_decoder_enabled:
            raise ValueError("lambda_rec > 0 requires reconstruction_decoder_enabled=True.")
        if self.reconstruction_decoder_enabled:
            self.reconstruction_decoder = FutureFeatureReconstructionDecoder(
                self.latent_dim,
                hidden_dim=self.reconstruction_hidden_dim,
                out_channels=self.reconstruction_out_channels,
            )
            self.wrist_reconstruction_decoder = FutureFeatureReconstructionDecoder(
                self.latent_dim,
                hidden_dim=self.reconstruction_hidden_dim,
                out_channels=self.reconstruction_out_channels,
            )
        else:
            # No decoder: the state dict holds no reconstruction weights.
            self.reconstruction_decoder = None
            self.wrist_reconstruction_decoder = None
        if self.stage == 1 and self.stage1_objective == "reconstruction_only":
            if self.visual_encoder_type not in ("dinov3", "wan_vae"):
                raise ValueError("reconstruction_only Stage 1 requires DINOv3 or Wan-VAE features.")
            if not self.reconstruction_active:
                raise ValueError(
                    "reconstruction_only Stage 1 requires an enabled decoder and lambda_rec > 0."
                )
            self._configure_reconstruction_only_trainability()
        self._global_step = 0
        self._logged_mask = False

    def _configure_reconstruction_only_trainability(self) -> None:
        """Train only E_phi and its feature decoders (reconstruction-only Stage 1)."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        modules = (
            self.latent_encoder,
            self.wrist_latent_encoder,
            self.reconstruction_decoder,
            self.wrist_reconstruction_decoder,
        )
        for module in modules:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def set_stage(self, stage: int) -> None:
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        self.stage = stage
        if stage == 2:
            for p in self.latent_encoder.parameters():
                p.requires_grad = False
            if self.wrist_latent_encoder is not None:
                for p in self.wrist_latent_encoder.parameters():
                    p.requires_grad = False
            self.latent_encoder.eval()
            if self.wrist_latent_encoder is not None:
                self.wrist_latent_encoder.eval()
            for decoder in (self.reconstruction_decoder, self.wrist_reconstruction_decoder):
                if decoder is not None:
                    decoder.eval()
                    for p in decoder.parameters():
                        p.requires_grad = False
            logger.info("Stage 2: Froze main/wrist latent encoders (E_φ)")

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.stage == 1 and self.stage1_objective == "reconstruction_only":
            self.world_dit.eval()
            self.action_dit.eval()
            self.context_encoder.eval()
            self.mot.eval()
            self.latent_encoder.train(True)
            self.wrist_latent_encoder.train(True)
            self.reconstruction_decoder.train(True)
            self.wrist_reconstruction_decoder.train(True)
        if mode and self.stage == 2:
            self.latent_encoder.eval()
            if self.wrist_latent_encoder is not None:
                self.wrist_latent_encoder.eval()
            for decoder in (self.reconstruction_decoder, self.wrist_reconstruction_decoder):
                if decoder is not None:
                    decoder.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def reconstruction_active(self) -> bool:
        return self.stage == 1 and self.reconstruction_decoder_enabled and self.lambda_rec > 0

    def get_latent_grid_shape(self) -> tuple[int, int, int]:
        """Derive the per-camera L grid from configured/runtime visual shapes."""
        horizon = self.future_rgb_horizon
        if horizon <= 0 or horizon % self.temporal_downsample_factor:
            raise ValueError(
                f"future_rgb_horizon={horizon} must be divisible by "
                f"temporal_downsample_factor={self.temporal_downsample_factor}."
            )
        if self.visual_encoder_type == "dinov3":
            h, w = self.latent_encoder.patch_grid
            input_grid = (1 + horizon, h, w)
        else:
            h, w = self.visual_encoder_metadata.get("feature_grid", (28, 28))
            input_grid = (1 + horizon // self.temporal_downsample_factor, int(h), int(w))
        main = self.latent_encoder.output_grid_shape(input_grid)
        wrist = self.wrist_latent_encoder.output_grid_shape(input_grid)
        if main != wrist:
            raise ValueError(f"Main/wrist E_phi grid mismatch: {main} vs {wrist}.")
        return main

    def forward(
        self,
        actions: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
        z_t_main: Optional[torch.Tensor] = None,
        z_fwd_main: Optional[torch.Tensor] = None,
        main_present: Optional[torch.Tensor] = None,
        z_t_wrist: Optional[torch.Tensor] = None,
        z_fwd_wrist: Optional[torch.Tensor] = None,
        wrist_present: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        return self.training_loss(
            actions=actions,
            text_embeds=text_embeds,
            text_mask=text_mask,
            proprio=proprio,
            action_is_pad=action_is_pad,
            z_t_main=z_t_main,
            z_fwd_main=z_fwd_main,
            main_present=main_present,
            z_t_wrist=z_t_wrist,
            z_fwd_wrist=z_fwd_wrist,
            wrist_present=wrist_present,
        )


    # ── Attention mask ────────────────────────────────────────────────────

    @torch.no_grad()
    def _build_attention_mask(
        self,
        S_z0: int,
        S_L: int,
        S_a: int,
        device: torch.device,
        bidirectional_L_a: bool = False,
    ) -> torch.Tensor:
        """
        Build MoT self-attention mask for [z0, L, a] sequence.

        Stage 1 and IDM keep L independent of action tokens. Stage 2 joint
        denoising is bidirectional between L and action tokens.
        """
        total = S_z0 + S_L + S_a
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        z0_end = S_z0
        L_end = S_z0 + S_L

        # z0 → z0 only
        mask[:z0_end, :z0_end] = True
        # L → z0, L
        mask[z0_end:L_end, :L_end] = True
        # a → z0, L, a (full row)
        mask[L_end:, :] = True

        if bidirectional_L_a:
            # L also sees a
            mask[z0_end:L_end, L_end:] = True

        if not self._logged_mask:
            logger.info(
                "MoT mask: S_z0=%d S_L=%d S_a=%d total=%d bidirectional_L_a=%s",
                S_z0, S_L, S_a, total, bidirectional_L_a,
            )
            self._logged_mask = True
        return mask

    # ── KL annealing ─────────────────────────────────────────────────────

    def _get_kl_weight(self) -> float:
        if self.kl_warmup_steps <= 0:
            return self.kl_beta
        progress = min(1.0, self._global_step / self.kl_warmup_steps)
        return self.kl_beta * progress

    def _native_reconstruction_loss_per_slot(
        self,
        *,
        latent_grid: torch.Tensor,
        future_features: torch.Tensor,
        encoder: LatentEncoder3D,
        decoder: FutureFeatureReconstructionDecoder,
        batch_size: int,
        num_slots: int,
        original_future_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Return the native-feature reconstruction MSE for every flattened slot."""
        if self.visual_encoder_type == "wan_vae":
            if future_features.ndim != 5 or future_features.shape[1] != 16:
                raise ValueError(
                    "Future Wan-VAE features must be [B,V,16,T,H,W], got "
                    f"{original_future_shape}."
                )
            target = future_features.detach()
        else:
            if (
                future_features.ndim != 4
                or future_features.shape[-1] != self.reconstruction_out_channels
            ):
                raise ValueError(
                    "Future DINO features must be [B,V,T,N,D], got "
                    f"{original_future_shape}."
                )
            h, w = encoder.patch_grid
            if future_features.shape[2] != h * w:
                raise ValueError(
                    f"Future DINO patch count must be {h*w}, got {future_features.shape[2]}."
                )
            target = (
                future_features.detach()
                .reshape(
                    batch_size * num_slots,
                    future_features.shape[1],
                    h,
                    w,
                    future_features.shape[3],
                )
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )
        prediction = decoder(latent_grid, target.shape[2:])
        return F.mse_loss(
            prediction.float(), target.float(), reduction="none",
        ).mean(dim=(1, 2, 3, 4))

    def _encode_view_slots(
        self,
        z_t_slots: torch.Tensor,
        z_fwd_slots: torch.Tensor,
        present: torch.Tensor,
        encoder: LatentEncoder3D,
        z_t_for_mot_slots: torch.Tensor,
        slot_start_id: int = 0,
        reconstruction_decoder: Optional[FutureFeatureReconstructionDecoder] = None,
    ) -> dict:
        """Encode fixed camera slots with one role-specific latent encoder.

        Args:
            slot_start_id: Global slot ID for the first slot in this group.
                Used by tiled RoPE to compute per-slot width offsets.
                Main slots start at 0, wrist slots start at max_main_slots.
        """
        flattened = _flatten_role_slot_features(
            z_t_slots, z_fwd_slots, present, self.visual_encoder_type,
        )
        batch_size, num_slots = flattened.batch_size, flattened.num_slots
        zf_flat = flattened.future
        zt_mot_flat = z_t_for_mot_slots.reshape(batch_size * num_slots, *z_t_for_mot_slots.shape[2:])

        z0_flat, z0_grid_shape = self.world_dit.patchify_frame0(zt_mot_flat)
        L_flat, L_grid_flat, grid_shape, kl_per_slot = encoder(
            flattened.concatenated,
            sample=(self.stage == 1),
            kl_reduction="none",
        )

        S_z0 = z0_flat.shape[1]
        S_L = L_flat.shape[1]
        z0_tokens = z0_flat.reshape(batch_size, num_slots * S_z0, -1)
        L_tokens = L_flat.reshape(batch_size, num_slots * S_L, -1)
        present_bool = present.to(device=z_t_slots.device, dtype=torch.bool)
        z0_present = present_bool.unsqueeze(-1).expand(-1, -1, S_z0).reshape(batch_size, num_slots * S_z0)
        L_present = present_bool.unsqueeze(-1).expand(-1, -1, S_L).reshape(batch_size, num_slots * S_L)
        present_float = present_bool.reshape(batch_size * num_slots).to(dtype=kl_per_slot.dtype)
        kl_sum = (kl_per_slot * present_float).sum()
        kl_count = present_float.sum()
        rec_sum = L_grid_flat.new_zeros(())
        rec_count = L_grid_flat.new_zeros(())
        if reconstruction_decoder is not None:
            rec_per_slot = self._native_reconstruction_loss_per_slot(
                latent_grid=L_grid_flat,
                future_features=zf_flat,
                encoder=encoder,
                decoder=reconstruction_decoder,
                batch_size=batch_size,
                num_slots=num_slots,
                original_future_shape=tuple(z_fwd_slots.shape),
            )
            rec_sum = (rec_per_slot * present_float.float()).sum()
            rec_count = present_float.sum()

        # Tiled RoPE: each camera slot gets its own horizontal coordinate band.
        z0_w = z0_grid_shape[1] if z0_grid_shape is not None else int(S_z0 ** 0.5)
        L_w = grid_shape[2]
        if z0_w > self.camera_rope_slot_stride or L_w > self.camera_rope_slot_stride:
            raise ValueError(
                "Camera grid width exceeds the configured tiled-RoPE stride: "
                f"z0_width={z0_w}, latent_width={L_w}, "
                f"stride={self.camera_rope_slot_stride}."
            )
        z0_freq_parts = []
        L_freq_parts = []
        for slot_i in range(num_slots):
            global_slot_id = slot_start_id + slot_i
            w_offset = global_slot_id * self.camera_rope_slot_stride
            z0_freq_parts.append(
                self.world_dit._build_z0_rope_freqs(
                    S_z0, z_t_slots.device,
                    z0_grid_shape=z0_grid_shape,
                    w_offset=w_offset,
                )
            )
            L_freq_parts.append(
                self.world_dit._build_3d_rope_freqs(
                    grid_shape, z_t_slots.device,
                    w_offset=w_offset,
                )
            )
        z0_freqs = torch.cat(z0_freq_parts, dim=0)
        L_freqs = torch.cat(L_freq_parts, dim=0)

        return {
            "z0_tokens": z0_tokens,
            "L_tokens": L_tokens,
            "z0_present": z0_present,
            "L_present": L_present,
            "z0_freqs": z0_freqs,
            "L_freqs": L_freqs,
            "kl_sum": kl_sum,
            "kl_count": kl_count,
            "grid_shape": grid_shape,
            "rec_sum": rec_sum,
            "rec_count": rec_count,
        }

    def _reconstruction_only_view_slots(
        self,
        *,
        z_t_slots: torch.Tensor,
        z_fwd_slots: torch.Tensor,
        present: torch.Tensor,
        encoder: LatentEncoder3D,
        decoder: FutureFeatureReconstructionDecoder,
    ) -> dict[str, torch.Tensor]:
        """Run only E_phi and its decoder, without constructing any DiT tokens."""
        flattened = _flatten_role_slot_features(
            z_t_slots, z_fwd_slots, present, self.visual_encoder_type,
        )
        batch_size, num_slots = flattened.batch_size, flattened.num_slots
        _, latent_grid, _, kl_per_slot = encoder(
            flattened.concatenated,
            sample=True,
            kl_reduction="none",
        )
        present_float = present.to(
            device=z_t_slots.device, dtype=kl_per_slot.dtype,
        ).reshape(batch_size * num_slots)
        reconstruction_per_slot = self._native_reconstruction_loss_per_slot(
            latent_grid=latent_grid,
            future_features=flattened.future,
            encoder=encoder,
            decoder=decoder,
            batch_size=batch_size,
            num_slots=num_slots,
            original_future_shape=tuple(z_fwd_slots.shape),
        )
        return {
            "kl_sum": (kl_per_slot * present_float).sum(),
            "kl_count": present_float.sum(),
            "rec_sum": (reconstruction_per_slot * present_float.float()).sum(),
            "rec_count": present_float.sum(),
        }

    def _training_loss_reconstruction_only(
        self,
        *,
        z_t_main: torch.Tensor,
        z_fwd_main: torch.Tensor,
        main_present: torch.Tensor,
        z_t_wrist: torch.Tensor,
        z_fwd_wrist: torch.Tensor,
        wrist_present: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if self.stage != 1 or self.stage1_objective != "reconstruction_only":
            raise RuntimeError("The reconstruction-only loss is valid only for its Stage-1 objective.")
        if float(self.lambda_action) != 0.0 or float(self.lambda_latent) != 0.0:
            raise ValueError(
                "reconstruction_only Stage 1 requires lambda_action=lambda_latent=0."
            )
        actual_future = int(
            z_fwd_main.shape[3]
            if self.visual_encoder_type == "wan_vae"
            else z_fwd_main.shape[2]
        )
        expected_future = (
            self.future_rgb_horizon // self.temporal_downsample_factor
            if self.visual_encoder_type == "wan_vae"
            else self.future_rgb_horizon
        )
        if actual_future != expected_future:
            raise ValueError(
                f"Configured {self.visual_encoder_type} future horizon expects {expected_future} feature frames "
                f"from future_rgb_horizon={self.future_rgb_horizon}, got {actual_future}."
            )
        num_main_slots = int(z_t_main.shape[1])
        num_wrist_slots = int(z_t_wrist.shape[1])
        if num_main_slots < 1 or num_main_slots > self.max_main_slots:
            raise ValueError(
                f"Main camera slot count {num_main_slots} is outside [1, {self.max_main_slots}]."
            )
        if num_wrist_slots < 0 or num_wrist_slots > self.max_wrist_slots:
            raise ValueError(
                f"Wrist camera slot count {num_wrist_slots} is outside [0, {self.max_wrist_slots}]."
            )

        parts = []
        if bool(main_present.any().item()):
            parts.append(self._reconstruction_only_view_slots(
                z_t_slots=z_t_main,
                z_fwd_slots=z_fwd_main,
                present=main_present,
                encoder=self.latent_encoder,
                decoder=self.reconstruction_decoder,
            ))
        if num_wrist_slots and bool(wrist_present.any().item()):
            parts.append(self._reconstruction_only_view_slots(
                z_t_slots=z_t_wrist,
                z_fwd_slots=z_fwd_wrist,
                present=wrist_present,
                encoder=self.wrist_latent_encoder,
                decoder=self.wrist_reconstruction_decoder,
            ))
        if not parts:
            raise ValueError("Multiview batch has no present camera slots.")

        kl_count = torch.stack([part["kl_count"] for part in parts]).sum().clamp(min=1.0)
        rec_count = torch.stack([part["rec_count"] for part in parts]).sum().clamp(min=1.0)
        kl_loss = torch.stack([part["kl_sum"] for part in parts]).sum() / kl_count
        reconstruction_loss = torch.stack([part["rec_sum"] for part in parts]).sum() / rec_count
        kl_weight = self._get_kl_weight()
        weighted_kl = kl_loss * kl_weight
        weighted_reconstruction = reconstruction_loss * self.lambda_rec
        loss_total = weighted_reconstruction + weighted_kl
        loss_dict = {
            "kl_loss": float(kl_loss.detach().item()),
            "weighted_kl_loss": float(weighted_kl.detach().item()),
            "loss_kl": float(weighted_kl.detach().item()),
            "feature_reconstruction_loss": float(reconstruction_loss.detach().item()),
            "weighted_feature_reconstruction_loss": float(
                weighted_reconstruction.detach().item()
            ),
            "loss_total": float(loss_total.detach().item()),
        }
        self._global_step += 1
        return loss_total, loss_dict

    def _prepare_multiview_inputs(
        self,
        z_t_main: torch.Tensor,
        z_fwd_main: torch.Tensor,
        main_present: torch.Tensor,
        z_t_wrist: torch.Tensor,
        z_fwd_wrist: torch.Tensor,
        wrist_present: torch.Tensor,
    ) -> dict:
        actual_future = int(z_fwd_main.shape[3] if self.visual_encoder_type == "wan_vae" else z_fwd_main.shape[2])
        expected_future = (
            self.future_rgb_horizon // self.temporal_downsample_factor
            if self.visual_encoder_type == "wan_vae"
            else self.future_rgb_horizon
        )
        if actual_future != expected_future:
            raise ValueError(
                f"Configured {self.visual_encoder_type} future horizon expects {expected_future} feature frames "
                f"from future_rgb_horizon={self.future_rgb_horizon}, got {actual_future}."
            )
        num_main_slots = int(z_t_main.shape[1])
        if num_main_slots < 1 or num_main_slots > self.max_main_slots:
            raise ValueError(
                f"Main camera slot count {num_main_slots} is outside [1, {self.max_main_slots}]."
            )
        num_wrist_slots = 0 if z_t_wrist is None else int(z_t_wrist.shape[1])
        if num_wrist_slots < 0 or num_wrist_slots > self.max_wrist_slots:
            raise ValueError(
                f"Wrist camera slot count {num_wrist_slots} is outside [0, {self.max_wrist_slots}]."
            )

        z_t_main_for_mot = z_t_main
        z_t_wrist_for_mot = z_t_wrist
        if self.stage == 1:
            # Stage 1 removes current-observation conditioning from the MoT.
            z_t_main_for_mot = torch.zeros_like(z_t_main)
            if z_t_wrist is not None:
                z_t_wrist_for_mot = torch.zeros_like(z_t_wrist)

        parts = []
        if z_t_main.shape[1] > 0 and bool(main_present.any().item()):
            parts.append(self._encode_view_slots(
                z_t_slots=z_t_main,
                z_fwd_slots=z_fwd_main,
                present=main_present,
                encoder=self.latent_encoder,
                z_t_for_mot_slots=z_t_main_for_mot,
                slot_start_id=0,
                reconstruction_decoder=(
                    self.reconstruction_decoder if self.reconstruction_active else None
                ),
            ))

        has_wrist_slots = z_t_wrist is not None and z_t_wrist.shape[1] > 0
        has_wrist_present = (
            has_wrist_slots
            and wrist_present is not None
            and bool(wrist_present.any().item())
        )
        if has_wrist_present:
            if self.wrist_latent_encoder is None:
                raise ValueError("Wrist-view slots are present but `wrist_latent_encoder` is not configured.")
            parts.append(self._encode_view_slots(
                z_t_slots=z_t_wrist,
                z_fwd_slots=z_fwd_wrist,
                present=wrist_present,
                encoder=self.wrist_latent_encoder,
                z_t_for_mot_slots=z_t_wrist_for_mot,
                slot_start_id=self.max_main_slots,
                reconstruction_decoder=(
                    self.wrist_reconstruction_decoder if self.reconstruction_active else None
                ),
            ))

        if not parts:
            raise ValueError("Multiview batch has no present camera slots.")

        z0_tokens = torch.cat([part["z0_tokens"] for part in parts], dim=1)
        L_clean = torch.cat([part["L_tokens"] for part in parts], dim=1)
        z0_present = torch.cat([part["z0_present"] for part in parts], dim=1)
        L_present = torch.cat([part["L_present"] for part in parts], dim=1)
        z0_freqs = torch.cat([part["z0_freqs"] for part in parts], dim=0)
        L_freqs = torch.cat([part["L_freqs"] for part in parts], dim=0)
        kl_sum = torch.stack([part["kl_sum"] for part in parts]).sum()
        kl_count = torch.stack([part["kl_count"] for part in parts]).sum().clamp(min=1.0)
        rec_sum = torch.stack([part["rec_sum"] for part in parts]).sum()
        rec_count = torch.stack([part["rec_count"] for part in parts]).sum().clamp(min=1.0)
        grid_shapes = {tuple(part["grid_shape"]) for part in parts}
        if len(grid_shapes) != 1:
            raise ValueError(f"All camera roles must produce the same latent grid, got {sorted(grid_shapes)}.")
        return {
            "z0_tokens": z0_tokens,
            "L_clean": L_clean,
            "z0_present": z0_present,
            "L_present": L_present,
            "z0_freqs": z0_freqs,
            "L_freqs": L_freqs,
            "kl_loss": kl_sum / kl_count,
            "feature_reconstruction_loss": rec_sum / rec_count,
            "grid_shape": grid_shapes.pop(),
        }

    @staticmethod
    def _apply_token_presence_to_mask(
        base_mask: torch.Tensor,
        token_present: torch.Tensor,
    ) -> torch.Tensor:
        """Create [B,1,S,S] mask; absent query rows attend only to themselves."""
        batch_size, seq_len = token_present.shape
        mask = base_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
        key_present = token_present.unsqueeze(1).expand(-1, seq_len, -1)
        query_present = token_present.unsqueeze(2).expand(-1, -1, seq_len)
        mask = mask & key_present & query_present
        absent = ~token_present
        if absent.any():
            eye = torch.eye(seq_len, dtype=torch.bool, device=token_present.device).unsqueeze(0)
            mask = torch.where(absent.unsqueeze(2), eye.expand(batch_size, -1, -1), mask)
        return mask.unsqueeze(1)

    def _compute_action_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        timestep_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_loss_per_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_per_token.device,
                dtype=action_loss_per_token.dtype,
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_per_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_per_token.mean(dim=1)
        action_weight = self.action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        return (action_loss_per_sample * action_weight).mean()

    def _compute_latent_loss(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        timestep_latent: torch.Tensor,
        world_present: torch.Tensor,
    ) -> torch.Tensor:
        latent_loss_per_token = F.mse_loss(
            pred_latent.float(), target_latent.float(), reduction="none"
        ).mean(dim=2)
        valid_latent = world_present.to(
            device=latent_loss_per_token.device,
            dtype=latent_loss_per_token.dtype,
        )
        valid_sum = valid_latent.sum(dim=1).clamp(min=1.0)
        latent_loss = (latent_loss_per_token * valid_latent).sum(dim=1) / valid_sum
        latent_weight = self.world_scheduler.training_weight(timestep_latent).to(
            latent_loss.device, dtype=latent_loss.dtype,
        )
        return (latent_loss * latent_weight).mean()

    # ── Training loss ────────────────────────────────────────────────────

    def training_loss(
        self,
        actions: Optional[torch.Tensor],
        text_embeds: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
        z_t_main: Optional[torch.Tensor] = None,
        z_fwd_main: Optional[torch.Tensor] = None,
        main_present: Optional[torch.Tensor] = None,
        z_t_wrist: Optional[torch.Tensor] = None,
        z_fwd_wrist: Optional[torch.Tensor] = None,
        wrist_present: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute multiview MiniWAM training loss."""
        required_mv = {
            "z_t_main": z_t_main,
            "z_fwd_main": z_fwd_main,
            "main_present": main_present,
            "z_t_wrist": z_t_wrist,
            "z_fwd_wrist": z_fwd_wrist,
            "wrist_present": wrist_present,
        }
        missing_mv = [name for name, value in required_mv.items() if value is None]
        if missing_mv:
            raise ValueError(
                f"MiniWAM is multiview-only; missing required tensors: {missing_mv}."
            )
        if self.stage == 1 and self.stage1_objective == "reconstruction_only":
            return self._training_loss_reconstruction_only(
                z_t_main=z_t_main,
                z_fwd_main=z_fwd_main,
                main_present=main_present,
                z_t_wrist=z_t_wrist,
                z_fwd_wrist=z_fwd_wrist,
                wrist_present=wrist_present,
            )
        if actions is None or text_embeds is None or text_mask is None:
            raise ValueError("`actions`, `text_embeds`, and `text_mask` are required.")
        batch_size = actions.shape[0]

        # 0. Stage 1 removes language conditioning: actions are inferred from the transition alone.
        if self.stage == 1:
            text_embeds = torch.zeros_like(text_embeds)

        # 1. Text context (cross-attention only)
        text_ctx, ctx_mask = self.context_encoder.encode_context(
            text_embeds, text_mask, proprio,
        )

        mv = self._prepare_multiview_inputs(
            z_t_main=z_t_main,
            z_fwd_main=z_fwd_main,
            main_present=main_present,
            z_t_wrist=z_t_wrist,
            z_fwd_wrist=z_fwd_wrist,
            wrist_present=wrist_present,
        )
        z0_tokens = mv["z0_tokens"]
        z0_grid_shape = None
        L_clean = mv["L_clean"]
        grid_shape = mv["grid_shape"]
        kl_loss = mv["kl_loss"]
        feature_reconstruction_loss = mv["feature_reconstruction_loss"]
        z0_freqs = mv["z0_freqs"]
        latent_freqs = mv["L_freqs"]
        z0_present = mv["z0_present"].to(device=self.device)
        world_present = mv["L_present"].to(device=self.device)

        # 4. Prepare noisy action (always noised)
        noise_action = torch.randn_like(actions)
        timestep_action = self.action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=actions.dtype,
        )
        noisy_action = self.action_scheduler.add_noise(actions, noise_action, timestep_action)
        target_action = self.action_scheduler.training_target(actions, noise_action, timestep_action)

        # 5. Prepare latent noise
        if self.stage == 1:
            noisy_latent = L_clean
            timestep_latent = torch.zeros(batch_size, device=self.device, dtype=L_clean.dtype)
            target_latent = None
        else:
            L_det = L_clean.detach()
            noise_latent = torch.randn_like(L_det)
            if self.independent_timesteps:
                timestep_latent = self.world_scheduler.sample_training_t(
                    batch_size=batch_size, device=self.device, dtype=L_det.dtype,
                )
            else:
                timestep_latent = timestep_action.clone()
            noisy_latent = self.world_scheduler.add_noise(L_det, noise_latent, timestep_latent)
            target_latent = self.world_scheduler.training_target(L_det, noise_latent, timestep_latent)

        # 6. Expert pre_dit
        world_pre = self.world_dit.pre_dit(
            world_tokens=noisy_latent,
            timestep=timestep_latent,
            context=text_ctx,
            context_mask=ctx_mask,
            grid_shape=grid_shape,
            z0_tokens=z0_tokens,
            z0_grid_shape=z0_grid_shape,
            z0_freqs=z0_freqs,
            latent_freqs=latent_freqs,
        )
        action_pre = self.action_dit.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=text_ctx,
            context_mask=ctx_mask,
        )

        # 7. Build mask and run MoT
        S_z0 = world_pre["meta"]["S_z0"]
        S_L = world_pre["meta"]["seq_len_L"]
        S_a = action_pre["meta"]["seq_len"]

        base_attention_mask = self._build_attention_mask(
            S_z0, S_L, S_a, self.device,
            bidirectional_L_a=(self.stage == 2),
        )
        action_present = torch.ones(
            (batch_size, S_a), dtype=torch.bool, device=self.device,
        )
        token_present = torch.cat([z0_present, world_present, action_present], dim=1)
        attention_mask = self._apply_token_presence_to_mask(
            base_attention_mask, token_present,
        )

        tokens_out = self.mot(
            embeds_all={"world": world_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"world": world_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "world": {"context": world_pre["context"], "mask": world_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"world": world_pre["t_mod"], "action": action_pre["t_mod"]},
        )

        # 8. Post-dit and losses
        pred_action = self.action_dit.post_dit(tokens_out["action"], action_pre)

        loss_action = self._compute_action_loss(
            pred_action, target_action, timestep_action, action_is_pad,
        )
        loss_dict = {
            "action_loss": float(loss_action.detach().item()),
            "loss_action": float(loss_action.detach().item()),
        }

        # Latent loss (Stage 2 only)
        if self.stage == 2 and target_latent is not None:
            pred_latent = self.world_dit.post_dit(tokens_out["world"], world_pre)
            loss_latent = self._compute_latent_loss(
                pred_latent, target_latent, timestep_latent, world_present,
            )
            loss_dict["loss_latent"] = float(loss_latent.detach().item())
        else:
            loss_latent = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # KL loss (Stage 1)
        kl_w = self._get_kl_weight() if self.stage == 1 else 0.0
        loss_kl = kl_w * kl_loss if kl_w > 0 else torch.tensor(0.0, device=self.device)
        weighted_rec = (
            self.lambda_rec * feature_reconstruction_loss
            if self.reconstruction_active
            else torch.tensor(0.0, device=self.device)
        )
        loss_dict["kl_loss"] = float(kl_loss.detach().item())
        loss_dict["weighted_kl_loss"] = float(loss_kl.detach().item())
        loss_dict["loss_kl"] = float(loss_kl.detach().item())
        loss_dict["feature_reconstruction_loss"] = float(
            feature_reconstruction_loss.detach().item()
        )
        loss_dict["weighted_feature_reconstruction_loss"] = float(weighted_rec.detach().item())
        if self.visual_encoder_type == "wan_vae":
            # Also log the reconstruction losses under VAE-specific names; the
            # feature_* keys above are the same for every visual encoder.
            loss_dict["vae_reconstruction_loss"] = loss_dict["feature_reconstruction_loss"]
            loss_dict["weighted_vae_reconstruction_loss"] = loss_dict[
                "weighted_feature_reconstruction_loss"
            ]

        loss_total = (
            self.lambda_action * loss_action
            + self.lambda_latent * loss_latent
            + loss_kl
            + weighted_rec
        )
        loss_dict["loss_total"] = float(loss_total.detach().item())
        self._global_step += 1
        return loss_total, loss_dict

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(
        self,
        z_t_main: torch.Tensor,
        main_present: torch.Tensor,
        z_t_wrist: torch.Tensor,
        wrist_present: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Multiview inference. Returns actions [B, action_horizon, action_dim]."""
        if any(x is None for x in (z_t_main, main_present, z_t_wrist, wrist_present)):
            raise ValueError("MiniWAM inference is multiview-only; z_t_main/main_present/z_t_wrist/wrist_present are required.")

        self.eval()
        # DINOv3 features arrive at their BF16 boundary; compute in the model's dtype.
        z_t_main = z_t_main.to(device=self.device, dtype=self.dtype)
        z_t_wrist = z_t_wrist.to(device=self.device, dtype=self.dtype)
        batch_size = z_t_main.shape[0]

        # Text context
        text_ctx, ctx_mask = self.context_encoder.encode_context(
            text_embeds, text_mask, proprio,
        )

        rgb_horizon = self.future_rgb_horizon
        if self.visual_encoder_type == "wan_vae":
            num_fwd = rgb_horizon // self.temporal_downsample_factor
            z_fwd_main = torch.zeros(
                (*z_t_main.shape[:3], num_fwd, *z_t_main.shape[4:]),
                device=z_t_main.device, dtype=z_t_main.dtype,
            )
            z_fwd_wrist = torch.zeros(
                (*z_t_wrist.shape[:3], num_fwd, *z_t_wrist.shape[4:]),
                device=z_t_wrist.device, dtype=z_t_wrist.dtype,
            )
        else:
            z_fwd_main = torch.zeros(
                (*z_t_main.shape[:2], rgb_horizon, *z_t_main.shape[2:]),
                device=z_t_main.device, dtype=z_t_main.dtype,
            )
            z_fwd_wrist = torch.zeros(
                (*z_t_wrist.shape[:2], rgb_horizon, *z_t_wrist.shape[2:]),
                device=z_t_wrist.device, dtype=z_t_wrist.dtype,
            )
        mv = self._prepare_multiview_inputs(
            z_t_main=z_t_main,
            z_fwd_main=z_fwd_main,
            main_present=main_present,
            z_t_wrist=z_t_wrist,
            z_fwd_wrist=z_fwd_wrist,
            wrist_present=wrist_present,
        )

        z0_tokens = mv["z0_tokens"]
        L_clean = mv["L_clean"]
        z0_freqs = mv["z0_freqs"]
        latent_freqs = mv["L_freqs"]
        z0_present = mv["z0_present"].to(device=self.device)
        world_present = mv["L_present"].to(device=self.device)
        z0_grid_shape = None
        grid_shape = mv["grid_shape"]
        S_z0 = z0_tokens.shape[1]
        num_L = L_clean.shape[1]

        # Initialize noise: actions first, then world tokens, from one generator.
        gen = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
        a_noise = torch.randn(
            (batch_size, action_horizon, self.action_dit.action_dim),
            generator=gen, device="cpu", dtype=torch.float32,
        ).to(device=self.device, dtype=self.dtype)
        L_noise = torch.randn(
            (batch_size, num_L, self.world_dit.latent_dim),
            generator=gen, device="cpu", dtype=torch.float32,
        ).to(device=self.device, dtype=self.dtype)

        # Build schedules
        action_ts, action_ds = self.action_scheduler.build_inference_schedule(
            num_inference_steps, self.device, self.dtype,
        )
        latent_ts, latent_ds = self.world_scheduler.build_inference_schedule(
            num_inference_steps, self.device, self.dtype,
        )

        # Joint denoising of world and action tokens
        for i in range(num_inference_steps):
            t_a = action_ts[i].unsqueeze(0)
            t_L = latent_ts[i].unsqueeze(0)
            world_pre = self.world_dit.pre_dit(
                L_noise, t_L, text_ctx, ctx_mask, grid_shape, z0_tokens,
                z0_grid_shape=z0_grid_shape,
                z0_freqs=z0_freqs,
                latent_freqs=latent_freqs,
            )
            action_pre = self.action_dit.pre_dit(
                a_noise, t_a, text_ctx, ctx_mask,
            )
            S_L = world_pre["meta"]["seq_len_L"]
            base_mask = self._build_attention_mask(
                S_z0, S_L, action_pre["meta"]["seq_len"], self.device,
                bidirectional_L_a=True,
            )
            action_present = torch.ones(
                (batch_size, action_pre["meta"]["seq_len"]), dtype=torch.bool, device=self.device,
            )
            mask = self._apply_token_presence_to_mask(
                base_mask, torch.cat([z0_present, world_present, action_present], dim=1),
            )

            tokens_out = self.mot(
                embeds_all={"world": world_pre["tokens"], "action": action_pre["tokens"]},
                attention_mask=mask,
                freqs_all={"world": world_pre["freqs"], "action": action_pre["freqs"]},
                context_all={
                    "world": {"context": world_pre["context"], "mask": world_pre["context_mask"]},
                    "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                },
                t_mod_all={"world": world_pre["t_mod"], "action": action_pre["t_mod"]},
            )

            pred_a = self.action_dit.post_dit(tokens_out["action"], action_pre)
            a_noise = self.action_scheduler.step(pred_a, action_ds[i], a_noise)
            pred_L = self.world_dit.post_dit(tokens_out["world"], world_pre)
            L_noise = self.world_scheduler.step(pred_L, latent_ds[i], L_noise)

        return a_noise.detach().cpu().float()

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        runtime_protocol: MiniWAMRuntimeProtocol,
    ) -> "MiniWAM":
        """Build MiniWAM from a resolved model config."""
        enc_cfg = cfg.get("latent_encoder", {})
        mv_cfg = cfg.get("multi_view_latent_encoder", {})
        encoders = build_miniwam_latent_encoders(cfg, runtime_protocol)
        latent_encoder = encoders.latent_encoder
        wrist_latent_encoder = encoders.wrist_latent_encoder
        visual_cfg = encoders.visual_encoder_metadata
        visual_type = encoders.visual_encoder_type
        future_cfg = encoders.future_config
        patch_grid = encoders.patch_grid

        mot_cfg = cfg.get("mot", {})
        num_heads = mot_cfg.get("num_heads", 12)
        attn_head_dim = mot_cfg.get("attn_head_dim", 128)
        num_layers = mot_cfg.get("num_layers", 12)

        world_dit_cfg = cfg.get("world_dit", {})
        world_dit = WorldDiT(
            hidden_dim=world_dit_cfg.get("hidden_dim", 1536),
            latent_dim=enc_cfg.get("latent_dim", 64),
            ffn_dim=world_dit_cfg.get("ffn_dim", 6144),
            text_dim=world_dit_cfg.get("text_dim", 4096),
            freq_dim=world_dit_cfg.get("freq_dim", 256),
            eps=world_dit_cfg.get("eps", 1e-6),
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            num_layers=num_layers,
            use_gradient_checkpointing=world_dit_cfg.get("use_gradient_checkpointing", True),
            use_3d_rope=world_dit_cfg.get("use_3d_rope", True),
            vae_channels=world_dit_cfg.get("vae_channels", 16),
            frame0_patch_size=tuple(world_dit_cfg.get("frame0_patch_size", [2, 2])),
            visual_encoder_type=visual_type,
            current_feature_dim=visual_cfg.get("adapter", {}).get("input_dim", 768),
            current_patch_grid=patch_grid or (14, 14),
        )

        adit_cfg = cfg.get("action_dit", {})
        action_dit = ActionDiT(
            hidden_dim=adit_cfg.get("hidden_dim", 1536),
            action_dim=adit_cfg.get("action_dim", 7),
            ffn_dim=adit_cfg.get("ffn_dim", 6144),
            text_dim=adit_cfg.get("text_dim", 4096),
            freq_dim=adit_cfg.get("freq_dim", 256),
            eps=adit_cfg.get("eps", 1e-6),
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            num_layers=num_layers,
            use_gradient_checkpointing=adit_cfg.get("use_gradient_checkpointing", True),
        )

        ctx_cfg = cfg.get("context_encoder", {})
        context_encoder = ContextEncoder(
            text_dim=ctx_cfg.get("text_dim", 4096),
            hidden_dim=ctx_cfg.get("hidden_dim", 1536),
            proprio_dim=ctx_cfg.get("proprio_dim"),
        )

        mot = MoT(
            mixtures={"world": world_dit, "action": action_dit},
            mot_checkpoint_mixed_attn=mot_cfg.get("checkpoint_mixed_attn", True),
        )

        ls_cfg = cfg.get("world_scheduler", {})
        world_scheduler = FlowMatchScheduler(
            num_train_timesteps=ls_cfg.get("num_train_timesteps", 1000),
            shift=ls_cfg.get("train_shift", 5.0),
        )
        as_cfg = cfg.get("action_scheduler", {})
        action_scheduler = FlowMatchScheduler(
            num_train_timesteps=as_cfg.get("num_train_timesteps", 1000),
            shift=as_cfg.get("train_shift", 5.0),
        )

        loss_cfg = cfg.get("loss", {})
        rec_cfg = cfg.get("reconstruction", {})
        temporal_factor = int(future_cfg.get("temporal_downsample_factor", 4))
        if temporal_factor != runtime_protocol.temporal_downsample_factor:
            raise ValueError(
                "Resolved temporal_downsample_factor does not match model configuration: "
                f"{runtime_protocol.temporal_downsample_factor} vs {temporal_factor}."
            )
        if visual_type != runtime_protocol.visual_encoder_type:
            raise ValueError(
                "Resolved visual encoder does not match model configuration: "
                f"{runtime_protocol.visual_encoder_type!r} vs {visual_type!r}."
            )
        return cls(
            latent_encoder=latent_encoder,
            world_dit=world_dit,
            action_dit=action_dit,
            context_encoder=context_encoder,
            mot=mot,
            world_scheduler=world_scheduler,
            action_scheduler=action_scheduler,
            stage=cfg.get("stage", 1),
            independent_timesteps=cfg.get("independent_timesteps", True),
            lambda_latent=loss_cfg.get("lambda_latent", 0.0),
            lambda_action=loss_cfg.get("lambda_action", 1.0),
            kl_beta=loss_cfg.get("kl_beta", 1e-4),
            kl_warmup_steps=loss_cfg.get("kl_warmup_steps", 1000),
            wrist_latent_encoder=wrist_latent_encoder,
            max_main_slots=mv_cfg.get("max_main_slots", 2),
            max_wrist_slots=mv_cfg.get("max_wrist_slots", 1),
            camera_rope_slot_stride=mv_cfg.get("camera_rope_slot_stride", 32),
            visual_encoder_type=visual_type,
            visual_encoder_metadata=visual_cfg,
            future_rgb_horizon=runtime_protocol.future_rgb_horizon,
            temporal_downsample_factor=temporal_factor,
            reconstruction_decoder_enabled=rec_cfg.get("decoder_enabled", False),
            lambda_rec=rec_cfg.get("lambda_rec", loss_cfg.get("lambda_rec", 0.0)),
            reconstruction_hidden_dim=rec_cfg.get("hidden_dim", 64),
            stage1_objective=cfg.get("stage1_objective", "inverse_dynamics"),
        )
