"""Multiview direct native-feature MiniWAM ablation.

This ablation predicts either Wan-VAE or DINOv3 future features directly
instead of predicting the compact E_phi latent. It uses one shared VideoDiT
for all camera slots and the same multiview slot/present contract as MiniWAM.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.action_dit import ActionDiT
from .models.context_encoder import ContextEncoder
from .models.mot import MoT
from .models.scheduler import FlowMatchScheduler
from .models.video_dit import VideoDiT


class MiniWAMNativeFeatureModel(nn.Module):
    """Direct native-feature ablation with a shared multiview video expert.

    """

    def __init__(
        self,
        video_dit: VideoDiT,
        action_dit: ActionDiT,
        context_encoder: ContextEncoder,
        mot: MoT,
        video_scheduler: FlowMatchScheduler,
        action_scheduler: FlowMatchScheduler,
        independent_timesteps: bool = True,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        default_num_future_frames: Optional[int] = None,
        max_main_slots: int = 2,
        max_wrist_slots: int = 1,
        camera_rope_slot_stride: int = 32,
        visual_encoder_type: str = "wan_vae",
        visual_encoder_metadata: Optional[dict] = None,
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
        self.video_dit = video_dit
        self.action_dit = action_dit
        self.context_encoder = context_encoder
        self.mot = mot
        self.video_scheduler = video_scheduler
        self.action_scheduler = action_scheduler
        self.independent_timesteps = bool(independent_timesteps)
        self.lambda_video = float(lambda_video)
        self.lambda_action = float(lambda_action)
        self.default_num_future_frames = default_num_future_frames
        self.max_main_slots = int(max_main_slots)
        self.max_wrist_slots = int(max_wrist_slots)
        self.camera_rope_slot_stride = int(camera_rope_slot_stride)
        self.visual_encoder_type = str(visual_encoder_type).lower()
        if self.visual_encoder_type not in ("wan_vae", "dinov3"):
            raise ValueError(
                "visual_encoder_type must be 'wan_vae' or 'dinov3', got "
                f"{visual_encoder_type!r}."
            )
        self.visual_encoder_metadata = dict(visual_encoder_metadata or {})
        self.temporal_downsample_factor = int(
            (self.visual_encoder_metadata.get("future", {}) or {}).get(
                "temporal_downsample_factor", 4,
            )
        )
        self.model_architecture = "native"
        self._global_step = 0

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

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

    @staticmethod
    def _apply_token_presence_to_mask(
        base_mask: torch.Tensor,
        token_present: torch.Tensor,
    ) -> torch.Tensor:
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

    def _compose_slots(
        self,
        z_t_main: torch.Tensor,
        z_fwd_main: torch.Tensor,
        main_present: torch.Tensor,
        z_t_wrist: torch.Tensor,
        z_fwd_wrist: torch.Tensor,
        wrist_present: torch.Tensor,
    ) -> dict:
        z_t_main, z_fwd_main = self._canonicalize_role_features(
            z_t_main, z_fwd_main, "main",
        )
        z_t_wrist, z_fwd_wrist = self._canonicalize_role_features(
            z_t_wrist, z_fwd_wrist, "wrist",
        )
        for name, tensor in {
            "z_t_main": z_t_main,
            "z_fwd_main": z_fwd_main,
            "z_t_wrist": z_t_wrist,
            "z_fwd_wrist": z_fwd_wrist,
        }.items():
            if tensor is None or tensor.ndim != 6:
                raise ValueError(f"`{name}` must be [B,V,C,T,H,W], got {None if tensor is None else tuple(tensor.shape)}")
        if main_present.shape != z_t_main.shape[:2]:
            raise ValueError(f"`main_present` must be {tuple(z_t_main.shape[:2])}, got {tuple(main_present.shape)}")
        if wrist_present.shape != z_t_wrist.shape[:2]:
            raise ValueError(f"`wrist_present` must be {tuple(z_t_wrist.shape[:2])}, got {tuple(wrist_present.shape)}")
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
        z_t_slots = torch.cat([z_t_main, z_t_wrist], dim=1)
        z_fwd_slots = torch.cat([z_fwd_main, z_fwd_wrist], dim=1)
        slot_present = torch.cat([main_present, wrist_present], dim=1).to(
            device=z_t_slots.device, dtype=torch.bool,
        )
        if not bool(slot_present.any().item()):
            raise ValueError("Multiview direct-feature batch has no present camera slots.")
        return {
            "z_t_slots": z_t_slots,
            "z_fwd_slots": z_fwd_slots,
            "z_video": torch.cat([z_t_slots, z_fwd_slots], dim=3),
            "slot_present": slot_present,
            "slot_ids": tuple(range(num_main_slots)) + tuple(
                self.max_main_slots + slot for slot in range(num_wrist_slots)
            ),
            "num_main_slots": num_main_slots,
            "num_wrist_slots": num_wrist_slots,
        }

    def _canonicalize_role_features(
        self,
        z_t: torch.Tensor,
        z_fwd: torch.Tensor,
        role: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert one camera role to ``[B,V,C,T,H,W]``.

        Wan caches already use the channel-first video layout. DINO caches and
        online features use patch-token layouts ``[B,V,N,D]`` and
        ``[B,V,T,N,D]``; those are reshaped without pooling or projection.
        """
        if z_t is None or z_fwd is None:
            return z_t, z_fwd
        expected_future = int(self.default_num_future_frames or 0)
        if self.visual_encoder_type == "wan_vae":
            if z_t.ndim != 6 or z_fwd.ndim != 6:
                raise ValueError(
                    f"Direct Wan {role} features must be [B,V,C,1,H,W] and "
                    f"[B,V,C,T,H,W], got {tuple(z_t.shape)} and {tuple(z_fwd.shape)}."
                )
            if z_t.shape[3] != 1:
                raise ValueError(
                    f"Direct Wan {role} current features must contain one frame, "
                    f"got T={z_t.shape[3]}."
                )
            return z_t, z_fwd

        if z_t.ndim != 4 or z_fwd.ndim != 5:
            raise ValueError(
                f"Direct DINOv3 {role} features must be [B,V,N,D] and "
                f"[B,V,T,N,D], got {tuple(z_t.shape)} and {tuple(z_fwd.shape)}."
            )
        if z_t.shape[:2] != z_fwd.shape[:2]:
            raise ValueError(
                f"Direct DINOv3 {role} current/future batch-slot shapes differ: "
                f"{tuple(z_t.shape[:2])} vs {tuple(z_fwd.shape[:2])}."
            )
        num_patches, feature_dim = int(z_t.shape[2]), int(z_t.shape[3])
        if z_fwd.shape[3:] != z_t.shape[2:]:
            raise ValueError(
                f"Direct DINOv3 {role} future patch shape must be "
                f"{tuple(z_t.shape[2:])}, got {tuple(z_fwd.shape[3:])}."
            )
        patch_side = int(num_patches**0.5)
        if patch_side * patch_side != num_patches:
            raise ValueError(
                f"Direct DINOv3 requires a square patch grid, got N={num_patches}."
            )
        input_size = int(self.visual_encoder_metadata.get("input_size", 224))
        backbone_patch = int(self.visual_encoder_metadata.get("patch_size", 16))
        expected_side = input_size // backbone_patch
        if patch_side != expected_side:
            raise ValueError(
                "Direct DINOv3 patch grid does not match the configured backbone: "
                f"expected {expected_side}x{expected_side} from input_size={input_size} and "
                f"patch_size={backbone_patch}, got {patch_side}x{patch_side}."
            )
        expected_dim = int(self.video_dit.in_channels)
        if feature_dim != expected_dim or int(self.video_dit.out_channels) != expected_dim:
            raise ValueError(
                "Direct DINOv3 feature channels must match VideoDiT input/output channels: "
                f"features={feature_dim}, input={self.video_dit.in_channels}, "
                f"output={self.video_dit.out_channels}."
            )
        actual_future = int(z_fwd.shape[2])
        if expected_future and actual_future != expected_future:
            raise ValueError(
                f"Direct DINOv3 expects all {expected_future} native future feature frames, "
                f"got {actual_future} for {role}."
            )
        z_t_grid = z_t.reshape(
            *z_t.shape[:2], patch_side, patch_side, feature_dim,
        ).permute(0, 1, 4, 2, 3).unsqueeze(3).contiguous()
        z_fwd_grid = z_fwd.reshape(
            *z_fwd.shape[:3], patch_side, patch_side, feature_dim,
        ).permute(0, 1, 5, 2, 3, 4).contiguous()
        return z_t_grid, z_fwd_grid

    def _pre_video_slots(
        self,
        z_video: torch.Tensor,
        timestep: torch.Tensor,
        text_ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        slot_ids: tuple[int, ...],
    ) -> dict:
        if z_video.ndim != 6:
            raise ValueError(f"`z_video` must be [B,V,C,T,H,W], got {tuple(z_video.shape)}")
        batch_size, num_slots = z_video.shape[:2]
        if len(slot_ids) != num_slots:
            raise ValueError(
                f"slot_ids must contain one ID per camera slot ({num_slots}), got {slot_ids}."
            )
        flat_video = z_video.reshape(batch_size * num_slots, *z_video.shape[2:])
        timestep_flat = timestep.repeat_interleave(num_slots)
        context_flat = text_ctx.repeat_interleave(num_slots, dim=0)
        mask_flat = ctx_mask.repeat_interleave(num_slots, dim=0)
        flat_pre = self.video_dit.pre_dit(flat_video, timestep_flat, context_flat, mask_flat)
        seq_len = flat_pre["tokens"].shape[1]
        grouped_context_mask = flat_pre["context_mask"].reshape(
            batch_size, num_slots * seq_len, ctx_mask.shape[1],
        )
        meta = flat_pre["meta"]
        grid_width = max(
            int(meta["grid_size"][2]), int(meta.get("current_grid_size", meta["grid_size"])[2]),
        )
        if grid_width > self.camera_rope_slot_stride:
            raise ValueError(
                "Camera grid width exceeds the configured tiled-RoPE stride: "
                f"video_width={grid_width}, stride={self.camera_rope_slot_stride}."
            )
        freqs = torch.cat([
            self.video_dit.rope_freqs(
                meta,
                z_video.device,
                w_offset=slot_id * self.camera_rope_slot_stride,
            )
            for slot_id in slot_ids
        ], dim=0)
        return {
            "flat": flat_pre,
            "tokens": flat_pre["tokens"].reshape(batch_size, num_slots * seq_len, -1),
            "freqs": freqs,
            "t_mod": flat_pre["t_mod"].reshape(batch_size, num_slots * seq_len, 6, -1),
            "context": text_ctx,
            "context_mask": grouped_context_mask,
            "tokens_per_slot": seq_len,
            "tokens_per_frame": int(meta["tokens_per_frame"]),
            "current_tokens_per_slot": int(meta["num_current_tokens"]),
            "num_slots": num_slots,
            "slot_ids": slot_ids,
        }

    def _post_video_slots(self, tokens: torch.Tensor, pre: dict) -> torch.Tensor:
        batch_size = tokens.shape[0]
        num_slots = int(pre["num_slots"])
        seq_len = int(pre["tokens_per_slot"])
        flat_tokens = tokens.reshape(batch_size * num_slots, seq_len, -1)
        flat_video = self.video_dit.post_dit(flat_tokens, pre["flat"])
        return flat_video.reshape(batch_size, num_slots, *flat_video.shape[1:])

    @staticmethod
    def _video_token_present(slot_present: torch.Tensor, tokens_per_slot: int) -> torch.Tensor:
        return slot_present.unsqueeze(-1).expand(-1, -1, tokens_per_slot).reshape(
            slot_present.shape[0], -1,
        )

    def _build_video_action_mask(
        self,
        num_slots: int,
        tokens_per_slot: int,
        tokens_per_frame: int,
        action_seq_len: int,
        device: torch.device,
        current_tokens_per_slot: Optional[int] = None,
    ) -> torch.Tensor:
        video_seq_len = num_slots * tokens_per_slot
        total = video_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        num_current = tokens_per_frame if current_tokens_per_slot is None else current_tokens_per_slot
        z0_indices, future_indices = [], []
        for slot in range(num_slots):
            start = slot * tokens_per_slot
            z0_indices.extend(range(start, start + num_current))
            future_indices.extend(range(start + num_current, start + tokens_per_slot))
        z0_idx = torch.tensor(z0_indices, dtype=torch.long, device=device)
        future_idx = torch.tensor(future_indices, dtype=torch.long, device=device)
        video_idx = torch.arange(video_seq_len, dtype=torch.long, device=device)
        action_idx = torch.arange(video_seq_len, total, dtype=torch.long, device=device)

        if z0_idx.numel() > 0:
            mask[z0_idx[:, None], z0_idx[None, :]] = True
        if future_idx.numel() > 0:
            mask[future_idx[:, None], video_idx[None, :]] = True
        mask[action_idx[:, None], action_idx[None, :]] = True

        if future_idx.numel() > 0:
            mask[future_idx[:, None], action_idx[None, :]] = True
        mask[action_idx[:, None], video_idx[None, :]] = True
        return mask

    def _compute_action_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        timestep_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_loss_per_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none",
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

    def _compute_video_loss(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        timestep_video: torch.Tensor,
        slot_present: torch.Tensor,
    ) -> torch.Tensor:
        pred_future = pred_video[:, :, :, 1:]
        target_future = target_video[:, :, :, 1:]
        loss_per_slot = F.mse_loss(
            pred_future.float(), target_future.float(), reduction="none",
        ).mean(dim=(2, 3, 4, 5))
        valid = slot_present.to(device=loss_per_slot.device, dtype=loss_per_slot.dtype)
        loss_per_sample = (loss_per_slot * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        weight = self.video_scheduler.training_weight(timestep_video).to(
            loss_per_sample.device,
            dtype=loss_per_sample.dtype,
        )
        return (loss_per_sample * weight).mean()

    def _run_video_action_mot(
        self,
        video_pre: dict,
        action_pre: dict,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )

    def _predict_video_action(
        self,
        z_video: torch.Tensor,
        actions: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        text_ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        slot_present: torch.Tensor,
        slot_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self._pre_video_slots(
            z_video, timestep_video, text_ctx, ctx_mask, slot_ids,
        )
        action_pre = self.action_dit.pre_dit(actions, timestep_action, text_ctx, ctx_mask)
        base_mask = self._build_video_action_mask(
            num_slots=video_pre["num_slots"],
            tokens_per_slot=video_pre["tokens_per_slot"],
            tokens_per_frame=video_pre["tokens_per_frame"],
            action_seq_len=action_pre["meta"]["seq_len"],
            device=self.device,
            current_tokens_per_slot=video_pre["current_tokens_per_slot"],
        )
        video_present = self._video_token_present(slot_present, video_pre["tokens_per_slot"])
        action_present = torch.ones(
            (actions.shape[0], action_pre["meta"]["seq_len"]),
            dtype=torch.bool,
            device=self.device,
        )
        mask = self._apply_token_presence_to_mask(
            base_mask, torch.cat([video_present, action_present], dim=1),
        )
        out = self._run_video_action_mot(video_pre, action_pre, mask)
        return self._post_video_slots(out["video"], video_pre), self.action_dit.post_dit(
            out["action"], action_pre,
        )

    def training_loss(
        self,
        actions: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
        z_t_main: Optional[torch.Tensor] = None,
        z_fwd_main: Optional[torch.Tensor] = None,
        main_present: Optional[torch.Tensor] = None,
        z_t_wrist: Optional[torch.Tensor] = None,
        z_fwd_wrist: Optional[torch.Tensor] = None,
        wrist_present: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if actions is None or text_embeds is None or text_mask is None:
            raise ValueError("`actions`, `text_embeds`, and `text_mask` are required.")
        missing = [
            name for name, value in {
                "z_t_main": z_t_main,
                "z_fwd_main": z_fwd_main,
                "main_present": main_present,
                "z_t_wrist": z_t_wrist,
                "z_fwd_wrist": z_fwd_wrist,
                "wrist_present": wrist_present,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"Direct-feature MiniWAM is multiview-only; missing tensors: {missing}")
        mv = self._compose_slots(
            z_t_main, z_fwd_main, main_present, z_t_wrist, z_fwd_wrist, wrist_present,
        )
        z_video = mv["z_video"]
        slot_present = mv["slot_present"].to(device=self.device)
        slot_ids = mv["slot_ids"]
        batch_size = z_video.shape[0]
        text_ctx, ctx_mask = self.context_encoder.encode_context(
            text_embeds, text_mask, proprio,
        )

        noise_video = torch.randn_like(z_video)
        timestep_video = self.video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=z_video.dtype,
        )
        noisy_video = self.video_scheduler.add_noise(z_video, noise_video, timestep_video)
        noisy_video[:, :, :, 0:1] = z_video[:, :, :, 0:1]
        target_video = self.video_scheduler.training_target(z_video, noise_video, timestep_video)

        noise_action = torch.randn_like(actions)
        timestep_action = self.action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=actions.dtype,
        )
        noisy_action = self.action_scheduler.add_noise(actions, noise_action, timestep_action)
        target_action = self.action_scheduler.training_target(actions, noise_action, timestep_action)
        if not self.independent_timesteps:
            timestep_video = timestep_action.to(dtype=z_video.dtype)
            noisy_video = self.video_scheduler.add_noise(z_video, noise_video, timestep_video)
            noisy_video[:, :, :, 0:1] = z_video[:, :, :, 0:1]
            target_video = self.video_scheduler.training_target(z_video, noise_video, timestep_video)

        pred_video, pred_action = self._predict_video_action(
            z_video=noisy_video,
            actions=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            text_ctx=text_ctx,
            ctx_mask=ctx_mask,
            slot_present=slot_present,
            slot_ids=slot_ids,
        )
        loss_video = self._compute_video_loss(pred_video, target_video, timestep_video, slot_present)
        loss_action = self._compute_action_loss(
            pred_action, target_action, timestep_action, action_is_pad,
        )
        loss_total = self.lambda_video * loss_video + self.lambda_action * loss_action
        self._global_step += 1
        return loss_total, {
            "loss_video": float(loss_video.detach().item()),
            "loss_action": float(loss_action.detach().item()),
            "loss_total": float(loss_total.detach().item()),
        }

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
        return_video: bool = False,
    ):
        self.eval()
        if any(x is None for x in (z_t_main, main_present, z_t_wrist, wrist_present)):
            raise ValueError("Direct-feature inference requires multiview z_t/present tensors.")
        batch_size = z_t_main.shape[0]
        if self.default_num_future_frames is None:
            raise ValueError(
                "Direct-feature inference requires the native future feature horizon "
                "derived from data.future_rgb_horizon by the shared runtime protocol."
            )
        num_future = int(self.default_num_future_frames)
        if self.visual_encoder_type == "dinov3":
            z_fwd_main = torch.zeros(
                (*z_t_main.shape[:2], num_future, *z_t_main.shape[2:]),
                device=z_t_main.device,
                dtype=z_t_main.dtype,
            )
            z_fwd_wrist = torch.zeros(
                (*z_t_wrist.shape[:2], num_future, *z_t_wrist.shape[2:]),
                device=z_t_wrist.device,
                dtype=z_t_wrist.dtype,
            )
        else:
            z_fwd_main = torch.zeros(
                (*z_t_main.shape[:3], num_future, *z_t_main.shape[4:]),
                device=z_t_main.device,
                dtype=z_t_main.dtype,
            )
            z_fwd_wrist = torch.zeros(
                (*z_t_wrist.shape[:3], num_future, *z_t_wrist.shape[4:]),
                device=z_t_wrist.device,
                dtype=z_t_wrist.dtype,
            )
        mv = self._compose_slots(
            z_t_main, z_fwd_main, main_present, z_t_wrist, z_fwd_wrist, wrist_present,
        )
        z_t_slots = mv["z_t_slots"]
        slot_present = mv["slot_present"].to(device=self.device)
        slot_ids = mv["slot_ids"]
        num_main = mv["num_main_slots"]
        num_wrist = mv["num_wrist_slots"]
        _, num_slots, channels, _, height, width = z_t_slots.shape
        text_ctx, ctx_mask = self.context_encoder.encode_context(
            text_embeds, text_mask, proprio,
        )

        gen = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_dit.action_dim),
            generator=gen,
            device="cpu",
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.dtype)
        latents_video = torch.randn(
            (batch_size, num_slots, channels, 1 + num_future, height, width),
            generator=gen,
            device="cpu",
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.dtype)
        latents_video[:, :, :, 0:1] = z_t_slots

        action_ts, action_ds = self.action_scheduler.build_inference_schedule(
            num_inference_steps, self.device, latents_action.dtype,
        )

        video_ts, video_ds = self.video_scheduler.build_inference_schedule(
            num_inference_steps, self.device, latents_video.dtype,
        )
        for i in range(num_inference_steps):
            timestep_video = video_ts[i].expand(batch_size)
            timestep_action = action_ts[i].expand(batch_size)
            pred_video, pred_action = self._predict_video_action(
                latents_video,
                latents_action,
                timestep_video,
                timestep_action,
                text_ctx,
                ctx_mask,
                slot_present,
                slot_ids,
            )
            latents_video = self.video_scheduler.step(pred_video, video_ds[i], latents_video)
            latents_video[:, :, :, 0:1] = z_t_slots
            latents_action = self.action_scheduler.step(pred_action, action_ds[i], latents_action)

        actions_cpu = latents_action.detach().cpu().float()
        if not return_video:
            return actions_cpu
        z_fwd_pred = latents_video[:, :, :, 1:].detach().cpu().float()
        z_fwd_main_pred = z_fwd_pred[:, :num_main]
        z_fwd_wrist_pred = z_fwd_pred[:, num_main:num_main + num_wrist]
        return {
            "actions": actions_cpu,
            "z_fwd_main_pred": z_fwd_main_pred,
            "z_fwd_wrist_pred": z_fwd_wrist_pred,
        }

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        default_num_future_frames: int,
    ) -> "MiniWAMNativeFeatureModel":
        mot_cfg = cfg.get("mot", {})
        mv_cfg = cfg.get("multi_view_latent_encoder", {})
        num_heads = mot_cfg.get("num_heads", 12)
        attn_head_dim = mot_cfg.get("attn_head_dim", 128)
        num_layers = mot_cfg.get("num_layers", 12)

        vdit_cfg = cfg.get("video_dit", {})
        frame0_patch_size = vdit_cfg.get("frame0_patch_size")
        if frame0_patch_size is not None and str(
            cfg.get("visual_encoder", {}).get("type", "wan_vae")
        ).lower() != "dinov3":
            raise ValueError(
                "video_dit.frame0_patch_size reproduces the PRISM model's per-patch DINOv3 "
                "current-frame tokens; Wan-VAE native and PRISM arms already share the "
                "2x2 current-frame patching."
            )
        video_dit = VideoDiT(
            hidden_dim=vdit_cfg.get("hidden_dim", 1536),
            in_channels=vdit_cfg.get("in_channels", 16),
            out_channels=vdit_cfg.get("out_channels", 16),
            ffn_dim=vdit_cfg.get("ffn_dim", 6144),
            text_dim=vdit_cfg.get("text_dim", 4096),
            freq_dim=vdit_cfg.get("freq_dim", 256),
            eps=vdit_cfg.get("eps", 1e-6),
            patch_size=tuple(vdit_cfg.get("patch_size", [1, 2, 2])),
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            num_layers=num_layers,
            video_attention_mask_mode=vdit_cfg.get("video_attention_mask_mode", "first_frame_causal"),
            use_gradient_checkpointing=vdit_cfg.get("use_gradient_checkpointing", True),
            rope_table_size=vdit_cfg.get("rope_table_size", 1024),
            frame0_patch_size=frame0_patch_size,
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
            mixtures={"video": video_dit, "action": action_dit},
            mot_checkpoint_mixed_attn=mot_cfg.get("checkpoint_mixed_attn", True),
        )
        vs_cfg = cfg.get("video_scheduler", {})
        video_scheduler = FlowMatchScheduler(
            num_train_timesteps=vs_cfg.get("num_train_timesteps", 1000),
            shift=vs_cfg.get("train_shift", 5.0),
        )
        as_cfg = cfg.get("action_scheduler", {})
        action_scheduler = FlowMatchScheduler(
            num_train_timesteps=as_cfg.get("num_train_timesteps", 1000),
            shift=as_cfg.get("train_shift", 5.0),
        )
        loss_cfg = cfg.get("loss", {})
        visual_cfg = dict(cfg.get("visual_encoder", {}))
        visual_type = str(visual_cfg.get("type", "wan_vae")).lower()
        return cls(
            video_dit=video_dit,
            action_dit=action_dit,
            context_encoder=context_encoder,
            mot=mot,
            video_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
            independent_timesteps=cfg.get("independent_timesteps", True),
            lambda_video=loss_cfg.get("lambda_video", 1.0),
            lambda_action=loss_cfg.get("lambda_action", 1.0),
            default_num_future_frames=int(default_num_future_frames),
            max_main_slots=mv_cfg.get("max_main_slots", 2),
            max_wrist_slots=mv_cfg.get("max_wrist_slots", 1),
            camera_rope_slot_stride=mv_cfg.get("camera_rope_slot_stride", 32),
            visual_encoder_type=visual_type,
            visual_encoder_metadata=visual_cfg,
        )
