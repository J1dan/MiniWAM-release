"""Resolve a MiniWAM training config into the settings every consumer shares."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

ARCHITECTURES = ("miniwam", "native")
VISUAL_ENCODERS = ("wan_vae", "dinov3")
STAGE1_OBJECTIVES = ("inverse_dynamics", "reconstruction_only")
POLICY_INITS = ("stage1", "fresh")


@dataclass(frozen=True)
class MiniWAMRuntimeProtocol:
    """Resolved values shared by extraction, data loading, models, and inference."""

    future_rgb_horizon: int
    temporal_downsample_factor: int
    visual_encoder_type: str
    action_horizon: int

    @property
    def future_feature_horizon(self) -> int:
        return self.future_rgb_horizon // self.temporal_downsample_factor

    @property
    def native_future_feature_horizon(self) -> int:
        """Number of native future features predicted by the native-feature model.

        Wan-VAE compresses time by four during feature extraction; DINOv3 stores
        one patch grid for every RGB frame.
        """
        return (
            self.future_rgb_horizon
            if self.visual_encoder_type == "dinov3"
            else self.future_feature_horizon
        )



def _resolve_reconstruction(model_cfg: dict, stage: int) -> None:
    rec_cfg = model_cfg.setdefault("reconstruction", {})
    rec_cfg.setdefault("hidden_dim", 64)
    if stage == 2:
        # The feature decoder only serves the Stage-1 reconstruction loss.
        rec_cfg["decoder_enabled"] = False
        rec_cfg["lambda_rec"] = 0.0
        return
    rec_cfg.setdefault("decoder_enabled", False)
    rec_cfg.setdefault("lambda_rec", 0.0)
    if float(rec_cfg["lambda_rec"]) < 0:
        raise ValueError("model.reconstruction.lambda_rec must be non-negative.")
    if float(rec_cfg["lambda_rec"]) > 0 and not bool(rec_cfg["decoder_enabled"]):
        raise ValueError(
            "model.reconstruction.lambda_rec > 0 requires reconstruction.decoder_enabled=true."
        )


def _resolve_stage1_objective(resolved: dict, stage: int) -> None:
    model_cfg = resolved["model"]
    objective = str(model_cfg.get("stage1_objective", "inverse_dynamics")).lower()
    if objective not in STAGE1_OBJECTIVES:
        raise ValueError(
            f"model.stage1_objective must be one of {STAGE1_OBJECTIVES}, got {objective!r}."
        )
    model_cfg["stage1_objective"] = objective
    if stage == 1 and objective == "reconstruction_only":
        rec_cfg = model_cfg["reconstruction"]
        loss_cfg = resolved.setdefault("loss", {})
        if not bool(rec_cfg["decoder_enabled"]) or float(rec_cfg["lambda_rec"]) <= 0:
            raise ValueError(
                "reconstruction_only Stage 1 requires reconstruction.decoder_enabled=true "
                "and lambda_rec > 0."
            )
        if float(loss_cfg.get("lambda_action", 0.0)) != 0.0:
            raise ValueError("reconstruction_only Stage 1 requires loss.lambda_action=0.0.")
        if float(loss_cfg.get("lambda_latent", 0.0)) != 0.0:
            raise ValueError("reconstruction_only Stage 1 requires loss.lambda_latent=0.0.")


def _resolve_checkpoint(resolved: dict, stage: int, architecture: str) -> None:
    checkpoint_cfg = resolved.setdefault("checkpoint", {})
    if stage == 2 and architecture == "miniwam":
        policy_init = str(checkpoint_cfg.setdefault("policy_init", "fresh")).lower()
        if policy_init not in POLICY_INITS:
            raise ValueError(
                f"checkpoint.policy_init must be one of {POLICY_INITS}, got {policy_init!r}."
            )
        checkpoint_cfg["policy_init"] = policy_init
    elif checkpoint_cfg.get("stage1_path") or checkpoint_cfg.get("policy_init"):
        raise ValueError(
            "checkpoint.stage1_path and checkpoint.policy_init apply only to Stage-2 MiniWAM runs."
        )


def _resolve_dino_features(visual_cfg: dict) -> None:
    visual_cfg.pop("feature_grid", None)  # Wan-VAE-only field inherited from the base config.
    visual_cfg.setdefault("compute_dtype", "float32")
    feature_cfg = visual_cfg.setdefault("feature", {})
    feature_cfg.setdefault("type", "patch_tokens")
    feature_cfg.setdefault("normalization", "final_layernorm")
    feature_cfg.setdefault("l2_normalize", False)
    feature_cfg.setdefault("include_cls", False)
    feature_cfg.setdefault("include_registers", False)
    feature_cfg.setdefault("output_dtype", "bfloat16")
    feature_cfg.setdefault("layers", {}).setdefault("mode", "final")
    cache_cfg = visual_cfg.setdefault("cache", {})
    cache_cfg.setdefault("per_frame", True)
    cache_cfg.setdefault("storage_dtype", "bfloat16")
    if (
        visual_cfg["compute_dtype"] != "float32"
        or feature_cfg["output_dtype"] != "bfloat16"
        or cache_cfg["storage_dtype"] != "bfloat16"
    ):
        raise ValueError(
            "DINOv3 requires FP32 backbone compute followed by a BF16 MiniWAM feature "
            "boundary and BF16 cache storage."
        )


def resolve_miniwam_protocol(config: Mapping[str, Any]) -> tuple[dict, MiniWAMRuntimeProtocol]:
    """Validate a training config and fill in defaults; returns a deep copy."""
    if not isinstance(config, Mapping):
        raise TypeError(f"MiniWAM configuration must be a mapping, got {type(config).__name__}.")
    resolved = deepcopy(dict(config))
    data_cfg = resolved.setdefault("data", {})
    model_cfg = resolved.setdefault("model", {})

    stage = int(resolved.get("stage", 1))
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}.")
    resolved["stage"] = stage
    model_cfg["stage"] = stage
    architecture = str(model_cfg.get("architecture", "miniwam")).lower()
    if architecture not in ARCHITECTURES:
        raise ValueError(f"model.architecture must be one of {ARCHITECTURES}, got {architecture!r}.")
    if architecture == "native" and stage != 2:
        raise ValueError("The native-feature model has no Stage 1; set stage: 2.")
    model_cfg["architecture"] = architecture

    visual_cfg = model_cfg.setdefault("visual_encoder", {})
    visual_type = str(visual_cfg.get("type", "wan_vae")).lower()
    if visual_type not in VISUAL_ENCODERS:
        raise ValueError(
            f"model.visual_encoder.type must be one of {VISUAL_ENCODERS}, got {visual_type!r}."
        )
    visual_cfg["type"] = visual_type
    future_cfg = visual_cfg.setdefault("future", {})
    temporal_factor = int(future_cfg.get("temporal_downsample_factor", 4))
    if temporal_factor != 4:
        raise ValueError(
            "MiniWAM requires visual_encoder.future.temporal_downsample_factor=4, got "
            f"{temporal_factor}."
        )
    future_cfg["temporal_downsample_factor"] = temporal_factor

    horizon = int(data_cfg.get("future_rgb_horizon", 16))
    if horizon <= 0 or horizon % temporal_factor:
        raise ValueError(
            "data.future_rgb_horizon must be positive and divisible by "
            f"temporal_downsample_factor={temporal_factor}; got {horizon}."
        )
    data_cfg["future_rgb_horizon"] = horizon
    action_horizon = int(data_cfg.get("action_horizon", 16))
    if action_horizon <= 0:
        raise ValueError("data.action_horizon must be positive.")
    data_cfg["action_horizon"] = action_horizon

    _resolve_reconstruction(model_cfg, stage)
    if visual_type == "dinov3":
        _resolve_dino_features(visual_cfg)
    _resolve_stage1_objective(resolved, stage)
    _resolve_checkpoint(resolved, stage, architecture)

    runtime = MiniWAMRuntimeProtocol(
        future_rgb_horizon=horizon,
        temporal_downsample_factor=temporal_factor,
        visual_encoder_type=visual_type,
        action_horizon=action_horizon,
    )
    return resolved, runtime
