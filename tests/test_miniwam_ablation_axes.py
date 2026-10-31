import json
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from miniwam.checkpointing import (
    build_model_protocol,
    load_stage1_encoders,
    load_stage1_for_stage2,
    validate_model_protocol,
)
from miniwam.factory import build_miniwam_model
from miniwam.dataset import MiniWAMPreextractedDataset
from miniwam.models.dinov3 import DINOv3FrameEncoder
from miniwam.models.video_dit import VideoDiT
from miniwam.native_feature_model import MiniWAMNativeFeatureModel
from miniwam.protocol import resolve_miniwam_protocol
from scripts.data.extraction_utils import encode_dinov3_frames, validate_multiview_episode


torch.set_num_threads(1)


def tiny_cfg(visual_type="wan_vae", horizon=16, latent_dim=8, grid=(None, 2, 2), reconstruction=False, stage=1):
    visual = {
        "type": visual_type,
        "future": {"temporal_downsample_factor": 4},
    }
    stages = [(16, (3, 3, 3), (2, 1, 1), (1, 1, 1))]
    if visual_type == "dinov3":
        visual.update({
            "architecture": "vitb16", "pretraining": "lvd1689m",
            "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "revision": "synthetic-revision", "resolved_revision": "synthetic-revision",
            "input_size": 224, "patch_size": 16,
            "feature": {
                "type": "patch_tokens", "normalization": "final_layernorm",
                "l2_normalize": False, "layers": {"mode": "final"},
            },
            "adapter": {"input_dim": 768, "hidden_dim": 8},
        })
    else:
        visual["feature_grid"] = [8, 8]
    return {"stage": stage, "data": {"future_rgb_horizon": horizon, "action_horizon": 3}, "model": {
        "stage": stage, "visual_encoder": visual,
        "latent_encoder": {
            "in_channels": 16, "latent_dim": latent_dim,
            "compression_stages": stages, "use_kl": False,
            "bottleneck": {"type": "adaptive_avg_pool3d", "target_grid": list(grid)},
        },
        "multi_view_latent_encoder": {"max_main_slots": 2, "max_wrist_slots": 1},
        "mot": {"num_heads": 2, "attn_head_dim": 16, "num_layers": 1, "checkpoint_mixed_attn": False},
        "world_dit": {"hidden_dim": 32, "ffn_dim": 64, "text_dim": 16, "freq_dim": 16,
                       "frame0_patch_size": [2, 2], "use_gradient_checkpointing": False},
        "action_dit": {"hidden_dim": 32, "ffn_dim": 64, "text_dim": 16, "freq_dim": 16,
                       "action_dim": 7, "use_gradient_checkpointing": False},
        "context_encoder": {"text_dim": 16, "hidden_dim": 32},
        "loss": {"lambda_action": 1.0, "lambda_latent": 1.0 if stage == 2 else 0.0,
                 "kl_beta": 1e-4, "kl_warmup_steps": 0},
        "reconstruction": {
            "decoder_enabled": reconstruction,
            "lambda_rec": 0.1 if reconstruction else 0.0,
            "hidden_dim": 8,
        },
    }}


def build_tiny(cfg):
    resolved, runtime = resolve_miniwam_protocol(cfg)
    return build_miniwam_model(resolved["model"], runtime), runtime


def direct_dino_tiny_cfg():
    cfg = tiny_cfg("dinov3", horizon=16, stage=2)
    model = cfg["model"]
    model["architecture"] = "native"
    model["video_dit"] = {
        "hidden_dim": 32,
        "ffn_dim": 64,
        "text_dim": 16,
        "freq_dim": 16,
        "in_channels": 768,
        "out_channels": 768,
        "patch_size": [1, 2, 2],
        "use_gradient_checkpointing": False,
    }
    model["loss"] = {"lambda_action": 1.0, "lambda_video": 1.0}
    return cfg


def batch_for(cfg, batch=2):
    visual = cfg["model"]["visual_encoder"]
    horizon = cfg["data"]["future_rgb_horizon"]
    common = {
        "actions": torch.randn(batch, 3, 7),
        "action_is_pad": torch.zeros(batch, 3, dtype=torch.bool),
        "text_embeds": torch.randn(batch, 5, 16),
        "text_mask": torch.ones(batch, 5, dtype=torch.bool),
        "main_present": torch.ones(batch, 1, dtype=torch.bool),
        "wrist_present": torch.ones(batch, 1, dtype=torch.bool),
    }
    if visual["type"] == "dinov3":
        common.update({
            "z_t_main": torch.randn(batch, 1, 196, 768),
            "z_fwd_main": torch.randn(batch, 1, horizon, 196, 768, requires_grad=True),
            "z_t_wrist": torch.randn(batch, 1, 196, 768),
            "z_fwd_wrist": torch.randn(batch, 1, horizon, 196, 768, requires_grad=True),
        })
    else:
        feature_t = horizon // 4
        common.update({
            "z_t_main": torch.randn(batch, 1, 16, 1, 8, 8),
            "z_fwd_main": torch.randn(batch, 1, 16, feature_t, 8, 8),
            "z_t_wrist": torch.randn(batch, 1, 16, 1, 8, 8),
            "z_fwd_wrist": torch.randn(batch, 1, 16, feature_t, 8, 8),
        })
    return common


def test_vae_horizons_latent_dims_and_grids_are_runtime_derived():
    for horizon, expected_t in ((16, 3), (24, 4), (32, 5)):
        for latent_dim in (16, 32):
            cfg = tiny_cfg(horizon=horizon, latent_dim=latent_dim, grid=(None, 2, 2))
            model, _ = build_tiny(cfg)
            assert model.get_latent_grid_shape() == (expected_t, 2, 2)
            encoded = model._prepare_multiview_inputs(**{
                k: v for k, v in batch_for(cfg, batch=1).items()
                if k in {"z_t_main", "z_fwd_main", "main_present", "z_t_wrist", "z_fwd_wrist", "wrist_present"}
            })
            assert encoded["L_clean"].shape == (1, 2 * expected_t * 4, latent_dim)


def test_spatial_bottleneck_token_variants_and_protocol_validation():
    tokens = []
    for side in (4, 2, 1):
        cfg = tiny_cfg(grid=(None, side, side))
        model, _ = build_tiny(cfg)
        shape = model.get_latent_grid_shape()
        tokens.append(shape[0] * shape[1] * shape[2])
    assert tokens == [48, 12, 3]
    model, runtime = build_tiny(tiny_cfg(grid=(None, 2, 2)))
    protocol = build_model_protocol(model, runtime)
    assert protocol["latent_dim"] == 8
    assert protocol["action_dim"] == 7
    assert protocol["proprio_dim"] is None
    assert protocol["latent_grid_shape"] == [3, 2, 2]
    assert protocol["latent_bottleneck_type"] == "adaptive_avg_pool3d"
    with pytest.raises(ValueError, match="latent_dim: checkpoint=8, config=16"):
        validate_model_protocol(protocol, dict(protocol, latent_dim=16), "test")
    with pytest.raises(ValueError, match="action_dim"):
        validate_model_protocol(protocol, dict(protocol, action_dim=14), "test")


def test_dino_all_frames_stage1_stage2_and_independent_inference_horizon():
    for horizon, expected_t in ((16, 3), (24, 4), (32, 5)):
        cfg = tiny_cfg("dinov3", horizon=horizon)
        model, _ = build_tiny(cfg)
        batch = batch_for(cfg)
        loss, _ = model(**batch)
        loss.backward()
        assert model.get_latent_grid_shape() == (expected_t, 2, 2)
        # Learned temporal compression must receive signal from every RGB frame.
        per_frame_grad = batch["z_fwd_main"].grad.abs().sum(dim=(0, 1, 3, 4))
        assert torch.all(per_frame_grad > 0)

    cfg = tiny_cfg("dinov3", horizon=16, stage=2)
    model, _ = build_tiny(cfg)
    model.set_stage(2)
    batch = batch_for(cfg)
    loss, stats = model(**batch)
    loss.backward()
    assert "loss_latent" in stats
    model.eval()
    for action_horizon in (24, 32):
        actions = model.infer(
            batch["z_t_main"][:1], batch["main_present"][:1],
            batch["z_t_wrist"][:1], batch["wrist_present"][:1],
            batch["text_embeds"][:1], batch["text_mask"][:1],
            action_horizon=action_horizon, num_inference_steps=1,
        )
        assert actions.shape == (1, action_horizon, 7)


@pytest.mark.parametrize("native", [False, True])
def test_dino_inference_accepts_bf16_features_at_any_model_dtype(native):
    # The DINOv3 encoder hands over BF16 features; FP32 and BF16 models both accept them.
    cfg = direct_dino_tiny_cfg() if native else tiny_cfg("dinov3", horizon=16, stage=2)
    model, _ = build_tiny(cfg)
    if not native:
        model.set_stage(2)
    batch = batch_for(cfg, batch=1)
    for dtype in (torch.float32, torch.bfloat16):
        model.to(dtype)
        actions = model.infer(
            batch["z_t_main"].to(torch.bfloat16), batch["main_present"],
            batch["z_t_wrist"].to(torch.bfloat16), batch["wrist_present"],
            batch["text_embeds"].to(dtype), batch["text_mask"],
            action_horizon=3, num_inference_steps=1, seed=0,
        )
        assert actions.shape == (1, 3, 7)
        assert torch.isfinite(actions).all()


def test_direct_dino_predicts_all_native_future_features_without_ephi():
    cfg = direct_dino_tiny_cfg()
    model, runtime = build_tiny(cfg)
    assert runtime.native_future_feature_horizon == 16
    assert model.model_architecture == "native"
    assert model.visual_encoder_type == "dinov3"
    assert model.default_num_future_frames == 16
    assert not hasattr(model, "latent_encoder")
    assert not hasattr(model, "reconstruction_decoder")

    batch = batch_for(cfg, batch=1)
    # Exercise a genuinely absent wrist role while retaining its canonical
    # token layout, as real padded multiview batches do.
    batch["z_t_wrist"] = torch.empty(1, 0, 196, 768)
    batch["z_fwd_wrist"] = torch.empty(1, 0, 16, 196, 768)
    batch["wrist_present"] = torch.zeros(1, 0, dtype=torch.bool)
    loss, stats = model(**batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert {"loss_video", "loss_action", "loss_total"} <= set(stats)
    per_frame_grad = batch["z_fwd_main"].grad.abs().sum(dim=(0, 1, 3, 4))
    assert per_frame_grad.shape == (16,)
    assert torch.all(per_frame_grad > 0)

    model.eval()
    result = model.infer(
        z_t_main=batch["z_t_main"],
        main_present=batch["main_present"],
        z_t_wrist=batch["z_t_wrist"],
        wrist_present=batch["wrist_present"],
        text_embeds=batch["text_embeds"],
        text_mask=batch["text_mask"],
        action_horizon=3,
        num_inference_steps=1,
        seed=7,
        return_video=True,
    )
    assert result["actions"].shape == (1, 3, 7)
    assert result["z_fwd_main_pred"].shape == (1, 1, 768, 16, 14, 14)
    assert result["z_fwd_wrist_pred"].shape == (1, 0, 768, 16, 14, 14)


def per_patch_native_cfg():
    cfg = direct_dino_tiny_cfg()
    cfg["model"]["video_dit"]["frame0_patch_size"] = [1, 1]
    return cfg


def test_native_current_per_patch_tokens_match_the_prism_current_frame():
    native, _ = build_tiny(per_patch_native_cfg())
    default, _ = build_tiny(direct_dino_tiny_cfg())
    prism, _ = build_tiny(tiny_cfg("dinov3", horizon=16, stage=2))
    cpu = torch.device("cpu")
    video = torch.randn(1, 768, 17, 14, 14)
    context = torch.randn(1, 5, 32)
    pre = native.video_dit.pre_dit(video, torch.tensor([500.0]), context)
    meta = pre["meta"]
    # 196 current tokens from the same per-patch projection as the PRISM world expert.
    assert meta["num_current_tokens"] == 196 and meta["grid_size"] == (16, 7, 7)
    assert pre["tokens"].shape[1] == 196 + 16 * 49
    assert isinstance(native.video_dit.frame0_projection, nn.Linear)
    assert (native.video_dit.frame0_projection.weight.shape
            == prism.world_dit.frame0_projection.weight.shape)
    # Current tokens carry diffusion timestep 0 whatever the future timestep is.
    other = native.video_dit.pre_dit(video, torch.tensor([100.0]), context)
    assert torch.equal(pre["t"][:, :196], other["t"][:, :196])
    assert not torch.equal(pre["t"][:, 196:], other["t"][:, 196:])
    # Current tokens take PRISM's z0 positions; future tokens keep the default positions.
    default_meta = default.video_dit.pre_dit(video, torch.tensor([500.0]), context)["meta"]
    assert default_meta["num_current_tokens"] == 49
    for w_offset in (0, 32):
        freqs = native.video_dit.rope_freqs(meta, cpu, w_offset=w_offset)
        prism_z0 = prism.world_dit._build_z0_rope_freqs(
            196, cpu, z0_grid_shape=(14, 14), w_offset=w_offset,
        )
        assert torch.equal(freqs[:196], prism_z0)
        default_freqs = default.video_dit.rope_freqs(default_meta, cpu, w_offset=w_offset)
        assert torch.equal(freqs[196:], default_freqs[49:])


def test_native_current_per_patch_attention_mask_keeps_the_prism_roles():
    native, _ = build_tiny(per_patch_native_cfg())
    per_slot = 196 + 16 * 49
    mask = native._build_video_action_mask(
        num_slots=2, tokens_per_slot=per_slot, tokens_per_frame=49, action_seq_len=3,
        device=torch.device("cpu"), current_tokens_per_slot=196,
    )
    current = torch.zeros(2 * per_slot + 3, dtype=torch.bool)
    for slot in range(2):
        current[slot * per_slot:slot * per_slot + 196] = True
    # As in PRISM Stage 2: current tokens see only current tokens (all cameras);
    # target and action tokens see everything.
    assert mask[current][:, current].all()
    assert not mask[current][:, ~current].any()
    assert mask[~current].all()


def test_native_current_per_patch_only_adds_the_projection():
    torch.manual_seed(0)
    default, _ = build_tiny(direct_dino_tiny_cfg())
    torch.manual_seed(0)
    native, _ = build_tiny(per_patch_native_cfg())
    assert default.video_dit.frame0_projection is None
    before, after = default.state_dict(), native.state_dict()
    extra = set(after) - set(before)
    assert extra and set(before) <= set(after)
    assert all("frame0_projection" in key for key in extra)
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_native_current_per_patch_trains_and_infers():
    cfg = per_patch_native_cfg()
    model, _ = build_tiny(cfg)
    patched_frames = []
    handle = model.video_dit.patch_embedding.register_forward_hook(
        lambda module, inputs, output: patched_frames.append(inputs[0].shape[2]),
    )
    batch = batch_for(cfg, batch=2)
    loss, _ = model(**batch)
    loss.backward()
    handle.remove()
    assert torch.isfinite(loss)
    assert set(patched_frames) == {16}  # the current frame bypasses the 2x2 patching
    assert model.video_dit.frame0_projection.weight.grad.abs().sum() > 0
    assert torch.all(batch["z_fwd_main"].grad.abs().sum(dim=(0, 1, 3, 4)) > 0)

    model.eval()
    result = model.infer(
        z_t_main=batch["z_t_main"][:1], main_present=batch["main_present"][:1],
        z_t_wrist=batch["z_t_wrist"][:1], wrist_present=batch["wrist_present"][:1],
        text_embeds=batch["text_embeds"][:1], text_mask=batch["text_mask"][:1],
        action_horizon=3, num_inference_steps=2, seed=7, return_video=True,
    )
    assert result["actions"].shape == (1, 3, 7)
    assert torch.isfinite(result["actions"]).all()
    assert result["z_fwd_main_pred"].shape == (1, 1, 768, 16, 14, 14)


def test_native_current_per_patch_is_dinov3_only_and_one_token_per_patch():
    with pytest.raises(ValueError, match="frame0_patch_size"):
        MiniWAMNativeFeatureModel.from_config(
            {"visual_encoder": {"type": "wan_vae"}, "video_dit": {"frame0_patch_size": [1, 1]}},
            default_num_future_frames=4,
        )
    with pytest.raises(ValueError, match="frame0_patch_size"):
        VideoDiT(hidden_dim=32, in_channels=8, out_channels=8, ffn_dim=64, text_dim=16,
                 freq_dim=16, num_heads=2, attn_head_dim=16, num_layers=1,
                 frame0_patch_size=(2, 2))


def test_direct_dino_robotwin_three_camera_14d_train_and_infer_smoke():
    cfg = direct_dino_tiny_cfg()
    cfg["data"].update({"future_rgb_horizon": 4, "action_horizon": 3})
    model_cfg = cfg["model"]
    model_cfg["multi_view_latent_encoder"].update({
        "max_main_slots": 1,
        "max_wrist_slots": 2,
        "camera_rope_slot_stride": 32,
    })
    model_cfg["context_encoder"]["proprio_dim"] = 14
    model_cfg["action_dit"]["action_dim"] = 14
    model, runtime = build_tiny(cfg)
    assert runtime.native_future_feature_horizon == 4

    batch = {
        "actions": torch.randn(1, 3, 14),
        "action_is_pad": torch.zeros(1, 3, dtype=torch.bool),
        "proprio": torch.randn(1, 14),
        "text_embeds": torch.randn(1, 5, 16),
        "text_mask": torch.ones(1, 5, dtype=torch.bool),
        "z_t_main": torch.randn(1, 1, 196, 768),
        "z_fwd_main": torch.randn(1, 1, 4, 196, 768),
        "main_present": torch.ones(1, 1, dtype=torch.bool),
        "z_t_wrist": torch.randn(1, 2, 196, 768),
        "z_fwd_wrist": torch.randn(1, 2, 4, 196, 768),
        "wrist_present": torch.ones(1, 2, dtype=torch.bool),
    }
    loss, stats = model(**batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert {"loss_video", "loss_action", "loss_total"} <= set(stats)

    result = model.infer(
        z_t_main=batch["z_t_main"],
        main_present=batch["main_present"],
        z_t_wrist=batch["z_t_wrist"],
        wrist_present=batch["wrist_present"],
        text_embeds=batch["text_embeds"],
        text_mask=batch["text_mask"],
        proprio=batch["proprio"],
        action_horizon=3,
        num_inference_steps=1,
        seed=7,
        return_video=True,
    )
    assert result["actions"].shape == (1, 3, 14)
    assert result["z_fwd_main_pred"].shape == (1, 1, 768, 4, 14, 14)
    assert result["z_fwd_wrist_pred"].shape == (1, 2, 768, 4, 14, 14)


def test_direct_dino_protocol_is_distinct_from_standard_and_direct_vae():
    dino_model, dino_runtime = build_tiny(direct_dino_tiny_cfg())
    direct_protocol = build_model_protocol(dino_model, dino_runtime)
    assert direct_protocol["model_architecture"] == "native"
    assert direct_protocol["visual_encoder_type"] == "dinov3"

    standard_model, standard_runtime = build_tiny(tiny_cfg("dinov3", stage=2))
    with pytest.raises(ValueError, match="model_architecture"):
        validate_model_protocol(
            build_model_protocol(standard_model, standard_runtime),
            direct_protocol,
            "direct_dino_test",
        )


def test_reconstruction_disabled_is_state_dict_compatible_and_enabled_backprops():
    off, _ = build_tiny(tiny_cfg(reconstruction=False))
    assert not any("reconstruction_decoder" in key for key in off.state_dict())
    cfg = tiny_cfg(reconstruction=True)
    on, _ = build_tiny(cfg)
    batch = batch_for(cfg)
    loss, logs = on(**batch)
    loss.backward()
    assert logs["feature_reconstruction_loss"] > 0
    assert logs["weighted_feature_reconstruction_loss"] > 0
    assert logs["vae_reconstruction_loss"] == logs["feature_reconstruction_loss"]
    assert on.reconstruction_target_type == "wan_vae_features"
    assert on.reconstruction_out_channels == 16
    assert on.reconstruction_decoder.net[0].weight.grad is not None
    assert on.latent_encoder.mu_head.weight.grad is not None


def test_stage2_drops_the_stage1_reconstruction_decoder():
    stage1_cfg = tiny_cfg(reconstruction=True, stage=1)
    stage1, runtime1 = build_tiny(stage1_cfg)
    stage1_protocol = build_model_protocol(stage1, runtime1)

    stage2_cfg = tiny_cfg(reconstruction=True, stage=2)
    stage2, runtime2 = build_tiny(stage2_cfg)
    assert stage2.reconstruction_decoder is None
    assert stage2.wrist_reconstruction_decoder is None
    assert not stage2.reconstruction_active
    validate_model_protocol(
        stage1_protocol, build_model_protocol(stage2, runtime2), "stage1_to_stage2_test",
        stage1_to_stage2=True,
    )
    load_stage1_for_stage2(stage2, stage1.state_dict())
    stage2.set_stage(2)
    loss, logs = stage2(**batch_for(stage2_cfg))
    loss.backward()
    assert logs["feature_reconstruction_loss"] == 0.0


def test_dino_reconstructs_native_future_patch_features():
    cfg = tiny_cfg("dinov3", reconstruction=True)
    model, _ = build_tiny(cfg)
    batch = batch_for(cfg, batch=1)
    outputs = []
    handle = model.reconstruction_decoder.register_forward_hook(
        lambda _module, _inputs, output: outputs.append(tuple(output.shape))
    )
    try:
        loss, logs = model(**batch)
        loss.backward()
    finally:
        handle.remove()
    assert logs["feature_reconstruction_loss"] > 0
    assert logs["weighted_feature_reconstruction_loss"] > 0
    assert "vae_reconstruction_loss" not in logs
    assert outputs == [(1, 768, 16, 14, 14)]
    assert model.reconstruction_target_type == "dinov3_patch_features"
    assert model.reconstruction_out_channels == 768
    assert model.latent_encoder.temporal_compressor[0].weight.grad is not None
    assert model.reconstruction_decoder.net[0].weight.grad is not None


@pytest.mark.parametrize("horizon", [16, 24, 32])
def test_dino_reconstruction_only_bypasses_policy_and_uses_every_future_frame(
    monkeypatch, horizon,
):
    cfg = tiny_cfg("dinov3", horizon=horizon, reconstruction=True, stage=1)
    cfg["model"]["stage1_objective"] = "reconstruction_only"
    cfg["model"]["latent_encoder"]["use_kl"] = True
    cfg["model"]["reconstruction"]["lambda_rec"] = 1.0
    cfg["loss"] = {
        "lambda_action": 0.0,
        "lambda_latent": 0.0,
        "kl_beta": 1.0e-4,
        "kl_warmup_steps": 0,
    }
    cfg["model"].pop("loss")
    model, _ = build_tiny(cfg)
    # The production trainers install top-level loss settings after model
    # construction; construction itself must therefore remain valid.
    model.lambda_action = cfg["loss"]["lambda_action"]
    model.lambda_latent = cfg["loss"]["lambda_latent"]
    model.kl_beta = cfg["loss"]["kl_beta"]
    model.kl_warmup_steps = cfg["loss"]["kl_warmup_steps"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reconstruction-only Stage 1 entered the policy path")

    monkeypatch.setattr(model.context_encoder, "encode_context", forbidden)
    monkeypatch.setattr(model.world_dit, "patchify_frame0", forbidden)
    monkeypatch.setattr(model.action_dit, "pre_dit", forbidden)
    monkeypatch.setattr(model.mot, "forward", forbidden)

    batch = batch_for(cfg, batch=1)
    for key in ("actions", "action_is_pad", "text_embeds", "text_mask"):
        batch[key] = None
    loss, logs = model(**batch)
    loss.backward()

    assert logs["feature_reconstruction_loss"] > 0
    assert logs["weighted_feature_reconstruction_loss"] == pytest.approx(
        logs["feature_reconstruction_loss"]
    )
    assert logs["weighted_kl_loss"] > 0
    assert "loss_action" not in logs
    per_frame_grad = batch["z_fwd_main"].grad.abs().sum(dim=(0, 1, 3, 4))
    assert torch.all(per_frame_grad > 0)
    assert model.reconstruction_decoder.net[-1].weight.grad is not None
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith((
            "latent_encoder.",
            "wrist_latent_encoder.",
            "reconstruction_decoder.",
            "wrist_reconstruction_decoder.",
        ))
        for name in trainable_names
    )


@pytest.mark.parametrize("horizon", [16, 24, 32])
def test_wan_vae_reconstruction_only_bypasses_policy_and_uses_every_future_frame(
    monkeypatch, horizon,
):
    cfg = tiny_cfg("wan_vae", horizon=horizon, reconstruction=True, stage=1)
    cfg["model"]["stage1_objective"] = "reconstruction_only"
    cfg["model"]["latent_encoder"]["use_kl"] = True
    cfg["model"]["reconstruction"]["lambda_rec"] = 1.0
    cfg["loss"] = {
        "lambda_action": 0.0,
        "lambda_latent": 0.0,
        "kl_beta": 1.0e-4,
        "kl_warmup_steps": 0,
    }
    cfg["model"].pop("loss")
    model, _ = build_tiny(cfg)
    model.lambda_action = cfg["loss"]["lambda_action"]
    model.lambda_latent = cfg["loss"]["lambda_latent"]
    model.kl_beta = cfg["loss"]["kl_beta"]
    model.kl_warmup_steps = cfg["loss"]["kl_warmup_steps"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reconstruction-only Stage 1 entered the policy path")

    monkeypatch.setattr(model.context_encoder, "encode_context", forbidden)
    monkeypatch.setattr(model.world_dit, "patchify_frame0", forbidden)
    monkeypatch.setattr(model.action_dit, "pre_dit", forbidden)
    monkeypatch.setattr(model.mot, "forward", forbidden)

    batch = batch_for(cfg, batch=1)
    batch["z_fwd_main"].requires_grad_(True)
    for key in ("actions", "action_is_pad", "text_embeds", "text_mask"):
        batch[key] = None
    loss, logs = model(**batch)
    loss.backward()

    assert logs["feature_reconstruction_loss"] > 0
    assert logs["weighted_feature_reconstruction_loss"] == pytest.approx(
        logs["feature_reconstruction_loss"]
    )
    assert logs["weighted_kl_loss"] > 0
    assert "loss_action" not in logs
    per_frame_grad = batch["z_fwd_main"].grad.abs().sum(dim=(0, 1, 2, 4, 5))
    assert torch.all(per_frame_grad > 0)
    assert model.reconstruction_decoder.net[-1].weight.grad is not None
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith((
            "latent_encoder.",
            "wrist_latent_encoder.",
            "reconstruction_decoder.",
            "wrist_reconstruction_decoder.",
        ))
        for name in trainable_names
    )


@pytest.mark.parametrize("visual_type", ["dinov3", "wan_vae"])
def test_reconstruction_only_encoder_transfer_preserves_selected_stage2_policy(visual_type):
    stage1_cfg = tiny_cfg(visual_type, reconstruction=True, stage=1)
    stage1_cfg["model"]["stage1_objective"] = "reconstruction_only"
    stage1_cfg["model"]["latent_encoder"]["use_kl"] = True
    stage1_cfg["model"]["reconstruction"]["lambda_rec"] = 1.0
    stage1_cfg["model"]["loss"].update({"lambda_action": 0.0, "lambda_latent": 0.0})
    torch.manual_seed(11)
    stage1, runtime1 = build_tiny(stage1_cfg)

    stage2_cfg = tiny_cfg(visual_type, reconstruction=False, stage=2)
    stage2_cfg["checkpoint"] = {"policy_init": "fresh"}
    stage2_cfg["model"]["stage1_objective"] = "reconstruction_only"
    stage2_cfg["model"]["latent_encoder"]["use_kl"] = True
    torch.manual_seed(22)
    stage2, runtime2 = build_tiny(stage2_cfg)
    policy_before = {
        key: value.clone()
        for key, value in stage2.state_dict().items()
        if not key.startswith(("latent_encoder.", "wrist_latent_encoder."))
    }

    stage1_protocol = build_model_protocol(stage1, runtime1)
    stage2_protocol = build_model_protocol(stage2, runtime2)
    assert stage1_protocol["latent_encoder_use_kl"] is True
    assert stage1_protocol["stage1_objective"] == "reconstruction_only"
    assert "encoder_pretraining_objective" not in stage1_protocol
    assert "stage1_transfer_scope" not in stage2_protocol
    with pytest.raises(ValueError, match="reconstruction_decoder_enabled"):
        validate_model_protocol(stage1_protocol, stage2_protocol, "full_transfer")
    validate_model_protocol(
        stage1_protocol, stage2_protocol, "encoder_transfer", stage1_to_stage2=True,
    )
    load_stage1_encoders(stage2, stage1.state_dict())
    assert all(
        torch.equal(stage2.state_dict()[key], value)
        for key, value in policy_before.items()
    )
    for prefix in ("latent_encoder.", "wrist_latent_encoder."):
        assert all(
            torch.equal(stage2.state_dict()[key], value)
            for key, value in stage1.state_dict().items()
            if key.startswith(prefix)
        )


def test_dino_reconstruction_tracks_latent_dim_and_grid_ablations():
    for latent_dim in (16, 32, 64, 128):
        cfg = tiny_cfg(
            "dinov3", latent_dim=latent_dim, grid=(None, 2, 2), reconstruction=True,
        )
        model, runtime = build_tiny(cfg)
        assert model.latent_dim == latent_dim
        assert model.reconstruction_decoder.net[0].in_channels == latent_dim
        assert model.wrist_reconstruction_decoder.net[0].in_channels == latent_dim
        protocol = build_model_protocol(model, runtime)
        assert protocol["latent_dim"] == latent_dim
        assert protocol["reconstruction_target_type"] == "dinov3_patch_features"
        assert protocol["reconstruction_out_channels"] == 768

    for side, tokens in ((4, 48), (2, 12), (1, 3)):
        cfg = tiny_cfg("dinov3", grid=(None, side, side), reconstruction=True)
        model, _ = build_tiny(cfg)
        shape = model.get_latent_grid_shape()
        assert shape == (3, side, side)
        assert shape[0] * shape[1] * shape[2] == tokens


def test_horizon_resolution():
    resolved, runtime = resolve_miniwam_protocol(
        {
            "data": {"future_rgb_horizon": 32, "action_horizon": 7},
            "model": {"visual_encoder": {"type": "wan_vae", "future": {}}},
        },
    )
    assert runtime.future_rgb_horizon == 32
    assert runtime.future_feature_horizon == 8
    assert runtime.native_future_feature_horizon == 8
    assert runtime.action_horizon == 7
    assert resolved["data"]["future_rgb_horizon"] == 32
    assert "rgb_horizon" not in resolved["model"]["visual_encoder"]["future"]

    _, dino_runtime = resolve_miniwam_protocol(tiny_cfg("dinov3", horizon=32))
    assert dino_runtime.future_feature_horizon == 8
    assert dino_runtime.native_future_feature_horizon == 32


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.last_input_dtype = None
        self.autocast_enabled = None

    def forward(self, pixel_values):
        self.last_input_dtype = pixel_values.dtype
        self.autocast_enabled = torch.is_autocast_enabled(pixel_values.device.type)
        batch = pixel_values.shape[0]
        return type("Output", (), {"last_hidden_state": torch.cat([
            torch.zeros(batch, 5, 768, device=pixel_values.device, dtype=pixel_values.dtype),
            torch.ones(batch, 196, 768, device=pixel_values.device, dtype=pixel_values.dtype),
        ], dim=1)})()


def test_dinov3_frame_encoder_returns_only_normalized_patch_tokens():
    backbone = FakeDINO()
    encoder = DINOv3FrameEncoder(backbone=backbone)
    rgb = torch.randint(0, 256, (2, 3, 240, 320), dtype=torch.uint8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = encoder(rgb)
        boundary = encoder.encode_for_miniwam(rgb)
    assert result.shape == (2, 196, 768)
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.ones_like(result))
    assert boundary.dtype == torch.bfloat16
    assert backbone.last_input_dtype == torch.float32
    assert backbone.autocast_enabled is False


def test_dino_offline_cache_and_online_boundary_are_exactly_equal(tmp_path):
    encoder = DINOv3FrameEncoder(backbone=FakeDINO()).eval()
    rgb = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    offline = encode_dinov3_frames(rgb, encoder, torch.device("cpu"), batch_size=2)
    cache_path = tmp_path / "dino_features.pt"
    torch.save({"features": offline}, cache_path)
    loaded = torch.load(str(cache_path), map_location="cpu", weights_only=True)["features"]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        online = encoder.encode_for_miniwam(rgb.float())
    assert loaded.dtype == online.dtype == torch.bfloat16
    assert torch.equal(loaded, online)


def _write_resumed_dino_episode(root, *, dtype=torch.bfloat16, storage_dtype="bfloat16"):
    episode = root / "episode_000000"
    episode.mkdir(parents=True)
    features = torch.ones((1, 1, 4, 32), dtype=dtype)
    torch.save({
        "dino_main": features,
        "dino_wrist": features.clone(),
        "main_present": torch.ones((1, 1), dtype=torch.bool),
        "wrist_present": torch.ones((1, 1), dtype=torch.bool),
    }, episode / "dino_features.pt")
    torch.save({
        "context": torch.zeros((4, 16), dtype=torch.bfloat16),
        "mask": torch.ones(4, dtype=torch.bool),
    }, episode / "text_embed.pt")
    np.save(episode / "actions.npy", np.zeros((1, 7), dtype=np.float32))
    np.save(episode / "proprio.npy", np.zeros((1, 8), dtype=np.float32))
    (episode / "metadata.json").write_text(json.dumps({
        "dataset_name": "test",
        "feature_format_version": "multiview_v1",
        "main_camera_keys": ["main"],
        "wrist_camera_keys": ["wrist"],
        "proprio_dim": 8,
        "num_raw_frames": 1,
        "dinov3": {
            "compute_dtype": "float32",
            "boundary_dtype": "bfloat16",
            "storage_dtype": storage_dtype,
        },
    }), encoding="utf-8")
    return episode


def test_resumed_dino_extraction_requires_bf16_tensors_and_matching_metadata(tmp_path):
    valid = _write_resumed_dino_episode(tmp_path / "valid")
    validate_multiview_episode(
        valid, "test", ["main"], ["wrist"], 8,
        require_vae=False, require_dino=True,
    )

    fp32 = _write_resumed_dino_episode(tmp_path / "fp32", dtype=torch.float32)
    with pytest.raises(ValueError, match="must use the BF16"):
        validate_multiview_episode(
            fp32, "test", ["main"], ["wrist"], 8,
            require_vae=False, require_dino=True,
        )

    bad_metadata = _write_resumed_dino_episode(
        tmp_path / "bad_metadata", storage_dtype="float32",
    )
    with pytest.raises(ValueError, match="precision metadata mismatch"):
        validate_multiview_episode(
            bad_metadata, "test", ["main"], ["wrist"], 8,
            require_vae=False, require_dino=True,
        )


@pytest.mark.parametrize("horizon", [16, 24, 32])
def test_dino_dataset_builds_windows_from_per_frame_cache_with_endpoint_padding(
    tmp_path, horizon,
):
    stats = {
        "test": {
            "stats_id": "test_stats",
            "action": {"normalization": "minmax_to_minus1_plus1", "min": [-1.0] * 7, "max": [1.0] * 7},
            "absolute_action_mask": [False] * 6 + [True],
        }
    }
    with (tmp_path / "dataset_statistics.pkl").open("wb") as handle:
        pickle.dump(stats, handle)
    episode = tmp_path / "episode_000000"
    episode.mkdir()
    main = torch.arange(3, dtype=torch.bfloat16).view(3, 1, 1, 1).expand(3, 1, 4, 32).clone()
    wrist = (main + 10).clone()
    torch.save({
        "dino_main": main, "dino_wrist": wrist,
        "main_present": torch.ones(3, 1, dtype=torch.bool),
        "wrist_present": torch.ones(3, 1, dtype=torch.bool),
    }, episode / "dino_features.pt")
    torch.save({"context": torch.randn(4, 16), "mask": torch.ones(4, dtype=torch.bool)}, episode / "text_embed.pt")
    np.save(episode / "actions.npy", np.zeros((3, 7), dtype=np.float32))
    (episode / "metadata.json").write_text(json.dumps({
        "dataset_name": "test", "num_raw_frames": 3, "num_main_slots": 1, "num_wrist_slots": 1,
        "dinov3": {"cache_layout": "per_frame_v1"}, "stats_id": "test_stats",
        "visual_window_contract": {
            "storage": "per_frame_v1",
            "future_horizon_source": "runtime_protocol",
            "endpoint_padding": "repeat_last_frame",
        },
    }))
    _, runtime = resolve_miniwam_protocol({
        "data": {"future_rgb_horizon": horizon, "action_horizon": 2},
        "model": {"visual_encoder": {"type": "dinov3", "future": {
            "temporal_downsample_factor": 4,
        }}},
    })
    dataset = MiniWAMPreextractedDataset(str(tmp_path), runtime_protocol=runtime)
    sample = dataset[1]
    assert sample["z_t_main"].shape == (1, 4, 32)
    assert sample["z_fwd_main"].shape == (1, horizon, 4, 32)
    assert torch.all(sample["z_fwd_main"][0, 0] == 2)
    assert torch.all(sample["z_fwd_main"][0, -1] == 2)
