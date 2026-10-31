"""Regression tests for tiled camera RoPE and protocol manifests."""

import pytest
import torch

from miniwam.checkpointing import build_model_protocol, validate_model_protocol
from miniwam.factory import build_miniwam_model
from miniwam.model import MiniWAM
from miniwam.native_feature_model import MiniWAMNativeFeatureModel
from miniwam.protocol import resolve_miniwam_protocol


def model_config(stage=2):
    return {
        "stage": stage,
        "multi_view_latent_encoder": {
            "max_main_slots": 2,
            "max_wrist_slots": 1,
            "camera_rope_slot_stride": 32,
        },
        "latent_encoder": {"in_channels": 16, "latent_dim": 64},
        "mot": {"num_heads": 4, "attn_head_dim": 64, "num_layers": 2},
        "world_dit": {"hidden_dim": 256, "ffn_dim": 512},
        "action_dit": {"hidden_dim": 256, "ffn_dim": 512},
        "context_encoder": {
            "hidden_dim": 256,
            **({"proprio_dim": 8} if stage == 2 else {}),
        },
    }


def build_model(stage=2, horizon=16):
    model_cfg = model_config(stage=stage)
    resolved, runtime = resolve_miniwam_protocol({
        "stage": stage,
        "data": {"future_rgb_horizon": horizon, "action_horizon": 16},
        "model": model_cfg,
    })
    return MiniWAM.from_config(resolved["model"], runtime), runtime


def native_vae_config(stride=32):
    return {
        "architecture": "native",
        "multi_view_latent_encoder": {
            "max_main_slots": 2,
            "max_wrist_slots": 1,
            "camera_rope_slot_stride": stride,
        },
        "mot": {
            "num_heads": 2,
            "attn_head_dim": 16,
            "num_layers": 1,
            "checkpoint_mixed_attn": False,
        },
        "video_dit": {
            "hidden_dim": 32,
            "ffn_dim": 64,
            "text_dim": 16,
            "freq_dim": 16,
            "in_channels": 16,
            "out_channels": 16,
            "patch_size": [1, 2, 2],
            "use_gradient_checkpointing": False,
        },
        "action_dit": {
            "hidden_dim": 32,
            "ffn_dim": 64,
            "text_dim": 16,
            "freq_dim": 16,
            "action_dim": 7,
            "use_gradient_checkpointing": False,
        },
        "context_encoder": {"text_dim": 16, "hidden_dim": 32},
    }


def build_native_vae(stride=32, horizon=16):
    resolved, runtime = resolve_miniwam_protocol({
        "stage": 2,
        "data": {"future_rgb_horizon": horizon, "action_horizon": 16},
        "model": native_vae_config(stride=stride),
    })
    return build_miniwam_model(resolved["model"], runtime), runtime


def test_tiled_rope_uses_shared_camera_origin_for_z0_and_latent():
    model, _ = build_model()
    wrist_offset = model.max_main_slots * model.camera_rope_slot_stride
    z0_wrist = model.world_dit._build_z0_rope_freqs(
        196, torch.device("cpu"), z0_grid_shape=(14, 14), w_offset=wrist_offset,
    )
    latent_wrist = model.world_dit._build_3d_rope_freqs(
        (3, 4, 4), torch.device("cpu"), w_offset=wrist_offset,
    )
    assert torch.allclose(z0_wrist[0], latent_wrist[0])


def test_wrist_latent_does_not_collide_with_main_z0():
    model, _ = build_model()
    main_z0 = model.world_dit._build_z0_rope_freqs(
        196, torch.device("cpu"), z0_grid_shape=(14, 14), w_offset=0,
    )
    wrist_latent = model.world_dit._build_3d_rope_freqs(
        (3, 4, 4), torch.device("cpu"),
        w_offset=model.max_main_slots * model.camera_rope_slot_stride,
    )
    collisions = torch.all(
        torch.isclose(main_z0[:, None], wrist_latent[None, :]), dim=(-1, -2),
    )
    assert not collisions.any()


def test_slot_stride_rejects_wide_camera_grid():
    model, _ = build_model()
    model.camera_rope_slot_stride = 3
    with pytest.raises(ValueError, match="exceeds the configured tiled-RoPE stride"):
        model._encode_view_slots(
            z_t_slots=torch.zeros(1, 1, 16, 1, 8, 8),
            z_fwd_slots=torch.zeros(1, 1, 16, 4, 8, 8),
            present=torch.ones(1, 1, dtype=torch.bool),
            encoder=model.latent_encoder,
            z_t_for_mot_slots=torch.zeros(1, 1, 16, 1, 8, 8),
        )


def test_tiled_non_square_z0_requires_explicit_grid_shape():
    model, _ = build_model()
    with pytest.raises(ValueError, match="requires z0_grid_shape"):
        model.world_dit._build_z0_rope_freqs(
            10, torch.device("cpu"), w_offset=model.camera_rope_slot_stride,
        )


def test_role_slot_limits_are_enforced_before_encoding():
    model, _ = build_model()
    common = {
        "z_fwd_main": torch.zeros(1, 3, 16, 4, 8, 8),
        "main_present": torch.ones(1, 3, dtype=torch.bool),
        "z_t_wrist": torch.zeros(1, 1, 16, 1, 8, 8),
        "z_fwd_wrist": torch.zeros(1, 1, 16, 4, 8, 8),
        "wrist_present": torch.ones(1, 1, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="Main camera slot count 3"):
        model._prepare_multiview_inputs(
            z_t_main=torch.zeros(1, 3, 16, 1, 8, 8), **common,
        )


def test_joint_attention_is_unconditionally_bidirectional():
    model, _ = build_model()
    mask = model._build_attention_mask(
        S_z0=4, S_L=3, S_a=2, device=torch.device("cpu"), bidirectional_L_a=True,
    )
    assert mask[4:7, 7:9].all()
    assert mask[7:9, :].all()
    assert "joint_attention_mode" not in model_config()["mot"]


def test_model_protocol_includes_camera_stride():
    model, runtime = build_model()
    protocol = build_model_protocol(model, runtime)
    assert protocol["camera_rope_slot_stride"] == 32
    assert "joint_attention_mode" not in protocol
    validate_model_protocol(protocol, protocol, context="resume")


def test_manifest_with_missing_stride_is_rejected():
    model, runtime = build_model()
    protocol = build_model_protocol(model, runtime)
    incomplete = dict(protocol)
    del incomplete["camera_rope_slot_stride"]

    with pytest.raises(ValueError, match="camera_rope_slot_stride: checkpoint='<missing>'"):
        validate_model_protocol(incomplete, protocol, context="resume")


def test_malformed_protocol_is_rejected():
    model, runtime = build_model()
    protocol = build_model_protocol(model, runtime)
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_model_protocol([], protocol, context="resume")


def test_tiled_joint_training_path_runs_with_separate_camera_slots():
    model, _ = build_model(stage=2)
    model.train()
    batch_size = 1
    loss, metrics = model.training_loss(
        actions=torch.randn(batch_size, 4, 7),
        text_embeds=torch.randn(batch_size, 5, 4096),
        text_mask=torch.ones(batch_size, 5, dtype=torch.bool),
        proprio=torch.randn(batch_size, 8),
        action_is_pad=torch.zeros(batch_size, 4, dtype=torch.bool),
        z_t_main=torch.randn(batch_size, 1, 16, 1, 8, 8),
        z_fwd_main=torch.randn(batch_size, 1, 16, 4, 8, 8),
        main_present=torch.ones(batch_size, 1, dtype=torch.bool),
        z_t_wrist=torch.randn(batch_size, 1, 16, 1, 8, 8),
        z_fwd_wrist=torch.randn(batch_size, 1, 16, 4, 8, 8),
        wrist_present=torch.ones(batch_size, 1, dtype=torch.bool),
    )

    assert torch.isfinite(loss)
    assert metrics["loss_latent"] >= 0.0
    assert metrics["loss_action"] >= 0.0


def test_direct_vae_tiled_rope_reserves_wrist_role_band():
    model, runtime = build_native_vae()
    assert isinstance(model, MiniWAMNativeFeatureModel)
    mv = model._compose_slots(
        z_t_main=torch.randn(1, 1, 16, 1, 8, 8),
        z_fwd_main=torch.randn(1, 1, 16, 2, 8, 8),
        main_present=torch.ones(1, 1, dtype=torch.bool),
        z_t_wrist=torch.randn(1, 1, 16, 1, 8, 8),
        z_fwd_wrist=torch.randn(1, 1, 16, 2, 8, 8),
        wrist_present=torch.ones(1, 1, dtype=torch.bool),
    )
    assert mv["slot_ids"] == (0, 2)

    pre = model._pre_video_slots(
        mv["z_video"],
        timestep=torch.ones(1),
        text_ctx=torch.randn(1, 3, 32),
        ctx_mask=torch.ones(1, 3, dtype=torch.bool),
        slot_ids=mv["slot_ids"],
    )
    tokens_per_slot = pre["tokens_per_slot"]
    grid_size = pre["flat"]["meta"]["grid_size"]
    expected_main = model.video_dit._build_3d_rope_freqs(
        grid_size, torch.device("cpu"), w_offset=0,
    )
    expected_wrist = model.video_dit._build_3d_rope_freqs(
        grid_size,
        torch.device("cpu"),
        w_offset=model.max_main_slots * model.camera_rope_slot_stride,
    )
    assert torch.equal(pre["freqs"][:tokens_per_slot], expected_main)
    assert torch.equal(pre["freqs"][tokens_per_slot:], expected_wrist)
    assert not torch.equal(expected_main, expected_wrist)

    protocol = build_model_protocol(model, runtime)
    assert protocol["camera_rope_slot_stride"] == model.camera_rope_slot_stride
    assert protocol["camera_rope_slot_stride"] == 32


def test_direct_vae_tiled_rope_rejects_too_small_stride():
    model, _ = build_native_vae(stride=3)
    mv = model._compose_slots(
        z_t_main=torch.randn(1, 1, 16, 1, 8, 8),
        z_fwd_main=torch.randn(1, 1, 16, 2, 8, 8),
        main_present=torch.ones(1, 1, dtype=torch.bool),
        z_t_wrist=torch.empty(1, 0, 16, 1, 8, 8),
        z_fwd_wrist=torch.empty(1, 0, 16, 2, 8, 8),
        wrist_present=torch.empty(1, 0, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="exceeds the configured tiled-RoPE stride"):
        model._pre_video_slots(
            mv["z_video"],
            timestep=torch.ones(1),
            text_ctx=torch.randn(1, 3, 32),
            ctx_mask=torch.ones(1, 3, dtype=torch.bool),
            slot_ids=mv["slot_ids"],
        )
