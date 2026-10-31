import json
import pickle

import numpy as np
import pytest
import torch

from miniwam.factory import build_miniwam_model
from miniwam.model import MiniWAM
from miniwam.native_feature_model import MiniWAMNativeFeatureModel
from miniwam.dataset import (
    VAE_LATENT_KEYS,
    MiniWAMCollator,
    MiniWAMPreextractedDataset,
)
from miniwam.prompting import DEFAULT_PROMPT, format_instruction_prompt
from miniwam.proprio import normalize_proprio
from miniwam.models.action_dit import ActionDiT
from miniwam.models.context_encoder import ContextEncoder
from miniwam.models.world_dit import WorldDiT
from miniwam.models.latent_encoder import LatentEncoder3D
from miniwam.models.mot import MoT
from miniwam.models.scheduler import FlowMatchScheduler
from miniwam.models.video_dit import VideoDiT
from miniwam.protocol import resolve_miniwam_protocol
from scripts.data.extraction_utils import (
    encode_rolling_vae_latents,
    format_prompt,
    resize_and_normalize_frames,
    stack_slot_latents,
)


torch.set_num_threads(1)


def make_runtime_protocol(*, future_rgb_horizon=8, action_horizon=4, visual_type="wan_vae"):
    _, runtime = resolve_miniwam_protocol(
        {
            "data": {
                "future_rgb_horizon": future_rgb_horizon,
                "action_horizon": action_horizon,
            },
            "model": {
                "visual_encoder": {
                    "type": visual_type,
                    "future": {"temporal_downsample_factor": 4},
                }
            },
        },
    )
    return runtime


def build_test_model(model_cfg, *, future_rgb_horizon=16, action_horizon=16):
    resolved, runtime = resolve_miniwam_protocol(
        {
            "stage": model_cfg.get("stage", 1),
            "data": {
                "future_rgb_horizon": future_rgb_horizon,
                "action_horizon": action_horizon,
            },
            "model": model_cfg,
        },
    )
    return build_miniwam_model(resolved["model"], runtime)


def test_instruction_prompt_consistency():
    instruction = "pick up the black bowl next to the ramekin and place it on the plate"
    expected = (
        "A video recorded from a robot's point of view executing the following instruction: "
        "pick up the black bowl next to the ramekin and place it on the plate"
    )
    assert DEFAULT_PROMPT == "A video recorded from a robot's point of view executing the following instruction: {task}"
    assert format_instruction_prompt(instruction) == expected
    assert format_prompt(instruction) == expected


def test_proprio_context_token_and_gradient():
    encoder = ContextEncoder(text_dim=16, hidden_dim=32, proprio_dim=8)
    text = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    proprio = torch.randn(2, 8, requires_grad=True)
    context, context_mask = encoder.encode_context(text, mask, proprio)
    assert context.shape == (2, 6, 32)
    assert context_mask.shape == (2, 6)
    assert context_mask[:, -1].all()
    context[:, -1].sum().backward()
    assert encoder.proprio_proj.weight.grad is not None
    assert torch.count_nonzero(encoder.proprio_proj.weight.grad) > 0


def test_proprio_normalization_clips_and_zeros_constant_dims():
    stats = {
        "min": np.zeros(8, dtype=np.float32),
        "max": np.ones(8, dtype=np.float32),
        "constant_mask": np.array([False] * 7 + [True]),
        "dim": 8,
        "representation": "test",
        "coordinate_frame": "test",
        "normalization": "minmax_to_minus1_plus1",
        "clip": 5.0,
    }
    raw = np.array([0.0, 0.5, 1.0, 3.0, -3.0, 0.25, 0.75, 99.0], dtype=np.float32)
    normalized = normalize_proprio(raw, stats)
    np.testing.assert_allclose(normalized[:3], [-1.0, 0.0, 1.0])
    assert normalized[3] == 5.0 and normalized[4] == -5.0
    assert normalized[-1] == 0.0


def tiny_encoder_cfg(use_kl=True):
    return {
        "in_channels": 16,
        "latent_dim": 8,
        "compression_stages": [
            (16, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
            (16, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
        ],
        "use_kl": use_kl,
    }


def make_tiny_model(stage=1, wrist=True, use_kl=True):
    enc_cfg = tiny_encoder_cfg(use_kl=use_kl)
    world_dit = WorldDiT(
        hidden_dim=32,
        latent_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=16,
        num_heads=2,
        attn_head_dim=16,
        num_layers=1,
        vae_channels=16,
        frame0_patch_size=(2, 2),
        use_gradient_checkpointing=False,
    )
    action_dit = ActionDiT(
        hidden_dim=32,
        action_dim=7,
        ffn_dim=64,
        text_dim=16,
        freq_dim=16,
        num_heads=2,
        attn_head_dim=16,
        num_layers=1,
        use_gradient_checkpointing=False,
    )
    model = MiniWAM(
        latent_encoder=LatentEncoder3D(**enc_cfg),
        wrist_latent_encoder=LatentEncoder3D(**enc_cfg) if wrist else None,
        world_dit=world_dit,
        action_dit=action_dit,
        context_encoder=ContextEncoder(text_dim=16, hidden_dim=32),
        mot=MoT({"world": world_dit, "action": action_dit}, mot_checkpoint_mixed_attn=False),
        world_scheduler=FlowMatchScheduler(),
        action_scheduler=FlowMatchScheduler(),
        stage=stage,
        kl_beta=1.0e-4,
        kl_warmup_steps=1,
        lambda_latent=1.0 if stage == 2 else 0.0,
        future_rgb_horizon=8,
    )
    if stage == 2:
        model.set_stage(2)
    model.train()
    return model


def test_joint_training_with_gradient_checkpointing_backward():
    model = make_tiny_model(stage=2, wrist=True, use_kl=False)
    model.world_dit.use_gradient_checkpointing = True
    model.action_dit.use_gradient_checkpointing = True
    model.mot.mot_checkpoint_mixed_attn = True

    loss, _ = model(**multiview_batch(batch_size=2, main_slots=1, wrist_slots=1))
    loss.backward()

    assert torch.isfinite(loss)
    assert model.action_dit.action_encoder.weight.grad is not None


def make_tiny_native_vae():
    video_dit = VideoDiT(
        hidden_dim=32,
        in_channels=16,
        out_channels=16,
        ffn_dim=64,
        text_dim=16,
        freq_dim=16,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=16,
        num_layers=1,
        use_gradient_checkpointing=False,
    )
    action_dit = ActionDiT(
        hidden_dim=32,
        action_dim=7,
        ffn_dim=64,
        text_dim=16,
        freq_dim=16,
        num_heads=2,
        attn_head_dim=16,
        num_layers=1,
        use_gradient_checkpointing=False,
    )
    model = MiniWAMNativeFeatureModel(
        video_dit=video_dit,
        action_dit=action_dit,
        context_encoder=ContextEncoder(text_dim=16, hidden_dim=32),
        mot=MoT({"video": video_dit, "action": action_dit}, mot_checkpoint_mixed_attn=False),
        video_scheduler=FlowMatchScheduler(),
        action_scheduler=FlowMatchScheduler(),
        lambda_video=1.0,
        lambda_action=1.0,
        default_num_future_frames=4,
    )
    model.train()
    return model


def base_batch(batch_size=2):
    return {
        "actions": torch.randn(batch_size, 4, 7),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        "text_embeds": torch.randn(batch_size, 5, 16),
        "text_mask": torch.ones(batch_size, 5, dtype=torch.bool),
    }


def multiview_batch(batch_size=2, main_slots=2, wrist_slots=1, wrist_present=True):
    batch = base_batch(batch_size)
    height = width = 8
    future = 2
    batch.update({
        "z_t_main": torch.randn(batch_size, main_slots, 16, 1, height, width),
        "z_fwd_main": torch.randn(batch_size, main_slots, 16, future, height, width),
        "main_present": torch.ones(batch_size, main_slots, dtype=torch.bool),
        "z_t_wrist": torch.randn(batch_size, wrist_slots, 16, 1, height, width),
        "z_fwd_wrist": torch.randn(batch_size, wrist_slots, 16, future, height, width),
        "wrist_present": torch.full((batch_size, wrist_slots), wrist_present, dtype=torch.bool),
    })
    if wrist_slots == 0:
        batch["z_t_wrist"] = torch.empty(batch_size, 0, 16, 1, height, width)
        batch["z_fwd_wrist"] = torch.empty(batch_size, 0, 16, future, height, width)
        batch["wrist_present"] = torch.zeros(batch_size, 0, dtype=torch.bool)
    return batch


def assert_self_only(mask, row):
    allowed = torch.nonzero(mask[row], as_tuple=False).flatten().tolist()
    assert allowed == [row]


def unsupported_batch(batch_size=2):
    batch = base_batch(batch_size)
    batch.update({
        "z_t": torch.randn(batch_size, 16, 1, 8, 8),
        "z_fwd": torch.randn(batch_size, 16, 2, 8, 8),
    })
    return batch


def write_dataset_statistics(root, stats=None):
    root.mkdir(parents=True, exist_ok=True)
    stats_path = root / "dataset_statistics.pkl"
    if stats_path.exists():
        return
    if stats is None:
        stats = {
            "test": {
                "action": {
                    "normalization": "q01_q99_to_minus1_plus1",
                    "q01": np.full(7, -1.0, dtype=np.float32),
                    "q99": np.full(7, 1.0, dtype=np.float32),
                },
                "absolute_action_mask": [False, False, False, False, False, False, True],
            }
        }
    for dataset_name, entry in stats.items():
        entry.setdefault("stats_id", f"{dataset_name}_test_stats")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)


def write_episode(
    root,
    idx,
    *,
    multiview=True,
    main_slots=2,
    wrist_slots=1,
    wrist_present=True,
    dataset_name="test",
    include_dataset_name=True,
):
    root.mkdir(parents=True, exist_ok=True)
    write_dataset_statistics(root)
    ep_dir = root / f"episode_{idx:06d}"
    ep_dir.mkdir(parents=True)
    num_raw = 3
    future = 2
    height = width = 8
    if multiview:
        torch.save({
            "z_t_main": torch.randn(num_raw, main_slots, 16, 1, height, width),
            "z_fwd_main": torch.randn(num_raw, main_slots, 16, future, height, width),
            "main_present": torch.ones(num_raw, main_slots, dtype=torch.bool),
            "z_t_wrist": torch.randn(num_raw, wrist_slots, 16, 1, height, width),
            "z_fwd_wrist": torch.randn(num_raw, wrist_slots, 16, future, height, width),
            "wrist_present": torch.full((num_raw, wrist_slots), wrist_present, dtype=torch.bool),
        }, ep_dir / "vae_latent.pt")
    else:
        torch.save({
            "unsupported_current_latent": torch.randn(num_raw, 16, 1, height, width),
            "unsupported_future_latent": torch.randn(num_raw, 16, future, height, width),
        }, ep_dir / "vae_latent.pt")
    torch.save({
        "context": torch.randn(5 + idx, 16),
        "mask": torch.ones(5 + idx, dtype=torch.bool),
    }, ep_dir / "text_embed.pt")
    np.save(ep_dir / "actions.npy", np.random.randn(num_raw, 7).astype("float32"))
    metadata = {
        "num_raw_frames": num_raw,
        "future_rgb_horizon": future * 4,
        "feature_format_version": "multiview_v1" if multiview else "unsupported",
        "num_main_slots": main_slots if multiview else 0,
        "num_wrist_slots": wrist_slots if multiview else 0,
        "per_view_resolution": [64, 64],
        "main_camera_keys": [f"main_{i}" for i in range(main_slots)] if multiview else [],
        "wrist_camera_keys": [f"wrist_{i}" for i in range(wrist_slots)] if multiview else [],
    }
    if include_dataset_name:
        metadata["dataset_name"] = dataset_name
        metadata["stats_id"] = f"{dataset_name}_test_stats"
    (ep_dir / "metadata.json").write_text(json.dumps(metadata))
    return ep_dir


def test_extraction_helpers_and_dummy_vae_padding():
    nhwc = torch.arange(2 * 3 * 4 * 3, dtype=torch.uint8).reshape(2, 3, 4, 3)
    out = resize_and_normalize_frames(nhwc, (6, 8))
    assert out.shape == (2, 3, 6, 8)
    assert out.min() >= -1.0 and out.max() <= 1.0

    nchw = torch.rand(2, 3, 6, 8)
    out_nchw = resize_and_normalize_frames(nchw, (6, 8))
    torch.testing.assert_close(out_nchw, nchw * 2.0 - 1.0)

    z_t_stacked, z_fwd_stacked = stack_slot_latents([], [], 3, 2, 7, 9)
    assert z_t_stacked.shape == (3, 0, 16, 1, 7, 9)
    assert z_fwd_stacked.shape == (3, 0, 16, 2, 7, 9)

    class DummyVAE:
        def __init__(self):
            self.inputs = []

        def single_encode(self, video_input, device):
            self.inputs.append(video_input.detach().cpu())
            bsz = video_input.shape[0]
            frames = 1 + (video_input.shape[2] - 1) // 4
            latents = torch.arange(bsz * 16 * frames, dtype=torch.float32).reshape(bsz, 16, frames, 1, 1)
            return latents.to(video_input.device)

    frames = torch.arange(3 * 3 * 2 * 2, dtype=torch.float32).reshape(3, 3, 2, 2)
    vae = DummyVAE()
    z_t, z_fwd = encode_rolling_vae_latents(
        frames, vae, device=torch.device("cpu"), dtype=torch.float32,
        num_future_raw_frames=4, batch_size=3,
    )
    assert z_t.shape == (3, 16, 1, 1, 1)
    assert z_fwd.shape == (3, 16, 1, 1, 1)
    assert len(vae.inputs) == 1
    joint_input = vae.inputs[0]
    assert joint_input.shape == (3, 3, 5, 2, 2)
    torch.testing.assert_close(joint_input[0, :, 0], frames[0])
    torch.testing.assert_close(joint_input[0, :, 1], frames[1])
    torch.testing.assert_close(joint_input[2, :, 0], frames[2])
    torch.testing.assert_close(joint_input[2, :, -1], frames[2])

    # DINO precision changes must not alter Wan's selected extraction dtype.
    bf16_vae = DummyVAE()
    encode_rolling_vae_latents(
        frames, bf16_vae, device=torch.device("cpu"), dtype=torch.bfloat16,
        num_future_raw_frames=4, batch_size=3,
    )
    assert bf16_vae.inputs[0].dtype == torch.bfloat16


def test_preextracted_dataset_and_collator_contracts(tmp_path):
    mv_root = tmp_path / "mv"
    write_episode(mv_root, 0, multiview=True, main_slots=1, wrist_slots=1, wrist_present=True)
    write_episode(mv_root, 1, multiview=True, main_slots=2, wrist_slots=1, wrist_present=False)
    ds = MiniWAMPreextractedDataset(str(mv_root), runtime_protocol=make_runtime_protocol())
    batch = MiniWAMCollator()([ds[0], ds[3]])
    assert batch["z_t_main"].shape == (2, 2, 16, 1, 8, 8)
    assert batch["z_fwd_main"].shape == (2, 2, 16, 2, 8, 8)
    assert batch["main_present"].tolist() == [[True, False], [True, True]]
    assert batch["z_t_main"][0, 1].abs().sum().item() == 0.0
    assert batch["z_t_wrist"].shape == (2, 1, 16, 1, 8, 8)
    assert batch["wrist_present"].tolist() == [[True], [False]]
    assert batch["text_embeds"].shape == (2, 6, 16)
    assert batch["action_is_pad"][0].tolist() == [False, False, False, True]
    assert ds.episode_metadata(1)["num_raw_frames"] == 3
    visual_only = ds.load_visual_sample(0)
    assert set(visual_only) == VAE_LATENT_KEYS
    for key in VAE_LATENT_KEYS:
        torch.testing.assert_close(visual_only[key], ds[0][key])

    main_only_root = tmp_path / "main_only"
    write_episode(main_only_root, 0, multiview=True, main_slots=2, wrist_slots=0)
    main_ds = MiniWAMPreextractedDataset(
        str(main_only_root), runtime_protocol=make_runtime_protocol()
    )
    main_batch = MiniWAMCollator()([main_ds[0], main_ds[1]])
    assert main_batch["z_t_wrist"].shape == (2, 0, 16, 1, 8, 8)
    assert main_batch["wrist_present"].shape == (2, 0)

    unsupported_root = tmp_path / "unsupported"
    write_episode(unsupported_root, 0, multiview=False)
    with pytest.raises(ValueError, match="Unsupported VAE latent schema"):
        MiniWAMPreextractedDataset(
            str(unsupported_root), runtime_protocol=make_runtime_protocol()
        )


@pytest.mark.parametrize("episode_schema_version", [2, 3])
def test_preextracted_dataset_loads_aligned_normalized_proprio(
    tmp_path, episode_schema_version,
):
    root = tmp_path / "proprio"
    proprio_stats = {
        "min": np.zeros(8, dtype=np.float32),
        "max": np.full(8, 2.0, dtype=np.float32),
        "variance": np.ones(8, dtype=np.float32),
        "constant_mask": np.array([False] * 7 + [True]),
        "dim": 8,
        "representation": "eef_xyz_axis_angle_gripper_qpos",
        "coordinate_frame": "simulator_world",
        "normalization": "minmax_to_minus1_plus1",
        "clip": 5.0,
    }
    write_dataset_statistics(root, stats={
        "test": {
            "action": {
                "normalization": "q01_q99_to_minus1_plus1",
                "q01": np.full(7, -1.0, dtype=np.float32),
                "q99": np.full(7, 1.0, dtype=np.float32),
            },
            "absolute_action_mask": [False] * 6 + [True],
            "proprio": proprio_stats,
            "schema_version": 2,
            "stats_id": "test_local_v2",
        }
    })
    ep_dir = write_episode(root, 0, main_slots=1, wrist_slots=1)
    raw = np.stack([
        np.zeros(8, dtype=np.float32),
        np.ones(8, dtype=np.float32),
        np.full(8, 2.0, dtype=np.float32),
    ])
    np.save(ep_dir / "proprio.npy", raw)
    metadata = json.loads((ep_dir / "metadata.json").read_text())
    metadata.update({"schema_version": episode_schema_version, "proprio_dim": 8,
                     "stats_id": "test_local_v2"})
    (ep_dir / "metadata.json").write_text(json.dumps(metadata))

    dataset = MiniWAMPreextractedDataset(
        str(root),
        runtime_protocol=make_runtime_protocol(action_horizon=2),
        proprio_dim=8,
    )
    torch.testing.assert_close(dataset[1]["proprio"][:7], torch.zeros(7))
    assert dataset[1]["proprio"][-1] == 0
    batch = MiniWAMCollator()([dataset[0], dataset[1]])
    assert batch["proprio"].shape == (2, 8)
    assert dataset.statistics["test"]["stats_id"] == "test_local_v2"


def test_mixed_root_uses_episode_dataset_statistics_for_padding(tmp_path):
    root = tmp_path / "mixed"
    write_dataset_statistics(
        root,
        stats={
            "synthetic_reference": {
                "action": {
                    "normalization": "q01_q99_to_minus1_plus1",
                    "q01": np.full(7, -1.0, dtype=np.float32),
                    "q99": np.full(7, 1.0, dtype=np.float32),
                },
                "absolute_action_mask": [False, False, False, False, False, False, True],
            },
            "synthetic_shifted": {
                "action": {
                    "normalization": "q01_q99_to_minus1_plus1",
                    "q01": np.full(7, 0.0, dtype=np.float32),
                    "q99": np.full(7, 2.0, dtype=np.float32),
                },
                "absolute_action_mask": [False, False, False, False, False, False, True],
            },
        },
    )
    write_episode(root, 0, dataset_name="synthetic_reference")
    write_episode(root, 1, dataset_name="synthetic_shifted")

    ds = MiniWAMPreextractedDataset(str(root), runtime_protocol=make_runtime_protocol())
    reference_last = ds[2]
    shifted_last = ds[5]
    torch.testing.assert_close(reference_last["actions"][1:, 0], torch.zeros(3))
    torch.testing.assert_close(shifted_last["actions"][1:, 0], -torch.ones(3))


def test_dataset_statistics_fail_fast_for_missing_or_bad_dataset_names(tmp_path):
    missing_name_root = tmp_path / "missing_name"
    write_episode(missing_name_root, 0, include_dataset_name=False)
    with pytest.raises(KeyError, match="dataset_name"):
        MiniWAMPreextractedDataset(
            str(missing_name_root), runtime_protocol=make_runtime_protocol()
        )

    unknown_name_root = tmp_path / "unknown_name"
    write_episode(unknown_name_root, 0, dataset_name="unknown_dataset")
    with pytest.raises(KeyError, match="unknown_dataset"):
        MiniWAMPreextractedDataset(
            str(unknown_name_root), runtime_protocol=make_runtime_protocol()
        )

    malformed_stats_root = tmp_path / "malformed_stats"
    write_dataset_statistics(
        malformed_stats_root,
        stats={"test": {"absolute_action_mask": [False, False, False, False, False, False, True]}},
    )
    write_episode(malformed_stats_root, 0)
    with pytest.raises(KeyError, match="action"):
        MiniWAMPreextractedDataset(
            str(malformed_stats_root), runtime_protocol=make_runtime_protocol()
        )


def test_main_slots_share_encoder_and_wrist_is_separate():
    model = make_tiny_model(stage=1, wrist=True)
    captures = {"main": [], "wrist": []}

    def main_hook(_module, inputs, _output):
        captures["main"].append(inputs[0].shape)

    def wrist_hook(_module, inputs, _output):
        captures["wrist"].append(inputs[0].shape)

    main_handle = model.latent_encoder.register_forward_hook(main_hook)
    wrist_handle = model.wrist_latent_encoder.register_forward_hook(wrist_hook)
    try:
        loss, stats = model(**multiview_batch(batch_size=2, main_slots=2, wrist_slots=1))
        loss.backward()
    finally:
        main_handle.remove()
        wrist_handle.remove()

    assert torch.isfinite(loss)
    assert {"loss_action", "loss_kl", "loss_total"}.issubset(stats)
    assert captures["main"] == [torch.Size([4, 16, 3, 8, 8])]
    assert captures["wrist"] == [torch.Size([2, 16, 3, 8, 8])]


@pytest.mark.parametrize("stage,expected_keys", [
    (1, {"loss_action", "loss_kl", "loss_total"}),
    (2, {"loss_action", "loss_latent", "loss_kl", "loss_total"}),
])
def test_training_loss_smokes_for_multiview_and_main_only(stage, expected_keys):
    torch.manual_seed(stage)
    for batch in [
        multiview_batch(main_slots=2, wrist_slots=1),
        multiview_batch(main_slots=2, wrist_slots=0),
    ]:
        model = make_tiny_model(stage=stage, wrist=True)
        loss, stats = model(**batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert expected_keys.issubset(stats)
        if stage == 2:
            assert not model.latent_encoder.training
            assert all(not p.requires_grad for p in model.latent_encoder.parameters())


def test_absent_wrist_is_skipped_and_present_wrist_requires_encoder():
    model = make_tiny_model(stage=1, wrist=True)
    wrist_calls = []
    handle = model.wrist_latent_encoder.register_forward_hook(
        lambda _module, inputs, _output: wrist_calls.append(inputs[0].shape)
    )
    try:
        loss, _ = model(**multiview_batch(main_slots=2, wrist_slots=1, wrist_present=False))
        loss.backward()
    finally:
        handle.remove()
    assert torch.isfinite(loss)
    assert wrist_calls == []
    assert all(p.grad is None for p in model.wrist_latent_encoder.parameters())

    with pytest.raises(ValueError, match="requires `wrist_latent_encoder`"):
        make_tiny_model(stage=1, wrist=False)


def test_attention_mask_mot_mask_shapes_and_dummy_view_rules():
    base = torch.ones(4, 4, dtype=torch.bool)
    present = torch.tensor([[True, False, True, True]])
    mask = MiniWAM._apply_token_presence_to_mask(base, present)
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 1].item() is False
    assert mask[0, 0, 1].tolist() == [False, True, False, False]

    model = make_tiny_model(stage=1, wrist=True)
    embeds = {
        "world": torch.randn(1, 2, 32),
        "action": torch.randn(1, 2, 32),
    }
    freqs = {
        "world": model.action_dit.freqs[:2].view(2, 1, -1),
        "action": model.action_dit.freqs[:2].view(2, 1, -1),
    }
    t_mod = {
        "world": torch.randn(1, 2, 6, 32),
        "action": torch.randn(1, 6, 32),
    }
    context = torch.randn(1, 3, 32)
    context_all = {
        "world": {"context": context, "mask": torch.ones(1, 2, 3, dtype=torch.bool)},
        "action": {"context": context, "mask": torch.ones(1, 2, 3, dtype=torch.bool)},
    }
    out_2d = model.mot(embeds, torch.ones(4, 4, dtype=torch.bool), freqs, context_all, t_mod)
    out_4d = model.mot(embeds, torch.ones(1, 1, 4, 4, dtype=torch.bool), freqs, context_all, t_mod)
    assert out_2d["world"].shape == (1, 2, 32)
    assert out_4d["action"].shape == (1, 2, 32)

    joint_model = make_tiny_model(stage=2, wrist=True)
    joint_base = joint_model._build_attention_mask(
        S_z0=2, S_L=3, S_a=2, device=torch.device("cpu"), bidirectional_L_a=True,
    )
    assert joint_base[0, :2].tolist() == [True, True]
    assert joint_base[0, 2:].any().item() is False
    assert joint_base[2:5].all().item() is True
    assert joint_base[5:].all().item() is True
    token_present = torch.tensor([[True, False, True, False, True, True, True]])
    masked_joint = MiniWAM._apply_token_presence_to_mask(joint_base, token_present)[0, 0]
    for idx in [1, 3]:
        assert_self_only(masked_joint, idx)
    assert masked_joint[5, 1].item() is False
    assert masked_joint[5, 4].item() is True

    vae_model = make_tiny_native_vae()
    joint_video_base = vae_model._build_video_action_mask(
        num_slots=2,
        tokens_per_slot=3,
        tokens_per_frame=1,
        action_seq_len=2,
        device=torch.device("cpu"),
    )
    slot_present = torch.tensor([[True, False]])
    video_present = vae_model._video_token_present(slot_present, tokens_per_slot=3)
    full_present = torch.cat([video_present, torch.ones(1, 2, dtype=torch.bool)], dim=1)
    joint_video_mask = MiniWAMNativeFeatureModel._apply_token_presence_to_mask(
        joint_video_base, full_present,
    )[0, 0]
    for idx in [3, 4, 5]:
        assert_self_only(joint_video_mask, idx)
    action_row = 6
    assert joint_video_mask[action_row, :3].all().item() is True
    assert joint_video_mask[action_row, 3:6].any().item() is False
    assert joint_video_mask[action_row, 6:].all().item() is True
    assert joint_video_mask[1, action_row].item() is True
    assert joint_video_mask[0, action_row].item() is False


def test_from_config_wrist_switch_and_checkpoint_key_extraction(tmp_path):
    cfg = {
        "latent_encoder": tiny_encoder_cfg(use_kl=False),
        "multi_view_latent_encoder": {"fusion": "token_concat"},
        "mot": {"num_heads": 2, "attn_head_dim": 16, "num_layers": 1, "checkpoint_mixed_attn": False},
        "world_dit": {
            "hidden_dim": 32,
            "ffn_dim": 64,
            "text_dim": 16,
            "freq_dim": 16,
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
    model = build_test_model(cfg)
    assert model.wrist_latent_encoder is not None
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    main_keys = {k.removeprefix("latent_encoder."): v for k, v in state.items() if k.startswith("latent_encoder.")}
    wrist_keys = {k.removeprefix("wrist_latent_encoder."): v for k, v in state.items() if k.startswith("wrist_latent_encoder.")}
    assert main_keys
    assert wrist_keys
    assert set(main_keys) == set(wrist_keys)


def test_native_vae_multiview_training_and_inference():
    torch.manual_seed(100)
    model = make_tiny_native_vae()
    batch = multiview_batch(batch_size=2, main_slots=2, wrist_slots=1, wrist_present=True)
    video_calls = []
    handle = model.video_dit.patch_embedding.register_forward_hook(
        lambda _module, inputs, _output: video_calls.append(inputs[0].shape)
    )
    try:
        loss, stats = model(**batch)
        loss.backward()
    finally:
        handle.remove()
    assert torch.isfinite(loss)
    assert {"loss_video", "loss_action", "loss_total"} <= set(stats)
    assert video_calls
    assert video_calls[0] == torch.Size([6, 16, 3, 8, 8])

    model.eval()
    with torch.no_grad():
        result = model.infer(
            z_t_main=batch["z_t_main"][:1],
            main_present=batch["main_present"][:1],
            z_t_wrist=batch["z_t_wrist"][:1],
            wrist_present=batch["wrist_present"][:1],
            text_embeds=batch["text_embeds"][:1],
            text_mask=batch["text_mask"][:1],
            action_horizon=4,
            num_inference_steps=1,
            seed=3,
            return_video=True,
        )
    assert result["actions"].shape == (1, 4, 7)
    assert result["z_fwd_main_pred"].shape == (1, 2, 16, 4, 8, 8)
    assert result["z_fwd_wrist_pred"].shape == (1, 1, 16, 4, 8, 8)


def test_native_vae_from_config_builds_multiview_model():
    cfg = {
        "architecture": "native",
        "stage": 2,
        "mot": {"num_heads": 2, "attn_head_dim": 16, "num_layers": 1, "checkpoint_mixed_attn": False},
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
    model = build_test_model(cfg)
    assert isinstance(model, MiniWAMNativeFeatureModel)
    loss, stats = model(**multiview_batch(batch_size=1, main_slots=1, wrist_slots=0))
    assert torch.isfinite(loss)
    assert {"loss_video", "loss_action", "loss_total"} <= set(stats)


def test_stage1_condition_zt_zeroes_only_mot_path():
    model = make_tiny_model(stage=1, wrist=True, use_kl=False)
    batch = multiview_batch(batch_size=2, main_slots=1, wrist_slots=0)
    captured = {}

    def encoder_hook(_module, inputs, _output):
        captured["encoder_abs_sum"] = inputs[0].abs().sum().item()

    def patcher_hook(_module, inputs, _output):
        captured["patcher_abs_sum"] = inputs[0].abs().sum().item()

    enc_handle = model.latent_encoder.register_forward_hook(encoder_hook)
    patch_handle = model.world_dit.frame0_patcher.register_forward_hook(patcher_hook)
    try:
        loss, _ = model(**batch)
    finally:
        enc_handle.remove()
        patch_handle.remove()
    assert torch.isfinite(loss)
    assert captured["encoder_abs_sum"] > 0
    assert captured["patcher_abs_sum"] == 0
