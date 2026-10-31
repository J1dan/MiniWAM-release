import json
import os
import re
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from miniwam.config import load_training_config
from miniwam.protocol import resolve_miniwam_protocol

from miniwam.image_preprocessing import dino_rgb_uint8_to_bchw, wan_vae_rgb_uint8_to_bchw
from miniwam.proprio import PROPRIO_ZSCORE, compute_proprio_statistics, normalize_proprio
from miniwam.robotwin import (
    ROBOTWIN_ACTION_DIM_INFO,
    ROBOTWIN_EPISODE_IDS,
    ROBOTWIN_TASKS,
    RoboTwinEpisode,
    RoboTwinSplitError,
    assert_robotwin_data_config,
    assert_robotwin_joint_contract,
    resolve_robotwin_replan_steps,
    select_training_episodes,
)
from miniwam.robotwin_features import json_sha256, wan_vae_feature_contract
from scripts.data.extract_dataset_feature_robotwin import (
    _finalize_parallel_audit,
    _pack_episode,
    _resolve_sharding,
    _shard_audit_path,
    _shard_bounds,
    process_episode,
)
from scripts.data.extraction_utils import resize_and_normalize_frames
from scripts.eval import summarize_robotwin
from scripts.eval.summarize_robotwin import frozen_provenance, load_setting


def _make_task(root: Path, *, missing: str | None = None, duplicate: str | None = None):
    data_dir = root / "demo_clean" / "adjust_bottle" / "aloha_agilex" / "data"
    data_dir.mkdir(parents=True)
    for episode_id in ROBOTWIN_EPISODE_IDS:
        if episode_id != missing:
            (data_dir / f"episode_{episode_id}.hdf5").touch()
    if duplicate is not None:
        (data_dir / f"episode{int(duplicate)}.h5").touch()
    return data_dir


def test_fixed_episode_split_selects_0_through_49_and_audits_extras(tmp_path):
    data_dir = _make_task(tmp_path)
    (data_dir / "episode_0000050.hdf5").touch()

    episodes, audit = select_training_episodes(tmp_path, tasks=("adjust_bottle",))

    assert [episode.episode_id for episode in episodes] == list(ROBOTWIN_EPISODE_IDS)
    excluded = audit["tasks"]["adjust_bottle"]["excluded_additional_episode_ids"]
    assert list(excluded) == ["0000050"]


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_fixed_episode_split_fails_on_missing_or_duplicate_selected_id(tmp_path, kind):
    kwargs = {kind: "0000007"}
    _make_task(tmp_path, **kwargs)

    with pytest.raises(RoboTwinSplitError) as error:
        select_training_episodes(tmp_path, tasks=("adjust_bottle",))

    task_audit = error.value.audit["tasks"]["adjust_bottle"]
    assert task_audit[f"{kind}_selected_episode_ids"]


def test_rgb_boundary_changes_layout_without_swapping_channels():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[..., 0] = 11
    rgb[..., 1] = 22
    rgb[..., 2] = 33

    result = dino_rgb_uint8_to_bchw(rgb, source="test")

    assert tuple(result.shape) == (1, 3, 2, 3)
    np.testing.assert_array_equal(result[0, :, 0, 0].numpy(), [11, 22, 33])
    with pytest.raises(TypeError, match="uint8"):
        dino_rgb_uint8_to_bchw(rgb.astype(np.float32), source="test")


def test_offline_pack_calls_use_both_dataset_source_modes():
    calls = []

    def fake_pack(_raw, **kwargs):
        calls.append(kwargs)
        return np.zeros((3, 14), dtype=np.float32)

    proprio, action = _pack_episode({}, fake_pack, Path("episode.hdf5"))

    assert proprio.shape == action.shape == (3, 14)
    assert [(call["source_type"], call["state_type"]) for call in calls] == [
        ("dataset", "state"),
        ("dataset", "action"),
    ]


def test_robotwin_extraction_shards_cover_global_indices_once():
    ranges = [_shard_bounds(2500, shard, 4) for shard in range(4)]
    indices = [index for start, end in ranges for index in range(start, end)]
    assert ranges == [(0, 625), (625, 1250), (1250, 1875), (1875, 2500)]
    assert indices == list(range(2500))


def test_robotwin_extraction_uses_torchrun_sharding_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert _resolve_sharding(SimpleNamespace(shard_id=None, num_shards=None)) == (2, 4)
    monkeypatch.setenv("RANK", "3")
    assert _resolve_sharding(SimpleNamespace(shard_id=None, num_shards=None)) == (3, 4)
    assert _resolve_sharding(SimpleNamespace(shard_id=1, num_shards=3)) == (1, 3)
    with pytest.raises(ValueError, match="shard-id"):
        _resolve_sharding(SimpleNamespace(shard_id=3, num_shards=3))


def test_parallel_shard_audits_finalize_only_after_complete_coverage(tmp_path):
    audit = {"selected_episode_count": 5, "source_manifest_sha256": "manifest"}
    for index in range(5):
        episode_dir = tmp_path / f"episode_{index:06d}"
        episode_dir.mkdir()
        (episode_dir / "metadata.json").write_text(json.dumps({
            "visual_encoder_type": "wan_vae", "wan_vae": {"contract": "test"},
        }), encoding="utf-8")
    visual_hash = json_sha256({"contract": "test"})
    reports = [
        {"shard_id": 0, "num_shards": 2, "assigned": 3, "processed": 3,
         "resumed": 0, "complete": True, "stats_id": "stats",
         "source_manifest_sha256": "manifest", "visual_encoder_type": "wan_vae",
         "visual_contract_sha256": visual_hash},
        {"shard_id": 1, "num_shards": 2, "assigned": 2, "processed": 1,
         "resumed": 1, "complete": True, "stats_id": "stats",
         "source_manifest_sha256": "manifest"},  # Visual provenance may be absent.
    ]
    _shard_audit_path(tmp_path, 0, 2).write_text(
        json.dumps(reports[0]), encoding="utf-8"
    )
    assert not _finalize_parallel_audit(
        tmp_path, audit, num_shards=2, stats_id="stats",
        visual_encoder_type="wan_vae", visual_contract_sha256=visual_hash,
    )
    _shard_audit_path(tmp_path, 1, 2).write_text(
        json.dumps(reports[1]), encoding="utf-8"
    )
    assert _finalize_parallel_audit(
        tmp_path, audit, num_shards=2, stats_id="stats",
        visual_encoder_type="wan_vae", visual_contract_sha256=visual_hash,
    )
    final = json.loads((tmp_path / "extraction_audit.json").read_text())
    assert final["extraction"]["complete"] is True
    assert final["extraction"]["processed"] == 4
    assert final["extraction"]["resumed"] == 1
    assert final["extraction"]["visual_encoder_type"] == "wan_vae"
    assert final["extraction"]["visual_contract_sha256"] == visual_hash


def test_action_contract_is_schema_based_and_absolute():
    assert_robotwin_joint_contract(
        action_type="joint",
        robot_action_dim_info=ROBOTWIN_ACTION_DIM_INFO,
        action_dim=14,
        proprio_dim=14,
        action_representation="absolute_qpos",
    )
    with pytest.raises(ValueError, match="absolute_qpos"):
        assert_robotwin_joint_contract(
            action_type="joint",
            robot_action_dim_info=ROBOTWIN_ACTION_DIM_INFO,
            action_dim=14,
            proprio_dim=14,
            action_representation="delta_qpos",
        )


def test_proprio_zscore_formula():
    raw = np.array([[0.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    stats = compute_proprio_statistics(
        raw,
        representation="test",
        coordinate_frame="test",
        normalization=PROPRIO_ZSCORE,
    )
    result = normalize_proprio(np.array([[3.0, 2.0]], dtype=np.float32), stats)
    np.testing.assert_allclose(result, [[2.0, 0.0]])


def test_tracked_full_eval_config_matches_authoritative_task_constant():
    config = yaml.safe_load(
        Path("configs/robotwin/eval_frozen.yaml").read_text(encoding="utf-8")
    )
    assert tuple(config["tasks"]) == ROBOTWIN_TASKS


def _robotwin_config(*paths):
    return load_training_config("configs/base.yaml", list(paths))


NO_COTRAIN = ("configs/robotwin/stage2.yaml", "configs/ablations/no_cotrain.yaml")


@pytest.mark.parametrize(
    "config_paths",
    [
        ("configs/robotwin/stage1.yaml",),
        ("configs/robotwin/stage2.yaml",),
        NO_COTRAIN,
        ("configs/ablations/robotwin/native_dinov3.yaml",),
        ("configs/ablations/robotwin/native_wan_vae.yaml",),
        ("configs/ablations/robotwin/wan_vae_stage1.yaml",),
        ("configs/ablations/robotwin/wan_vae_stage2.yaml",),
    ],
)
def test_robotwin_training_configs_lock_split_and_normalization(config_paths):
    assert_robotwin_data_config(_robotwin_config(*config_paths)["data"])


def test_robotwin_no_cotrain_controlled_differences():
    baseline = _robotwin_config("configs/robotwin/stage2.yaml")
    ablation = _robotwin_config(*NO_COTRAIN)

    assert baseline["data"] == ablation["data"]
    assert baseline["model"] == ablation["model"]
    assert baseline["inference"] == ablation["inference"]
    assert baseline["checkpoint"] | {"policy_init": "fresh"} == ablation["checkpoint"]
    assert baseline["loss"] | {"lambda_latent": 0.0} == ablation["loss"]
    for key in (
        "global_batch_size", "per_device_batch_size",
        "gradient_accumulation_steps", "max_steps", "save_interval",
    ):
        assert baseline["training"][key] == ablation["training"][key]
    assert baseline["training"]["output_dir"] != ablation["training"]["output_dir"]


def test_robotwin_miniwam_uses_idm_plus_unit_reconstruction():
    stage1 = _robotwin_config("configs/robotwin/stage1.yaml")
    assert stage1["model"]["stage1_objective"] == "inverse_dynamics"
    assert stage1["loss"]["lambda_action"] == 1.0
    assert stage1["model"]["reconstruction"] == {
        "decoder_enabled": True,
        "lambda_rec": 1.0,
        "hidden_dim": 64,
    }

    for paths in (("configs/robotwin/stage2.yaml",), NO_COTRAIN):
        stage2, _ = resolve_miniwam_protocol(_robotwin_config(*paths))
        assert stage2["model"]["reconstruction"] == {
            "decoder_enabled": False,
            "lambda_rec": 0.0,
            "hidden_dim": 64,
        }


def test_robotwin_direct_dino_matches_native_feature_contract():
    config = yaml.safe_load(
        Path("configs/ablations/robotwin/native_dinov3.yaml").read_text(
            encoding="utf-8"
        )
    )
    model = config["model"]
    visual = model["visual_encoder"]
    video = model["video_dit"]

    assert config["stage"] == 2
    assert "checkpoint" not in config
    assert model["architecture"] == "native"
    assert "training_variant" not in model
    assert "latent_encoder" not in model
    assert "world_dit" not in model
    assert "adapter" not in visual
    assert config["data"]["future_rgb_horizon"] == 16
    assert visual["future"]["encode_all_frames"] is True
    assert video["in_channels"] == video["out_channels"] == 768
    assert video["patch_size"] == [1, 2, 2]
    assert model["multi_view_latent_encoder"]["max_main_slots"] == 1
    assert model["multi_view_latent_encoder"]["max_wrist_slots"] == 2
    assert model["context_encoder"]["proprio_dim"] == 14
    assert model["action_dit"]["action_dim"] == 14
    assert config["loss"]["lambda_video"] == 1.0
    assert "lambda_latent" not in config["loss"]
    assert config["training"]["per_device_batch_size"] == 32
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["inference"]["replan_steps"] == "action_horizon"


def test_robotwin_direct_vae_matches_native_feature_contract():
    config = yaml.safe_load(
        Path("configs/ablations/robotwin/native_wan_vae.yaml").read_text(
            encoding="utf-8"
        )
    )
    miniwam = yaml.safe_load(
        Path("configs/ablations/robotwin/wan_vae_stage2.yaml").read_text(
            encoding="utf-8"
        )
    )
    model = config["model"]
    visual = model["visual_encoder"]
    video = model["video_dit"]

    assert config["stage"] == 2
    assert "checkpoint" not in config
    assert config["data"] == miniwam["data"]
    assert (
        model["multi_view_latent_encoder"]
        == miniwam["model"]["multi_view_latent_encoder"]
    )
    assert model["context_encoder"] == miniwam["model"]["context_encoder"]
    assert model["architecture"] == "native"
    assert "training_variant" not in model
    assert "latent_encoder" not in model
    assert "world_dit" not in model
    assert visual["type"] == "wan_vae"
    assert config["data"]["features_root"].startswith("datasets/vae_features_")
    assert config["data"]["future_rgb_horizon"] == 16
    assert video["in_channels"] == video["out_channels"] == 16
    assert video["patch_size"] == [1, 2, 2]
    assert model["multi_view_latent_encoder"]["max_main_slots"] == 1
    assert model["multi_view_latent_encoder"]["max_wrist_slots"] == 2
    assert model["context_encoder"]["proprio_dim"] == 14
    assert model["action_dit"]["action_dim"] == 14
    assert config["loss"]["lambda_video"] == 1.0
    assert "lambda_latent" not in config["loss"]
    assert config["training"]["per_device_batch_size"] == 32
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["inference"]["replan_steps"] == "action_horizon"




def test_xpolicylab_model_does_not_decode_runtime_images():
    source = Path("integrations/xpolicylab/policy/MiniWAM/model.py").read_text(encoding="utf-8")
    assert "decode_image_bit(" not in source
    assert 'source_type="obs"' in source
    assert 'state_type="state"' in source
    assert "unpack_robot_state" in source


def test_robotwin_runtime_has_no_top_level_scripts_import():
    source = Path("deploy/robotwin_runtime.py").read_text(encoding="utf-8")
    assert "scripts.data" not in source
    assert "from miniwam.robotwin_text_cache import" in source


def test_robotwin_wan_vae_preprocessing_exactly_matches_libero_boundary():
    rgb = np.arange(2 * 9 * 11 * 3, dtype=np.uint8).reshape(2, 9, 11, 3)
    runtime = wan_vae_rgb_uint8_to_bchw(
        rgb, resolution=(16, 20), source="synthetic RoboTwin RGB",
    )
    offline_reference = resize_and_normalize_frames(rgb, (16, 20))
    assert runtime.shape == (2, 3, 16, 20)
    assert runtime.dtype == torch.float32
    torch.testing.assert_close(runtime, offline_reference, rtol=0, atol=0)


def test_robotwin_wan_vae_contract_is_hashed_and_horizon_bound(tmp_path):
    checkpoint = tmp_path / "Wan2.1_VAE.pth"
    checkpoint.write_bytes(b"synthetic-vae-weights")
    first = wan_vae_feature_contract(checkpoint, (224, 224), torch.bfloat16, 16)
    second = wan_vae_feature_contract(checkpoint, (224, 224), torch.bfloat16, 16)
    assert first == second
    assert len(first["checkpoint_sha256"]) == 64
    assert first["temporal_encoding"] == "joint_current_future_v1"
    assert first["resize"] == {
        "size_hw": [224, 224], "mode": "bilinear",
        "align_corners": False, "crop": None,
    }
    assert first["future_rgb_horizon"] == 16


def test_robotwin_wan_vae_episode_extraction_preserves_three_camera_slots(
    tmp_path, monkeypatch,
):
    import scripts.data.extract_dataset_feature_robotwin as extractor

    total, horizon = 3, 4
    raw = {
        "instruction": "move the object",
        "vision": {
            camera: {"colors": np.full((total, 6, 8, 3), value, dtype=np.uint8)}
            for camera, value in (
                ("cam_head", 10), ("cam_left_wrist", 20),
                ("cam_right_wrist", 30),
            )
        },
    }

    class DummyVAE:
        def single_encode(self, video, device):
            assert video.shape[1:] == (3, 1 + horizon, 8, 8)
            assert video.dtype == torch.bfloat16
            frames = 1 + horizon // 4
            return torch.zeros(
                video.shape[0], 16, frames, 1, 1,
                dtype=video.dtype, device=video.device,
            )

    monkeypatch.setattr(
        extractor, "encode_text",
        lambda *_args, **_kwargs: (
            torch.ones(2, 4, dtype=torch.bfloat16), torch.ones(2, dtype=torch.bool),
        ),
    )

    def pack(_raw, *, state_type, **_kwargs):
        fill = 0.25 if state_type == "action" else 0.5
        return np.full((total, 14), fill, dtype=np.float32)

    stats = {
        "stats_id": "synthetic",
        "action": {
            "normalization": "zscore_clip", "mean": [0.0] * 14,
            "std": [1.0] * 14, "clip": 5.0, "epsilon": 1e-8,
        },
    }
    changed = process_episode(
        0, RoboTwinEpisode("adjust_bottle", "0000000", tmp_path / "source.hdf5"),
        output_dir=tmp_path / "features", load_xspark_v1=lambda *_args, **_kwargs: raw,
        decode_image_bit=lambda value: value, pack_robot_state=pack,
        visual_encoder_type="wan_vae", dino_encoder=None, vae=DummyVAE(),
        visual_contract={"synthetic": True}, future_rgb_horizon=horizon,
        per_view_resolution=(8, 8), text_encoder=None, tokenizer=None,
        device=torch.device("cpu"), text_dtype=torch.bfloat16, batch_size=2,
        stats=stats, text_contract={"synthetic": True},
    )
    assert changed is True
    episode = tmp_path / "features/episode_000000"
    payload = torch.load(episode / "vae_latent.pt", weights_only=True)
    assert payload["z_t_main"].shape == (total, 1, 16, 1, 1, 1)
    assert payload["z_fwd_main"].shape == (total, 1, 16, 1, 1, 1)
    assert payload["z_t_wrist"].shape == (total, 2, 16, 1, 1, 1)
    assert payload["wrist_present"].tolist() == [[True, True]] * total
    metadata = json.loads((episode / "metadata.json").read_text())
    assert metadata["visual_encoder_type"] == "wan_vae"
    assert metadata["future_rgb_horizon"] == horizon
    assert metadata["main_camera_keys"] == ["cam_head"]
    assert metadata["wrist_camera_keys"] == ["cam_left_wrist", "cam_right_wrist"]


@pytest.mark.parametrize("stage", [1, 2])
def test_robotwin_wan_vae_configs_are_complete(stage):
    config = yaml.safe_load(
        Path(f"configs/ablations/robotwin/wan_vae_stage{stage}.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["model"]["visual_encoder"]["type"] == "wan_vae"
    assert config["model"]["visual_encoder"]["feature_grid"] == [28, 28]
    assert config["data"]["features_root"].startswith("datasets/vae_features_")
    assert config["data"]["action_horizon"] == 16
    assert config["data"]["future_rgb_horizon"] == 16
    assert config["model"]["multi_view_latent_encoder"]["max_main_slots"] == 1
    assert config["model"]["multi_view_latent_encoder"]["max_wrist_slots"] == 2
    if stage == 1:
        assert config["model"]["reconstruction"]["decoder_enabled"] is True
        assert config["model"]["reconstruction"]["lambda_rec"] == 1.0


@pytest.mark.parametrize("action_horizon", [16, 24, 32])
def test_robotwin_default_replan_executes_full_checkpoint_chunk(action_horizon):
    deploy = yaml.safe_load(
        Path("integrations/xpolicylab/policy/MiniWAM/deploy.yml").read_text(
            encoding="utf-8"
        )
    )
    assert deploy["action_horizon"] is None
    assert deploy["replan_steps"] == "action_horizon"
    assert resolve_robotwin_replan_steps(
        action_horizon, deploy["replan_steps"]
    ) == action_horizon
    assert resolve_robotwin_replan_steps(action_horizon, None) == action_horizon
    assert resolve_robotwin_replan_steps(action_horizon, 4) == 4


def test_robotwin_replan_override_is_strict():
    with pytest.raises(ValueError, match="must be in"):
        resolve_robotwin_replan_steps(16, 17)
    with pytest.raises(ValueError, match="integer or 'action_horizon'"):
        resolve_robotwin_replan_steps(16, "full")




def test_policy_environment_declares_xpolicylab_transport_dependencies():
    environment = yaml.safe_load(Path("environment.yml").read_text(encoding="utf-8"))
    pip_dependencies = next(
        item["pip"] for item in environment["dependencies"]
        if isinstance(item, dict) and "pip" in item
    )
    assert "websockets==15.0.1" in pip_dependencies
    assert "msgpack==1.1.1" in pip_dependencies
    assert "msgpack-numpy==0.4.8" in pip_dependencies
    assert "pydantic==2.13.4" in pip_dependencies

    server = Path(
        "integrations/xpolicylab/policy/MiniWAM/setup_eval_policy_server.sh"
    ).read_text(encoding="utf-8")
    assert "import websockets" in server
    assert "import msgpack_numpy" in server
    assert "install.sh ${policy_conda_env}" in server


@pytest.mark.parametrize(
    "checkpoint_argument",
    ["example_run", "example_run/latest", "outputs/example_run/step_000020"],
)
def test_policy_server_resolves_checkpoint_folders(tmp_path, checkpoint_argument):
    miniwam_root = tmp_path / "miniwam"
    run = miniwam_root / "outputs" / "example_run"
    for step in ("step_000010", "step_000020"):
        (run / step).mkdir(parents=True)
        for name in ("metadata.json", "statistics.json", "config.yaml"):
            (run / step / name).write_text(step)
        (run / step / "model.safetensors").write_bytes(step.encode())
    (run / "latest").symlink_to("step_000020")

    environment = os.environ.copy()
    environment.update({"MINIWAM_ROOT": str(miniwam_root), "MINIWAM_RESOLVE_ONLY": "1"})
    result = subprocess.run(
        [
            "bash", "integrations/xpolicylab/policy/MiniWAM/setup_eval_policy_server.sh",
            "RoboTwin", "adjust_bottle", checkpoint_argument, "aloha_agilex", "joint",
            "0", "0", "unused-conda-env", "54321",
        ],
        cwd=Path.cwd(), env=environment, text=True, capture_output=True, check=True,
    )
    # A moving latest link is pinned to the step folder it points to.
    assert f"[MiniWAM] checkpoint_path={run / 'step_000020'}" in result.stdout
    assert f"[MiniWAM] dataset_stats_path={run / 'step_000020' / 'statistics.json'}" in result.stdout


def test_xpolicylab_client_uses_official_dispatcher():
    source = Path(
        "integrations/xpolicylab/policy/MiniWAM/setup_eval_env_client.sh"
    ).read_text(encoding="utf-8")
    assert '"${UTILS_DIR}/setup_env_client.sh"' in source
    assert "run_sim_env_client.sh" in source
    assert "RoboTwin/scripts/eval_policy.sh" in source


def test_robotwin_renderer_compatibility_is_explicit_and_oidn_pinned():
    client = Path(
        "integrations/xpolicylab/policy/MiniWAM/setup_eval_env_client.sh"
    ).read_text(encoding="utf-8")
    helper = Path(
        "integrations/xpolicylab/policy/MiniWAM/compat/configure_renderer_runtime.sh"
    ).read_text(encoding="utf-8")
    assert "--query-gpu=compute_cap" not in client
    assert 'denoiser_override="optix"' not in client
    assert 'configure_renderer_runtime.sh" "${env_gpu_id}"' in client
    assert 'MINIWAM_ROBOTWIN_RENDER_PROFILE:-upstream' in helper
    assert 'oidn_2_3_3_blackwell)' in helper
    assert "libOpenImageDenoise_device_cuda.so.2.3.3" in helper
    assert 'export LD_PRELOAD=' in helper


def test_robotwin_client_has_stall_watchdog_and_busy_gpu_warning():
    client = Path(
        "integrations/xpolicylab/policy/MiniWAM/setup_eval_env_client.sh"
    ).read_text(encoding="utf-8")
    watchdog = Path(
        "integrations/xpolicylab/policy/MiniWAM/compat/run_with_inactivity_timeout.py"
    ).read_text(encoding="utf-8")
    assert "MINIWAM_EVAL_INACTIVITY_TIMEOUT_S:-900" in client
    assert "utilization.gpu" in client
    assert "already ${gpu_utilization}% utilized" in client
    assert "start_new_session=True" in watchdog
    assert "os.killpg(process.pid, signal.SIGTERM)" in watchdog
    assert "partial episodes must not be counted" in watchdog
    assert 'b"OIDN Error:"' in watchdog
    assert "return 125" in watchdog
    assert "validate_renderer.py" in client
    assert "renderer_preflight=passed" in client


def test_nvidia_runtime_paths_are_portable_and_opt_in():
    client = Path(
        "integrations/xpolicylab/policy/MiniWAM/setup_eval_env_client.sh"
    ).read_text(encoding="utf-8")
    server = Path(
        "integrations/xpolicylab/policy/MiniWAM/setup_eval_policy_server.sh"
    ).read_text(encoding="utf-8")
    helper = Path(
        "integrations/xpolicylab/policy/MiniWAM/compat/configure_nvidia_runtime.sh"
    ).read_text(encoding="utf-8")
    assert "/home/" not in client + server + helper
    assert not re.search(r"nvidia-\d+\.\d+", client + server + helper)
    assert "MINIWAM_NVIDIA_LIBRARY_PATH" in helper
    assert "MINIWAM_VULKAN_ICD_FILENAMES" in helper
    assert "reboot the host" in helper
    assert 'configure_nvidia_runtime.sh" simulator' in client
    assert 'configure_nvidia_runtime.sh" policy' in server


def test_miniwam_eval_fails_fast_for_missing_conda_environments():
    source = Path(
        "integrations/xpolicylab/policy/MiniWAM/eval.sh"
    ).read_text(encoding="utf-8")
    assert 'require_conda_env "${policy_conda_env}" "MiniWAM policy"' in source
    assert (
        'require_conda_env "${eval_env_conda_env}" "RoboTwin simulator"'
        in source
    )
    assert "import torch._C" in source
    assert "import sapien" in source
    assert "setup_robotwin_eval_env.sh" in source
    assert 'eval_env_conda_env=$(cd "${eval_env_conda_env}" && pwd)' in source
    assert "Conda compiler activation hooks are not nounset-safe" in source


def test_robotwin_eval_installer_keeps_large_files_on_workspace():
    source = Path("scripts/setup_robotwin_eval_env.sh").read_text(encoding="utf-8")
    assert 'ENV_PREFIX="${ROBOTWIN_EVAL_ENV_PREFIX:-${MINIWAM_ROOT}/.robotwin-env}"' in source
    assert 'export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda-pkgs"' in source
    assert 'export PIP_CACHE_DIR="${CACHE_ROOT}/pip"' in source
    assert 'export TMPDIR="${CACHE_ROOT}/tmp"' in source
    assert "import torch._C" in source
    assert 'TORCH_VERSION="2.4.1"' in source
    assert 'TORCH_VERSION="2.7.1"' in source
    assert 'PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"' in source
    assert 'export TORCH_CUDA_ARCH_LIST="12.0"' in source
    assert "nvidia-smi --query-gpu=compute_cap" in source
    assert "torch.sign" in source
    assert '"pillow==11.3.0"' in source
    assert "python -m pip check" in source
    assert '"kinematics_fused_cu"' in source
    assert "imageio_ffmpeg.get_ffmpeg_exe" in source
    assert 'ffmpeg -version >/dev/null' in source
    assert "set +u\nconda install" in source
    assert "at least 30 GiB free" in source
    for package in (
        "cuda-nvcc=${CUDA_FAMILY}",
        "cuda-cudart=${CUDA_FAMILY}",
        "cuda-cudart-dev=${CUDA_FAMILY}",
        "cuda-cccl=${CUDA_FAMILY}",
        "cuda-version=${CUDA_FAMILY}",
    ):
        assert package in source
    assert 'export CUDA_HOME="${ENV_PREFIX}"' in source
    assert "d64c4b005459db10c5dd867d8b30a87d5bda9bdb" in source
    assert "scripts/update_xpolicylab.sh" not in source


def test_setup_and_downloader_pin_verified_upstream_revisions():
    setup = Path("scripts/setup_robotwin.sh").read_text(encoding="utf-8")
    download = Path("scripts/download_robotwin_training_data.sh").read_text(
        encoding="utf-8"
    )
    assert "96c1feab536306b50c26af200044fcdf126e8904" in setup
    assert "785feb15aa4a4f532395ad2b1d2be5f28cb561ad" in download


def test_official_summary_parser_requires_complete_counted_results(tmp_path):
    run_dir = tmp_path / "run"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    jobs = []
    for index, task in enumerate(ROBOTWIN_TASKS):
        job_id = f"{index:03d}_{task}_seed0"
        (logs / f"{job_id}.log").write_text(
            "[MiniWAM] checkpoint_sha256=" + "a" * 64 + "\n"
            "[MiniWAM] dataset_stats_sha256=" + "b" * 64 + "\n"
            "Final success rate: 1/2 = 50.0%\n", encoding="utf-8"
        )
        jobs.append({
            "job_id": job_id,
            "task": task,
            "status": "success",
            "log": str(logs / f"{job_id}.log"),
            "command": ["bash", "eval.sh", "RoboTwin", task, "checkpoint-A"],
            "overrides": {"task_config": "demo_clean"},
        })
    summary = run_dir / "summary.json"
    summary.write_text(json.dumps({
        "jobs_total": 50,
        "jobs_succeeded": 50,
        "jobs": jobs,
    }), encoding="utf-8")

    rows, checkpoint = load_setting(
        summary, expected_setting="demo_clean", expected_trials=2,
    )

    assert len(rows) == 50
    assert checkpoint == "checkpoint-A@sha256:" + "a" * 64
    assert all(row["success_rate"] == 0.5 for row in rows)


def test_aggregate_frozen_provenance_is_sealed_and_complete(tmp_path, monkeypatch):
    manifest = tmp_path / "robotwin_eval_manifest.jsonl"
    manifest.write_text("manifest\n", encoding="utf-8")
    index = tmp_path / "frozen_index.json"
    index.write_text(json.dumps({"eval_manifest_sha256": "m"}), encoding="utf-8")
    index.with_suffix(".sha256.json").write_text(json.dumps({
        "text_cache_sha256": summarize_robotwin.sha256(index),
    }), encoding="utf-8")
    monkeypatch.setenv("MINIWAM_EVAL_MANIFEST", str(manifest))
    monkeypatch.setenv("MINIWAM_FROZEN_TEXT_INDEX", str(index))
    monkeypatch.setattr(summarize_robotwin, "load_manifest", lambda *_: ([], {
        "eval_manifest_sha256": "m",
        "schema_version": 2, "benchmark_protocol": "official_split_v1",
        "release_status": "certified_exact_local_replay",
        "strict_replay_validation": dict(exact_cases=10000, expected_cases=10000,
                                        scene_mismatch_cases=0, setup_error_cases=0),
        "prism_commit": "p" * 40,
        "sources": {"robotwin_commit": "r" * 40, "xpolicylab_commit": "x" * 40},
    }))
    from scripts.eval import robotwin_split_contract
    monkeypatch.setattr(robotwin_split_contract, 'require_marked_contract', lambda *_: {})

    result = frozen_provenance()

    assert result == {
        "eval_manifest_sha256": "m",
        "text_cache_sha256": summarize_robotwin.sha256(index),
        "prism_commit": "p" * 40,
        "robotwin_commit": "r" * 40,
        "xpolicylab_commit": "x" * 40,
    }


@pytest.mark.parametrize("stage", [1, 2])
def test_robotwin_release_defaults_preserve_training_and_inference_contract(stage):
    config = yaml.safe_load(Path(f"configs/robotwin/stage{stage}.yaml").read_text())
    assert config["data"]["action_horizon"] == 24
    assert config["data"]["future_rgb_horizon"] == 24
    assert config["training"]["max_steps"] == ("6epoch" if stage == 1 else "25epoch")
    if stage == 2:
        assert config["checkpoint"]["policy_init"] == "stage1"
        assert config["inference"]["replan_steps"] == "action_horizon"
