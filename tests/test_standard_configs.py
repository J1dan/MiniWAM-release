from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from miniwam.config import load_training_config
from miniwam.protocol import resolve_miniwam_protocol
from scripts.train import resolve_per_device_batch_size

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "configs/base.yaml"
ABL = "configs/ablations"


def load(*paths):
    cfg = load_training_config(BASE, [ROOT / path for path in paths])
    resolved, runtime = resolve_miniwam_protocol(cfg)
    return resolved, runtime


@pytest.mark.parametrize("stage", [1, 2])
def test_libero_benchmark_recipe(stage):
    cfg, runtime = load(f"configs/libero/stage{stage}.yaml")
    model, training = cfg["model"], cfg["training"]
    assert cfg["stage"] == stage
    assert runtime.visual_encoder_type == "dinov3"
    assert runtime.action_horizon == runtime.future_rgb_horizon == 16
    assert model["mot"] == {"num_heads": 12, "attn_head_dim": 64, "num_layers": 12,
                            "checkpoint_mixed_attn": True}
    assert model["world_dit"]["hidden_dim"] == model["action_dit"]["hidden_dim"] == 768
    assert model["latent_encoder"]["latent_dim"] == 64
    assert model["latent_encoder"]["bottleneck"]["type"] == "none"
    assert training["global_batch_size"] == 256
    assert resolve_per_device_batch_size(training, world_size=1) == 256
    four_gpus = deepcopy(training)
    four_gpus["per_device_batch_size"] = 64
    assert resolve_per_device_batch_size(four_gpus, world_size=4) == 64
    if stage == 1:
        assert training["max_steps"] == 16260
        assert model["reconstruction"]["lambda_rec"] == 1.0
        assert cfg["loss"]["lambda_action"] == 1.0
    else:
        assert training["max_steps"] == 54200
        assert cfg["loss"]["lambda_latent"] == 1.0
        assert model["context_encoder"]["proprio_dim"] == 8


@pytest.mark.parametrize("stage", [1, 2])
def test_robotwin_benchmark_recipe(stage):
    cfg, runtime = load(f"configs/robotwin/stage{stage}.yaml")
    assert runtime.visual_encoder_type == "dinov3"
    assert runtime.action_horizon == runtime.future_rgb_horizon == 24
    assert cfg["model"]["action_dit"]["action_dim"] == 14
    assert cfg["model"]["multi_view_latent_encoder"]["max_main_slots"] == 1
    assert cfg["model"]["multi_view_latent_encoder"]["max_wrist_slots"] == 2
    assert cfg["training"]["max_steps"] == ("6epoch" if stage == 1 else "25epoch")
    if stage == 2:
        assert cfg["inference"]["replan_steps"] == "action_horizon"


BENCHMARK_STAGE2 = ["configs/libero/stage2.yaml", "configs/robotwin/stage2.yaml"]
ABLATION_STAGE2 = [
    ("configs/libero/stage2.yaml", f"{ABL}/fresh_policy.yaml"),
    ("configs/libero/stage2.yaml", f"{ABL}/no_cotrain.yaml"),
    ("configs/libero/stage2.yaml", f"{ABL}/objective/inverse_only.yaml"),
    ("configs/libero/stage2.yaml", f"{ABL}/objective/recon_only.yaml"),
    ("configs/libero/stage2.yaml", f"{ABL}/capacity/grid2x2_d64.yaml"),
    ("configs/libero/stage2.yaml", f"{ABL}/capacity/grid1x1_d16.yaml"),
    ("configs/libero/stage2.yaml", f"{ABL}/capacity/grid2x2_d64.yaml", f"{ABL}/objective/recon_only.yaml"),
    (f"{ABL}/libero/wan_vae_stage2.yaml",),
    (f"{ABL}/libero/wan_vae_stage2.yaml", f"{ABL}/no_cotrain.yaml"),
    ("configs/robotwin/stage2.yaml", f"{ABL}/no_cotrain.yaml"),
    (f"{ABL}/robotwin/wan_vae_stage2.yaml",),
]


@pytest.mark.parametrize("path", BENCHMARK_STAGE2)
def test_benchmark_stage2_starts_from_stage1(path):
    cfg, _ = load(path)
    assert cfg["checkpoint"]["policy_init"] == "stage1"
    assert cfg["model"]["reconstruction"]["decoder_enabled"] is False


def test_policy_init_defaults_to_fresh():
    cfg = load_training_config(BASE, [ROOT / "configs/libero/stage2.yaml"])
    del cfg["checkpoint"]["policy_init"]
    resolved, _ = resolve_miniwam_protocol(cfg)
    assert resolved["checkpoint"]["policy_init"] == "fresh"


@pytest.mark.parametrize("paths", ABLATION_STAGE2)
def test_ablation_stage2_starts_from_seeded_policy(paths):
    cfg, _ = load(*paths)
    assert cfg["checkpoint"]["policy_init"] == "fresh"
    assert cfg["model"]["reconstruction"]["decoder_enabled"] is False


@pytest.mark.parametrize("path", [f"{ABL}/libero/native_dinov3.yaml", f"{ABL}/libero/native_wan_vae.yaml",
                                  f"{ABL}/robotwin/native_dinov3.yaml",
                                  f"{ABL}/robotwin/native_wan_vae.yaml"])
def test_native_feature_arms_have_no_stage1(path):
    cfg, _ = load(path)
    assert cfg["stage"] == 2
    assert cfg["model"]["architecture"] == "native"
    assert "stage1_path" not in cfg["checkpoint"]
    assert cfg["loss"]["lambda_video"] == 1.0
    assert cfg["training"]["global_batch_size"] == 256


def test_objective_overlays_change_only_their_axis():
    main1, _ = load("configs/libero/stage1.yaml")
    inverse1, _ = load("configs/libero/stage1.yaml", f"{ABL}/objective/inverse_only.yaml")
    recon1, _ = load("configs/libero/stage1.yaml", f"{ABL}/objective/recon_only.yaml")
    assert inverse1["model"]["reconstruction"]["decoder_enabled"] is False
    assert inverse1["model"]["reconstruction"]["lambda_rec"] == 0.0
    assert recon1["model"]["stage1_objective"] == "reconstruction_only"
    assert recon1["loss"]["lambda_action"] == 0.0
    assert recon1["model"]["reconstruction"]["lambda_rec"] == 1.0
    for cfg in (main1, inverse1, recon1):
        cfg["training"].pop("output_dir")
        cfg["logging"].pop("wandb_run_name")
        cfg["model"].pop("reconstruction")
        cfg["model"].pop("stage1_objective")
        cfg["loss"].pop("lambda_action")
    assert inverse1 == main1
    assert recon1 == main1


@pytest.mark.parametrize(("overlay", "grid", "dim"), [("grid2x2_d64", [None, 2, 2], 64),
                                                       ("grid1x1_d16", [None, 1, 1], 16)])
def test_capacity_overlays_apply_to_both_stages(overlay, grid, dim):
    for stage in (1, 2):
        cfg, _ = load(f"configs/libero/stage{stage}.yaml", f"{ABL}/capacity/{overlay}.yaml")
        encoder = cfg["model"]["latent_encoder"]
        assert encoder["latent_dim"] == dim
        assert encoder["bottleneck"] == {"type": "adaptive_avg_pool3d", "target_grid": grid}


def test_stacked_runs_are_named_and_linked_to_their_stage1_run():
    paths = [f"{ABL}/capacity/grid2x2_d64.yaml", f"{ABL}/objective/recon_only.yaml"]
    stage1, _ = load("configs/libero/stage1.yaml", *paths)
    stage2, _ = load("configs/libero/stage2.yaml", *paths)
    assert stage1["training"]["output_dir"] == "outputs/libero_stage1__grid2x2_d64__recon_only"
    assert stage2["checkpoint"]["stage1_path"] == f"{stage1['training']['output_dir']}/latest"
    # A Stage-2-only overlay reuses the stage config's own Stage-1 run.
    no_cotrain, _ = load("configs/libero/stage2.yaml", f"{ABL}/no_cotrain.yaml")
    main2, _ = load("configs/libero/stage2.yaml")
    assert no_cotrain["checkpoint"]["stage1_path"] == main2["checkpoint"]["stage1_path"]
    assert no_cotrain["loss"]["lambda_latent"] == 0.0
    fresh, _ = load("configs/libero/stage2.yaml", f"{ABL}/fresh_policy.yaml")
    assert fresh["checkpoint"]["stage1_path"] == main2["checkpoint"]["stage1_path"]
    assert fresh["training"]["output_dir"] == "outputs/libero_stage2__fresh_policy"
    assert {k: v for k, v in fresh.items() if k not in ("checkpoint", "training", "logging")} == \
        {k: v for k, v in main2.items() if k not in ("checkpoint", "training", "logging")}


def test_overlays_on_native_configs_do_not_link_a_stage1_run(tmp_path):
    overlay = tmp_path / "small_batch.yaml"
    overlay.write_text(yaml.safe_dump({"training": {"per_device_batch_size": 8}}))
    cfg, _ = load(f"{ABL}/libero/native_dinov3.yaml", str(overlay))
    assert "stage1_path" not in cfg["checkpoint"]


def test_overlays_cannot_change_the_stage(tmp_path):
    overlay = tmp_path / "bad.yaml"
    overlay.write_text(yaml.safe_dump({"stage": 2}))
    with pytest.raises(ValueError, match="must not set the training stage"):
        load_training_config(BASE, [ROOT / "configs/libero/stage1.yaml", overlay])


def test_every_stage_config_resolves():
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        if path.name.startswith("eval_") or path == BASE:
            continue
        if "stage" not in yaml.safe_load(path.read_text()):
            continue  # overlays are exercised together with a stage config above
        load(str(path.relative_to(ROOT)))
