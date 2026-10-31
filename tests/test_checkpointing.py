import json

import numpy as np
import pytest
import torch
from torch import nn

from miniwam.checkpointing import (
    FORMAT_VERSION,
    METADATA_FILE,
    build_cpu_model_with_seed,
    load_checkpoint,
    load_stage1_encoders,
    load_stage1_for_stage2,
    load_training_state,
    restore_model_forward_step,
    save_checkpoint,
    validate_model_protocol,
)


class _Context(nn.Module):
    def __init__(self, proprio_dim=None):
        super().__init__()
        self.text_proj = nn.Linear(3, 4)
        self.proprio_proj = None if proprio_dim is None else nn.Linear(proprio_dim, 4)


class _EncoderModel(nn.Module):
    def __init__(self, proprio_dim=None, reconstruction=False):
        super().__init__()
        self.latent_encoder = nn.Linear(3, 4)
        self.wrist_latent_encoder = nn.Linear(3, 4)
        self.context_encoder = _Context(proprio_dim)
        self.policy = nn.Linear(4, 2)
        self.reconstruction_decoder = nn.Linear(4, 3) if reconstruction else None
        self.wrist_reconstruction_decoder = nn.Linear(4, 3) if reconstruction else None


STATISTICS = {
    "libero_10_no_noops_lerobot": {
        "action": {
            "normalization": "minmax_to_minus1_plus1",
            "min": np.array([-0.9375, 0.1, 1e-7], dtype=np.float32),
            "max": np.array([0.9375, 0.3, 2.0], dtype=np.float32),
            "constant_mask": np.array([False, False, True]),
        },
        "absolute_action_mask": [False, False, True],
        "stats_id": "libero_10_v2_abc",
        "schema_version": 2,
        "clip": 5.0,
    }
}


@pytest.mark.parametrize("saved_counter,expected_weight", [(None, 1.0e-6), (800, 0.8e-6)])
def test_resume_preserves_kl_warmup_with_accumulation(saved_counter, expected_weight):
    from miniwam.model import MiniWAM

    model = _EncoderModel()
    model._global_step = 0
    model.kl_beta = 1.0e-6
    model.kl_warmup_steps = 1000
    state = {} if saved_counter is None else {"model_forward_step": saved_counter}
    restore_model_forward_step(model, state, global_step=100, gradient_accumulation_steps=16)
    assert MiniWAM._get_kl_weight(model) == pytest.approx(expected_weight, rel=1e-12, abs=0)


def test_seeded_model_build_is_identical_and_preserves_rank_rng():
    torch.manual_seed(101)
    rank0_state = torch.get_rng_state().clone()
    rank0_model = build_cpu_model_with_seed(_EncoderModel, 42)
    assert torch.equal(torch.get_rng_state(), rank0_state)

    torch.manual_seed(202)
    rank1_state = torch.get_rng_state().clone()
    rank1_model = build_cpu_model_with_seed(_EncoderModel, 42)
    assert torch.equal(torch.get_rng_state(), rank1_state)

    assert all(
        torch.equal(rank0_model.state_dict()[key], value)
        for key, value in rank1_model.state_dict().items()
    )


def test_checkpoint_round_trip_is_exact(tmp_path):
    model = _EncoderModel(proprio_dim=8)
    optimizer = torch.optim.AdamW(model.parameters())
    saved = save_checkpoint(
        tmp_path / "step_000010",
        state_dict=model.state_dict(),
        config={"stage": 2, "model": {"architecture": "miniwam"}},
        statistics=STATISTICS,
        global_step=10,
        model_protocol={"action_dim": 7},
        training_state={"optimizer": optimizer.state_dict(), "scheduler": {"last_epoch": 10}},
    )
    assert not (tmp_path / "step_000010.tmp").exists()

    checkpoint = load_checkpoint(saved)
    assert checkpoint.metadata == {
        "format_version": FORMAT_VERSION, "global_step": 10, "model_protocol": {"action_dim": 7},
    }
    assert checkpoint.config["stage"] == 2
    for key, value in model.state_dict().items():
        assert torch.equal(checkpoint.state_dict[key], value)
    action = checkpoint.statistics["libero_10_no_noops_lerobot"]["action"]
    for key in ("min", "max", "constant_mask"):
        expected = STATISTICS["libero_10_no_noops_lerobot"]["action"][key]
        assert action[key].dtype == expected.dtype
        assert np.array_equal(action[key], expected)
    assert checkpoint.statistics["libero_10_no_noops_lerobot"]["absolute_action_mask"] == [
        False, False, True,
    ]
    assert load_training_state(saved)["scheduler"] == {"last_epoch": 10}


def test_checkpoint_rejects_another_format_version(tmp_path):
    saved = save_checkpoint(
        tmp_path / "step", state_dict={"w": torch.zeros(1)}, config={}, statistics={},
        global_step=0, model_protocol={},
    )
    metadata = json.loads((saved / METADATA_FILE).read_text())
    metadata["format_version"] = FORMAT_VERSION + 1
    (saved / METADATA_FILE).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checkpoint format"):
        load_checkpoint(saved)


def test_stage1_transfer_copies_everything_but_proprio_and_decoders():
    torch.manual_seed(10)
    stage1 = _EncoderModel(reconstruction=True)
    torch.manual_seed(20)
    target = _EncoderModel(proprio_dim=8)
    proprio_before = {
        key: value.clone() for key, value in target.context_encoder.proprio_proj.state_dict().items()
    }

    load_stage1_for_stage2(target, stage1.state_dict())

    target_state = target.state_dict()
    for key, value in stage1.state_dict().items():
        if not key.startswith(("reconstruction_decoder.", "wrist_reconstruction_decoder.")):
            assert torch.equal(target_state[key], value)
    for key, value in proprio_before.items():
        assert torch.equal(target.context_encoder.proprio_proj.state_dict()[key], value)


def test_stage1_transfer_rejects_proprio_and_missing_weights():
    with pytest.raises(ValueError, match="proprioception"):
        load_stage1_for_stage2(_EncoderModel(8), _EncoderModel(8).state_dict())
    state = _EncoderModel().state_dict()
    del state["policy.bias"]
    with pytest.raises(RuntimeError, match="policy.bias"):
        load_stage1_for_stage2(_EncoderModel(proprio_dim=8), state)


def test_fresh_policy_init_copies_only_the_encoders():
    torch.manual_seed(10)
    stage1 = _EncoderModel(reconstruction=True)
    torch.manual_seed(20)
    target = _EncoderModel(proprio_dim=8)
    policy_before = target.policy.weight.clone()

    load_stage1_encoders(target, stage1.state_dict())

    assert torch.equal(target.latent_encoder.weight, stage1.latent_encoder.weight)
    assert torch.equal(target.wrist_latent_encoder.weight, stage1.wrist_latent_encoder.weight)
    assert torch.equal(target.policy.weight, policy_before)


def test_protocol_validation_reports_every_difference():
    current = {"action_dim": 7, "proprio_dim": 8, "latent_dim": 64}
    validate_model_protocol(dict(current), current, context="resume")
    with pytest.raises(ValueError, match=r"latent_dim: checkpoint=16, config=64"):
        validate_model_protocol({**current, "latent_dim": 16}, current, context="resume")
    with pytest.raises(ValueError, match=r"action_dim: checkpoint='<missing>'"):
        validate_model_protocol({"proprio_dim": 8, "latent_dim": 64}, current, context="resume")


def test_stage1_protocol_may_differ_only_in_stage2_fields():
    stage2 = {"action_dim": 7, "proprio_dim": 8, "reconstruction_decoder_enabled": False}
    stage1 = {"action_dim": 7, "proprio_dim": None, "reconstruction_decoder_enabled": True}
    validate_model_protocol(stage1, stage2, context="init", stage1_to_stage2=True)
    with pytest.raises(ValueError, match="proprio_dim"):
        validate_model_protocol(stage1, stage2, context="init")
    with pytest.raises(ValueError, match="action_dim"):
        validate_model_protocol(
            {**stage1, "action_dim": 14}, stage2, context="init", stage1_to_stage2=True,
        )
