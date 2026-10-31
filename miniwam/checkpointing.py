"""MiniWAM checkpoints and Stage-2 initialization.

Every saved step is one self-contained folder::

    model.safetensors   model weights
    config.yaml         resolved training config
    statistics.json     action/proprio normalization statistics per dataset
    metadata.json       format version, training step and model protocol
    training_state.pt   optimizer and scheduler state (training runs only)

The model protocol records every setting that changes the meaning of the
weights; loading checks it against the model built from the config.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

from .protocol import MiniWAMRuntimeProtocol, resolve_miniwam_protocol

FORMAT_VERSION = 1
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.yaml"
STATISTICS_FILE = "statistics.json"
METADATA_FILE = "metadata.json"
TRAINING_STATE_FILE = "training_state.pt"

PROPRIO_PROJECTION_KEYS = frozenset({
    "context_encoder.proprio_proj.weight",
    "context_encoder.proprio_proj.bias",
})
ENCODER_MODULE_NAMES = ("latent_encoder", "wrist_latent_encoder")
RECONSTRUCTION_MODULE_NAMES = ("reconstruction_decoder", "wrist_reconstruction_decoder")
# Settings that legitimately differ between a Stage-1 checkpoint and the
# Stage-2 model it initializes: Stage 2 adds proprioception and drops the
# Stage-1 feature decoder.
STAGE2_ONLY_PROTOCOL_FIELDS = (
    "proprio_dim",
    "reconstruction_decoder_enabled",
    "reconstruction_target_type",
    "reconstruction_hidden_dim",
    "reconstruction_out_channels",
)


# ── Statistics ──────────────────────────────────────────────────────────────

def _statistics_to_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"ndarray": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _statistics_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_statistics_to_json(item) for item in value]
    return value


def _statistics_from_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"ndarray", "dtype"}:
            return np.asarray(value["ndarray"], dtype=value["dtype"])
        return {key: _statistics_from_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_statistics_from_json(item) for item in value]
    return value


# ── Folder format ──────────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    path: Path
    state_dict: dict
    config: dict
    statistics: dict
    metadata: dict

    @property
    def model_protocol(self) -> dict:
        return self.metadata["model_protocol"]


def save_checkpoint(
    directory,
    *,
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping,
    statistics: Mapping,
    global_step: int,
    model_protocol: Mapping,
    training_state: Mapping | None = None,
) -> Path:
    """Write one checkpoint folder; files appear only once complete."""
    directory = Path(directory)
    temporary = directory.with_name(directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in state_dict.items()}
    save_file(tensors, temporary / WEIGHTS_FILE, metadata={"format": "pt"})
    (temporary / CONFIG_FILE).write_text(yaml.safe_dump(dict(config), sort_keys=False))
    (temporary / STATISTICS_FILE).write_text(
        json.dumps(_statistics_to_json(statistics), indent=1) + "\n"
    )
    metadata = {
        "format_version": FORMAT_VERSION,
        "global_step": int(global_step),
        "model_protocol": dict(model_protocol),
    }
    (temporary / METADATA_FILE).write_text(json.dumps(metadata, indent=1) + "\n")
    if training_state is not None:
        torch.save(dict(training_state), temporary / TRAINING_STATE_FILE)
    if directory.exists():
        shutil.rmtree(directory)
    temporary.rename(directory)
    return directory


def load_checkpoint(directory) -> Checkpoint:
    """Read a checkpoint folder (or a ``latest`` link to one) on CPU."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"MiniWAM checkpoint folder not found: {directory}")
    metadata = json.loads((directory / METADATA_FILE).read_text())
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{directory} has checkpoint format {metadata.get('format_version')!r}; "
            f"this code reads format {FORMAT_VERSION}."
        )
    return Checkpoint(
        path=directory,
        state_dict=load_file(directory / WEIGHTS_FILE, device="cpu"),
        config=yaml.safe_load((directory / CONFIG_FILE).read_text()),
        statistics=_statistics_from_json(json.loads((directory / STATISTICS_FILE).read_text())),
        metadata=metadata,
    )


def load_training_state(directory) -> dict:
    return torch.load(Path(directory) / TRAINING_STATE_FILE, map_location="cpu", weights_only=True)


def restore_model_forward_step(model, training_state: Mapping, *, global_step: int,
                               gradient_accumulation_steps: int) -> None:
    """Restore the forward counter used by MiniWAM's KL warmup.

    Older checkpoints saved only optimizer updates. At an update boundary,
    each rank has made updates x accumulation forwards, so they remain usable.
    """
    if hasattr(model, "_global_step"):
        model._global_step = int(training_state.get(
            "model_forward_step", int(global_step) * int(gradient_accumulation_steps),
        ))


# ── Model protocol ─────────────────────────────────────────────────────────

def build_model_protocol(model, runtime_protocol: MiniWAMRuntimeProtocol) -> dict:
    """Describe every setting that changes the meaning of the model weights."""
    temporal_factor = int(getattr(model, "temporal_downsample_factor", 4))
    if temporal_factor != runtime_protocol.temporal_downsample_factor:
        raise ValueError(
            "Model temporal factor does not match resolved runtime protocol: "
            f"{temporal_factor} vs {runtime_protocol.temporal_downsample_factor}."
        )
    visual_type = str(getattr(model, "visual_encoder_type", "wan_vae"))
    if visual_type != runtime_protocol.visual_encoder_type:
        raise ValueError(
            "Model visual encoder does not match resolved runtime protocol: "
            f"{visual_type!r} vs {runtime_protocol.visual_encoder_type!r}."
        )
    is_dino = visual_type == "dinov3"
    visual = getattr(model, "visual_encoder_metadata", {}) or {}
    feature = visual.get("feature", {}) or {}

    def dino(value):
        return value if is_dino else None

    latent_encoder = getattr(model, "latent_encoder", None)
    decoder_enabled = bool(getattr(model, "reconstruction_decoder_enabled", False))
    return {
        "model_architecture": str(getattr(model, "model_architecture", "miniwam")),
        "action_dim": int(model.action_dit.action_dim),
        "proprio_dim": model.context_encoder.proprio_dim,
        "max_main_slots": int(model.max_main_slots),
        "max_wrist_slots": int(model.max_wrist_slots),
        "camera_rope_slot_stride": int(model.camera_rope_slot_stride),
        "action_horizon": runtime_protocol.action_horizon,
        "future_rgb_horizon": runtime_protocol.future_rgb_horizon,
        "temporal_downsample_factor": temporal_factor,
        "visual_encoder_type": visual_type,
        "dino_architecture": dino(visual.get("architecture")),
        "dino_pretraining": dino(visual.get("pretraining")),
        "dino_model_id": dino(visual.get("model_id")),
        "dino_revision": dino(visual.get("revision")),
        "dino_resolved_revision": dino(visual.get("resolved_revision", visual.get("revision"))),
        "dino_feature_type": dino(feature.get("type")),
        "dino_feature_layers": dino((feature.get("layers", {}) or {}).get("mode")),
        "dino_feature_normalization": dino(feature.get("normalization")),
        "dino_compute_dtype": dino(visual.get("compute_dtype")),
        "dino_boundary_dtype": dino(feature.get("output_dtype")),
        "dino_l2_normalize": dino(feature.get("l2_normalize")),
        "dino_include_cls": dino(feature.get("include_cls")),
        "dino_include_registers": dino(feature.get("include_registers")),
        "dino_input_size": dino(visual.get("input_size")),
        "dino_encode_all_frames": dino((visual.get("future", {}) or {}).get("encode_all_frames")),
        "dino_frozen": dino(visual.get("frozen")),
        "dino_cache_dtype": dino((visual.get("cache", {}) or {}).get("storage_dtype")),
        "latent_dim": int(getattr(model, "latent_dim", 64)),
        "latent_grid_shape": (
            list(model.get_latent_grid_shape()) if hasattr(model, "get_latent_grid_shape") else None
        ),
        "latent_bottleneck_type": str(getattr(model, "latent_bottleneck_type", "none")),
        "latent_encoder_use_kl": bool(latent_encoder.use_kl) if latent_encoder is not None else None,
        "stage1_objective": str(getattr(model, "stage1_objective", "inverse_dynamics")),
        "reconstruction_decoder_enabled": decoder_enabled,
        "reconstruction_target_type": model.reconstruction_target_type if decoder_enabled else None,
        "reconstruction_hidden_dim": int(model.reconstruction_hidden_dim) if decoder_enabled else None,
        "reconstruction_out_channels": int(model.reconstruction_out_channels) if decoder_enabled else None,
    }


def validate_model_protocol(
    saved: Mapping,
    current: Mapping,
    context: str,
    *,
    stage1_to_stage2: bool = False,
) -> None:
    """Raise if a checkpoint's protocol differs from the model built from the config."""
    if not isinstance(saved, Mapping):
        raise TypeError(f"{context}: checkpoint model protocol must be a mapping.")
    ignored = STAGE2_ONLY_PROTOCOL_FIELDS if stage1_to_stage2 else ()
    missing = object()
    errors = [
        f"{key}: checkpoint={saved.get(key, '<missing>')!r}, config={current.get(key, '<missing>')!r}"
        for key in sorted(set(saved) | set(current))
        if key not in ignored and saved.get(key, missing) != current.get(key, missing)
    ]
    if errors:
        raise ValueError(
            f"{context}: the checkpoint does not match the model config:\n  - "
            + "\n  - ".join(errors)
        )


# ── Model construction and Stage-2 initialization ──────────────────────────

def build_cpu_model_with_seed(builder, seed: int):
    """Build a CPU model reproducibly without consuming rank-local RNG state.

    Training constructs MiniWAM on CPU before moving or wrapping it. Isolating
    that initialization makes all fresh parameters identical for a fixed seed,
    independent of distributed rank and world size, while leaving each rank's
    stochastic-training RNG stream unchanged.
    """
    previous_state = torch.get_rng_state()
    seeded_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    try:
        torch.set_rng_state(seeded_generator.get_state())
        return builder()
    finally:
        torch.set_rng_state(previous_state)


def load_stage1_for_stage2(model: torch.nn.Module, state_dict: Mapping) -> None:
    """Copy every Stage-1 weight into the Stage-2 model.

    The Stage-2 proprio projection keeps its seeded initialization, and the
    Stage-1 feature decoders are dropped; every other key must match.
    """
    if PROPRIO_PROJECTION_KEYS.intersection(state_dict):
        raise ValueError("Stage 2 adds proprioception; the Stage-1 checkpoint must not have it.")
    target_state = model.state_dict()
    decoder_prefixes = tuple(f"{name}." for name in RECONSTRUCTION_MODULE_NAMES)
    source = {key: value for key, value in state_dict.items() if not key.startswith(decoder_prefixes)}
    missing = set(target_state) - PROPRIO_PROJECTION_KEYS - set(source)
    unexpected = set(source) - set(target_state)
    if missing or unexpected:
        raise RuntimeError(
            "Stage-1 checkpoint does not fit the Stage-2 model: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    model.load_state_dict({**target_state, **source}, strict=True)


def load_stage1_encoders(model: torch.nn.Module, state_dict: Mapping) -> None:
    """Copy only the main and wrist PRISM encoders; the policy stays seeded."""
    for module_name in ENCODER_MODULE_NAMES:
        prefix = f"{module_name}."
        encoder_state = {
            key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)
        }
        if not encoder_state:
            raise KeyError(f"Stage-1 checkpoint has no parameters under {prefix!r}.")
        getattr(model, module_name).load_state_dict(encoder_state, strict=True)


def initialize_stage2(model, stage1: Checkpoint, policy_init: str, runtime_protocol) -> None:
    """Initialize a Stage-2 MiniWAM from a Stage-1 checkpoint and freeze its encoder."""
    if int(stage1.config.get("stage", 0)) != 1:
        raise ValueError(f"{stage1.path} is not a Stage-1 checkpoint.")
    validate_model_protocol(
        stage1.model_protocol,
        build_model_protocol(model, runtime_protocol),
        context=f"Stage-2 initialization from {stage1.path}",
        stage1_to_stage2=True,
    )
    if policy_init == "stage1":
        load_stage1_for_stage2(model, stage1.state_dict)
    elif policy_init == "fresh":
        load_stage1_encoders(model, stage1.state_dict)
    else:
        raise ValueError(f"Unknown policy_init {policy_init!r}.")
    model.set_stage(2)


# ── Inference ──────────────────────────────────────────────────────────────

@dataclass
class LoadedPolicy:
    model: torch.nn.Module
    config: dict
    runtime: MiniWAMRuntimeProtocol
    statistics: dict
    path: Path


def load_policy(directory, device="cpu", dtype=torch.float32) -> LoadedPolicy:
    """Load a trained checkpoint folder as an evaluation-mode policy."""
    from .factory import build_miniwam_model

    checkpoint = load_checkpoint(directory)
    config, runtime = resolve_miniwam_protocol(checkpoint.config)
    model = build_miniwam_model(config["model"], runtime)
    if hasattr(model, "set_stage"):
        model.set_stage(config["stage"])
    validate_model_protocol(
        checkpoint.model_protocol,
        build_model_protocol(model, runtime),
        context=f"loading {checkpoint.path}",
    )
    model.load_state_dict(checkpoint.state_dict, strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    return LoadedPolicy(
        model=model,
        config=config,
        runtime=runtime,
        statistics=checkpoint.statistics,
        path=checkpoint.path,
    )
