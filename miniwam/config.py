"""Training configuration: a base file, one stage config, and optional overlays.

The stage config (e.g. ``configs/libero/stage1.yaml``) is merged over the base.
Each further file is an overlay for an ablation. Overlays are applied in order;
keys under an overlay's ``stage1:`` or ``stage2:`` section apply only to that
training stage, so one overlay file serves both stages of an ablation.
"""
from pathlib import Path

import yaml

STAGE_SECTIONS = ("stage1", "stage2")


def deep_merge(base: dict, override: dict) -> dict:
    """Merge ``override`` into ``base`` in place; lists and scalars are replaced."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _read(path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def overlay_affects_stage1(path) -> bool:
    """Whether an overlay changes anything besides its ``stage2:`` section."""
    return any(key != "stage2" for key in _read(path))


def stack_name(config_paths) -> str:
    """Name a run after its stage config and the overlays stacked on it."""
    first = Path(config_paths[0])
    parts = [f"{first.parent.name}_{first.stem}"]
    parts += [Path(path).stem for path in config_paths[1:]]
    return "__".join(parts)


def load_training_config(base_config, config_paths) -> dict:
    """Merge the base config, the stage config, and any overlays."""
    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]
    config_paths = [str(path) for path in config_paths]
    cfg = deep_merge(_read(base_config), _read(config_paths[0]))

    stage = int(cfg.get("stage", 1))
    for path in config_paths[1:]:
        layer = _read(path)
        if "stage" in layer:
            raise ValueError(f"{path}: overlays must not set the training stage.")
        sections = {name: layer.pop(name) for name in STAGE_SECTIONS if name in layer}
        deep_merge(cfg, layer)
        deep_merge(cfg, sections.get(f"stage{stage}") or {})

    if len(config_paths) > 1:
        name = stack_name(config_paths)
        cfg.setdefault("training", {})["output_dir"] = f"outputs/{name}"
        cfg.setdefault("logging", {})["wandb_run_name"] = name.replace("_", "-")
        native = cfg.get("model", {}).get("architecture") == "native"
        if stage == 2 and not native:  # the native-feature model has no Stage 1
            stage1_overlays = [p for p in config_paths[1:] if overlay_affects_stage1(p)]
            if stage1_overlays:
                stage1_config = Path(config_paths[0]).with_name(
                    Path(config_paths[0]).name.replace("stage2", "stage1")
                )
                stage1_name = stack_name([stage1_config, *stage1_overlays])
                cfg.setdefault("checkpoint", {})["stage1_path"] = f"outputs/{stage1_name}/latest"
    return cfg
