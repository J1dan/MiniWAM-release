"""Build a MiniWAM model from a resolved config."""

from __future__ import annotations

from typing import Any, Dict

from .model import MiniWAM
from .native_feature_model import MiniWAMNativeFeatureModel
from .protocol import MiniWAMRuntimeProtocol


def is_native_config(model_cfg: Dict[str, Any]) -> bool:
    """Whether the config selects the native-feature model (no PRISM encoder)."""
    return str(model_cfg.get("architecture", "miniwam")).lower() == "native"


def build_miniwam_model(
    model_cfg: Dict[str, Any],
    runtime_protocol: MiniWAMRuntimeProtocol,
):
    if is_native_config(model_cfg):
        return MiniWAMNativeFeatureModel.from_config(
            model_cfg,
            default_num_future_frames=runtime_protocol.native_future_feature_horizon,
        )
    return MiniWAM.from_config(model_cfg, runtime_protocol)
