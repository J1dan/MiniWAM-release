"""XPolicyLab adapter for a dual-arm RoboTwin MiniWAM checkpoint."""

from __future__ import annotations

from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_action_dim,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from deploy.robotwin_runtime import MiniWAMRoboTwinRuntime
from miniwam.robotwin import assert_robotwin_joint_contract


def _instruction(obs: dict) -> str:
    value: Any = obs.get("instruction", obs.get("task_instruction", obs.get("instructions")))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, str):
        raise ValueError('Evaluation instruction must be an exact string')
    result = value
    if not result:
        raise ValueError('Missing evaluation instruction; default-instruction fallback is disabled.')
    return result


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg["action_type"])
        self.env_cfg_type = str(self.model_cfg["env_cfg_type"])
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        action_dim = int(get_action_dim(self.env_cfg_type))
        assert_robotwin_joint_contract(
            action_type=self.action_type,
            robot_action_dim_info=self.robot_action_dim_info,
            action_dim=action_dim,
            proprio_dim=14,
            action_representation="absolute_qpos",
            action_transform=None,
            delta_action_mask=None,
        )
        checkpoint_path = self.model_cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise FileNotFoundError("MiniWAM requires checkpoint_path (a checkpoint folder).")
        self.runtime = MiniWAMRoboTwinRuntime(
            checkpoint_path=checkpoint_path,
            project_root=self.model_cfg.get("project_root"),
            device=str(self.model_cfg.get("device", "cuda")),
            mixed_precision=str(self.model_cfg.get("mixed_precision", "bf16")),
            action_horizon=self.model_cfg.get("action_horizon"),
            replan_steps=self.model_cfg.get("replan_steps"),
            num_inference_steps=self.model_cfg.get("num_inference_steps"),
            seed=self.model_cfg.get("seed"),
        )
        self.action_horizon = self.runtime.action_horizon
        self.replan_steps = self.runtime.replan_steps
        self.last_obs: dict | None = None
        self._batch_obs: dict[int, dict] = {}

    @staticmethod
    def _require_cameras(obs: dict) -> None:
        vision = obs.get("vision")
        if not isinstance(vision, dict):
            raise KeyError("Runtime observation is missing vision.")
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            if camera not in vision or "color" not in vision[camera]:
                raise KeyError(f"Runtime observation is missing vision/{camera}/color.")
        # Do not decode here. The XPolicyLab websocket server has already run
        # decode_obs_images and supplies each color as HWC uint8 RGB.

    def update_obs(self, obs):
        self._require_cameras(obs)
        self.last_obs = obs

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty list.")
        updated: dict[int, dict] = {}
        for obs in obs_list:
            self._require_cameras(obs)
            if "env_idx" not in obs:
                raise KeyError("Batched RoboTwin observations require env_idx.")
            env_idx = int(obs["env_idx"])
            if env_idx in updated:
                raise ValueError(f"Duplicate env_idx in observation batch: {env_idx}.")
            updated[env_idx] = obs
        self._batch_obs = updated

    def _infer(self, observations: list[dict]):
        packed_states = []
        heads, left_wrists, right_wrists, instructions = [], [], [], []
        for obs in observations:
            vision = obs["vision"]
            heads.append(np.asarray(vision["cam_head"]["color"]))
            left_wrists.append(np.asarray(vision["cam_left_wrist"]["color"]))
            right_wrists.append(np.asarray(vision["cam_right_wrist"]["color"]))
            # Runtime observations intentionally use singular XPolicyLab keys.
            packed_states.append(
                pack_robot_state(
                    obs,
                    action_type=self.action_type,
                    robot_action_dim_info=self.robot_action_dim_info,
                    source_type="obs",
                    state_type="state",
                )
            )
            instructions.append(_instruction(obs))
        actions = self.runtime.infer_actions(
            head_rgb=heads,
            left_wrist_rgb=left_wrists,
            right_wrist_rgb=right_wrists,
            proprio=np.stack(packed_states).astype(np.float32),
            instructions=instructions,
        )
        # Make execution/replanning behavior explicit at the adapter boundary.
        actions = actions[:, : self.replan_steps]
        return [
            unpack_robot_state(
                chunk,
                action_type=self.action_type,
                robot_action_dim_info=self.robot_action_dim_info,
                source_type="obs",
            )
            for chunk in actions
        ]

    def get_action(self):
        if self.last_obs is None:
            raise ValueError("Call update_obs() before get_action().")
        return self._infer([self.last_obs])[0]

    def get_action_batch(self, env_idx_list=None):
        if not self._batch_obs:
            raise ValueError("Call update_obs_batch() before get_action_batch().")
        indices = (
            sorted(self._batch_obs)
            if env_idx_list is None else [int(index) for index in env_idx_list]
        )
        missing = [index for index in indices if index not in self._batch_obs]
        if missing:
            raise KeyError(f"No observation is stored for env indices {missing}.")
        return self._infer([self._batch_obs[index] for index in indices])

    def reset(self):
        self.last_obs = None
        self._batch_obs = {}
        self.runtime.reset()
