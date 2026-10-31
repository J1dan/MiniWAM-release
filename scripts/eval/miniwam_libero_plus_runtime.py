"""Persistent MiniWAM runtime used by LIBERO-Plus workers."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from miniwam.checkpointing import load_policy
from miniwam.libero_plus_text_cache import LiberoPlusTextFeatureCache
from miniwam.models.dinov3 import DINOv3FrameEncoder
from miniwam.statistics import get_action_bounds, select_statistics
from scripts.eval.libero_plus_protocol import (
    MAX_STEPS,
    get_libero_plus_env,
    get_libero_plus_init_states,
    resolve_libero_plus_instructions,
)


class MiniWAMLiberoPlusRuntime:
    """Load checkpoint/encoders once, then evaluate many LIBERO-Plus tasks."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        gpu_id: int,
        dtype: str,
        vae_path: Path,
        action_horizon: int | None,
        replan_steps: int | None,
        num_inference_steps: int | None,
        policy_seed: int,
        env_seed: int,
        num_steps_wait: int,
        save_videos: str,
        output_dir: Path,
        vanilla_instruction_registry: dict[str, dict[str, str]],
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MiniWAM LIBERO-Plus evaluation.")
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(f"cuda:{gpu_id}")
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.output_dir = Path(output_dir)
        self.save_videos = save_videos
        self.vanilla_instruction_registry = vanilla_instruction_registry

        self.policy = load_policy(self.checkpoint_path, device=self.device, dtype=self.dtype)
        self.cfg, self.runtime_protocol = self.policy.config, self.policy.runtime
        self.model_cfg = self.cfg["model"]
        self.model = self.policy.model

        from scripts.eval import eval_libero_miniwam as baseline

        self.baseline = baseline
        if getattr(self.model, "visual_encoder_type", "wan_vae") == "dinov3":
            visual_cfg = self.model_cfg["visual_encoder"]
            self.visual_encoder = DINOv3FrameEncoder(
                architecture=visual_cfg.get("architecture", "vitb16"),
                pretraining=visual_cfg.get("pretraining", "lvd1689m"),
                model_id=visual_cfg["model_id"],
                revision=visual_cfg.get("revision"),
                input_size=visual_cfg.get("input_size", 224),
                frozen=True,
            ).to(device=self.device, dtype=torch.float32).eval()
        else:
            self.visual_encoder = baseline.load_vae_model(vae_path, self.device, self.dtype)

        inference_cfg = self.cfg.get("inference", {})
        resolved_action_horizon = (
            int(action_horizon) if action_horizon is not None else int(self.cfg["data"]["action_horizon"])
        )
        resolved_replan_steps = (
            replan_steps if replan_steps is not None else inference_cfg["replan_steps"]
        )
        self.args = SimpleNamespace(
            action_horizon=resolved_action_horizon,
            replan_steps=int(resolved_replan_steps),
            num_inference_steps=int(
                num_inference_steps
                if num_inference_steps is not None
                else inference_cfg.get("num_inference_steps", 20)
            ),
            seed=int(policy_seed),
            num_steps_wait=int(num_steps_wait),
            binarize_gripper=True,
            capture_video=False,
        )
        self.policy_seed = int(policy_seed)
        self.env_seed = int(env_seed)
        self._statistics = {}

    def _suite_statistics(self, suite_name: str):
        cached = self._statistics.get(suite_name)
        if cached is not None:
            return cached
        dataset_name = f"{suite_name}_no_noops_lerobot"
        stats_path = self.policy.path
        ds_stats = select_statistics(self.policy.statistics, dataset_name, stats_path)
        lower, upper, constant, _ = get_action_bounds(ds_stats, stats_path)
        action_lower = torch.tensor(lower, device=self.device, dtype=torch.float32)
        action_upper = torch.tensor(upper, device=self.device, dtype=torch.float32)
        action_constant = torch.tensor(constant, device=self.device, dtype=torch.bool)
        proprio_stats = None
        proprio_dim = self.model_cfg.get("context_encoder", {}).get("proprio_dim")
        if proprio_dim not in (None, 0):
            proprio_stats = ds_stats["proprio"]
        cached = (action_lower, action_upper, action_constant, proprio_stats)
        self._statistics[suite_name] = cached
        return cached

    def evaluate(self, record, task_suite, text_cache: LiberoPlusTextFeatureCache) -> dict:
        task = task_suite.get_task(record.task_id)
        if task.name != record.task_name:
            raise ValueError(
                f"Runtime LIBERO-Plus task mismatch for {record.key}: {task.name!r} != {record.task_name!r}."
            )
        instructions = resolve_libero_plus_instructions(
            record, task, self.vanilla_instruction_registry,
        )
        expected_text_dim = int(self.model_cfg.get("context_encoder", {}).get("text_dim", 4096))
        context_len = int(self.model_cfg.get("context_encoder", {}).get("context_len", 128))
        context, mask = text_cache.get(
            record,
            instructions,
            expected_context_len=context_len,
            expected_text_dim=expected_text_dim,
        )
        text_embeds = context.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        text_mask = mask.unsqueeze(0).to(device=self.device, dtype=torch.bool)
        action_lower, action_upper, action_constant, proprio_stats = self._suite_statistics(record.suite)

        initial_states = get_libero_plus_init_states(task_suite, record.task_id)
        if len(initial_states) < 1:
            raise ValueError(f"LIBERO-Plus task {record.key} has no initial state.")
        env, runtime_language = get_libero_plus_env(
            task, self.baseline.LIBERO_ENV_RESOLUTION, self.env_seed,
        )
        if runtime_language != instructions.raw:
            env.close()
            raise ValueError(f"Environment changed task.language for {record.key}.")

        video_sample_selected = int.from_bytes(
            hashlib.sha256(
                f"libero_plus_video_sample10_v1\0{record.key}".encode("utf-8")
            ).digest()[:8],
            "big",
        ) % 10 == 0
        self.args.capture_video = (
            self.save_videos in ("all", "failures")
            or (self.save_videos == "sample10" and video_sample_selected)
        )

        random.seed(self.policy_seed)
        np.random.seed(self.policy_seed)
        torch.manual_seed(self.policy_seed)
        torch.cuda.manual_seed_all(self.policy_seed)
        try:
            success, replay_images = self.baseline.run_single_episode(
                env=env,
                initial_state=initial_states[0],
                task_description=instructions.clean,
                model=self.model,
                vae=self.visual_encoder,
                text_embeds=text_embeds,
                text_mask=text_mask,
                action_lower=action_lower,
                action_upper=action_upper,
                action_constant_mask=action_constant,
                proprio_stats=proprio_stats,
                args=self.args,
                device=self.device,
                dtype=self.dtype,
                episode_idx=0,
                max_steps=MAX_STEPS[record.suite],
            )
        finally:
            env.close()

        video_path = None
        should_save_video = (
            self.save_videos == "all"
            or (self.save_videos == "failures" and not success)
            or (self.save_videos == "sample10" and video_sample_selected)
        )
        if should_save_video:
            video_dir = self.output_dir / "videos" / record.suite / f"task_{record.task_id}"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = self.baseline.save_rollout_video(
                str(video_dir), replay_images, 0, success, instructions.clean, fps=24,
            )
        return {
            "suite": record.suite,
            "task_id": record.task_id,
            "official_id": record.official_id,
            "task_name": record.task_name,
            "category": record.category,
            "difficulty": record.difficulty,
            "task_language_raw": instructions.raw,
            "task_language_clean": instructions.clean,
            "task_language_base": instructions.base,
            "instruction_source": instructions.source,
            "success": bool(success),
            "status": "completed",
            "policy_seed": self.policy_seed,
            "env_seed": self.env_seed,
            "initial_state_index": 0,
            "num_steps_wait": self.args.num_steps_wait,
            "max_policy_steps": MAX_STEPS[record.suite],
            "video_path": video_path,
            "video_sample_selected": video_sample_selected,
        }
