"""Persistent MiniWAM runtime shared by the XPolicyLab RoboTwin adapter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import os

import hashlib

import numpy as np
import torch

from miniwam.action_normalization import denormalize_actions
from miniwam.checkpointing import CONFIG_FILE, STATISTICS_FILE, WEIGHTS_FILE, load_policy
from miniwam.image_preprocessing import dino_rgb_uint8_to_bchw, wan_vae_rgb_uint8_to_bchw
from miniwam.models.dinov3 import DINOv3FrameEncoder
from miniwam.proprio import PROPRIO_ZSCORE, normalize_proprio
from miniwam.robotwin import (
    ROBOTWIN_ACTION_DIM_INFO,
    ROBOTWIN_DATASET_KEY,
    assert_robotwin_data_config,
    assert_robotwin_joint_contract,
    resolve_robotwin_replan_steps,
    validate_robotwin_training_statistics,
)
from miniwam.statistics import ACTION_ZSCORE, select_statistics, validate_action_statistics
from miniwam.robotwin_text_cache import RoboTwinTextCache
from miniwam.vae_loading import load_wan_video_vae


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MiniWAMRoboTwinRuntime:
    """Load all policy assets once and infer normalized 14-D action chunks."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        project_root: str | Path | None = None,
        device: str = "cuda",
        mixed_precision: str = "bf16",
        action_horizon: int | None = None,
        replan_steps: int | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
    ):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for MiniWAM inference but is unavailable.")
        if mixed_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported mixed_precision={mixed_precision!r}.")
        self.device = torch.device(device)
        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[mixed_precision]

        policy = load_policy(self.checkpoint_path, device=self.device, dtype=self.dtype)
        self.model, self.config, self.protocol = policy.model, policy.config, policy.runtime
        self.model_config = self.config["model"]
        self.data_config = self.config["data"]
        if self.data_config.get("dataset_name") != ROBOTWIN_DATASET_KEY:
            raise ValueError(
                f"MiniWAM checkpoint config dataset_name must be {ROBOTWIN_DATASET_KEY!r}."
            )
        assert_robotwin_data_config(self.data_config)
        assert_robotwin_joint_contract(
            action_type=self.data_config.get("action_type"),
            robot_action_dim_info=ROBOTWIN_ACTION_DIM_INFO,
            action_dim=self.data_config.get("action_dim"),
            proprio_dim=self.model_config.get("context_encoder", {}).get("proprio_dim"),
            action_representation=self.data_config.get("action_representation"),
            action_transform=self.data_config.get("action_transform"),
            delta_action_mask=self.data_config.get("delta_action_mask"),
        )

        self.statistics = select_statistics(
            policy.statistics, ROBOTWIN_DATASET_KEY, self.checkpoint_path,
        )
        validate_robotwin_training_statistics(self.statistics)
        if validate_action_statistics(self.statistics, self.checkpoint_path) != ACTION_ZSCORE:
            raise ValueError("RoboTwin action statistics must use global zscore_clip.")
        if self.statistics.get("proprio", {}).get("normalization") != PROPRIO_ZSCORE:
            raise ValueError("RoboTwin proprio statistics must use global zscore_clip.")

        visual = self.model_config["visual_encoder"]
        feature_contract = self.data_config.get("feature_contract")
        if feature_contract is None:
            raise ValueError("RoboTwin checkpoints record the feature-extraction contract in "
                             "data.feature_contract; this config has none.")
        self.visual_encoder_type = str(visual.get("type"))
        self.vae_resolution = None
        if self.visual_encoder_type == "dinov3":
            self.visual_encoder = DINOv3FrameEncoder(
                architecture=visual.get("architecture", "vitb16"),
                pretraining=visual.get("pretraining", "lvd1689m"),
                model_id=visual["model_id"],
                revision=visual.get("revision"),
                input_size=visual.get("input_size", 224),
                frozen=True,
            ).to(device=self.device, dtype=torch.float32).eval()
            actual_revision = self.visual_encoder.resolved_revision
            expected_revision = visual.get("resolved_revision", visual.get("revision"))
            if str(actual_revision) != str(expected_revision):
                raise ValueError(
                    f"DINO resolved revision mismatch: runtime={actual_revision!r}, "
                    f"training={expected_revision!r}."
                )
            if feature_contract.get("dinov3") != self.visual_encoder.feature_metadata():
                raise ValueError("Runtime DINO processor/features differ from training extraction.")
        elif self.visual_encoder_type == "wan_vae":
            vae_path = self._resolve_project_path(
                os.environ.get("MINIWAM_VAE_PATH")
                or visual.get("vae_path")
                or self.data_config.get("vae_path")
                or self.config.get("pretrained", {}).get("vae_path")
            )
            self.visual_encoder = load_wan_video_vae(
                vae_path, device=self.device, dtype=self.dtype,
            )
            if "wan_vae" not in feature_contract:
                raise ValueError("RoboTwin Wan-VAE checkpoints require a bound feature contract.")
            expected_vae = feature_contract["wan_vae"]
            from miniwam.robotwin_features import wan_vae_feature_contract
            self.vae_resolution = tuple(expected_vae["resize"]["size_hw"])
            actual_vae = wan_vae_feature_contract(
                vae_path, self.vae_resolution, self.dtype, self.protocol.future_rgb_horizon,
            )
            if actual_vae != expected_vae:
                raise ValueError("Runtime Wan-VAE preprocessing/features differ from training extraction.")
        else:
            raise ValueError(
                "RoboTwin MiniWAM deployment requires visual_encoder.type to be "
                f"'dinov3' or 'wan_vae', got {self.visual_encoder_type!r}."
            )
        text_path = (
            os.environ.get("MINIWAM_TEXT_ENCODER_PATH")
            or self.data_config.get("text_encoder_path")
            or self.config.get("pretrained", {}).get("text_encoder_path")
        )
        tokenizer_id = (
            self.data_config.get("tokenizer_id")
            or self.config.get("pretrained", {}).get("tokenizer_id")
        )
        context_len = int(
            self.data_config.get("context_len")
            or self.config.get("pretrained", {}).get("context_len", 128)
        )
        evaluation_protocol = os.environ.get('MINIWAM_EVAL_PROTOCOL')
        if evaluation_protocol not in {'frozen', 'official'}:
            raise ValueError('MINIWAM_EVAL_PROTOCOL must be resolved by the RoboTwin scheduler.')
        miss_policy = os.environ.get('MINIWAM_TEXT_CACHE_MISS')
        if evaluation_protocol == 'frozen':
            if miss_policy != 'error' or os.environ.get('MINIWAM_CASE_MODE') != 'replay':
                raise ValueError('Frozen evaluation must use replay and cache-only text features.')
            if not os.environ.get('MINIWAM_FROZEN_TEXT_INDEX'):
                raise ValueError('Frozen replay requires MINIWAM_FROZEN_TEXT_INDEX.')
            # Replay only reads cached embeddings: check them against the encoder
            # recorded at training time, so the encoder file is not needed.
            encoder_sha256 = feature_contract['text']['encoding']['encoder_sha256']
        else:
            text_path = self._resolve_project_path(text_path)
            encoder_sha256 = None
            if miss_policy != 'build':
                raise ValueError('Official evaluation must explicitly build missing text features.')
            if (os.environ.get('MINIWAM_CASE_MODE') or
                    os.environ.get('MINIWAM_FROZEN_TEXT_INDEX') or
                    os.environ.get('MINIWAM_EVAL_MANIFEST')):
                raise ValueError('Official evaluation cannot use frozen-case assets.')
        expected_tokenizer_sha = feature_contract['text'].get('tokenizer_backend_sha256')
        self.instruction_cache = RoboTwinTextCache(
            os.environ.get('MINIWAM_TEXT_CACHE_DIR', str(self.project_root / 'datasets/robotwin_eval_text_cache')),
            text_path, tokenizer_id, context_len, self.dtype,
            miss_policy=miss_policy,
            expected_tokenizer_backend_sha256=(
                expected_tokenizer_sha if evaluation_protocol == 'official' else None
            ),
            encoder_sha256=encoder_sha256,
        )
        from miniwam.robotwin_features import validate_runtime_text_contract
        validate_runtime_text_contract(
            feature_contract["text"], self.instruction_cache,
            require_frozen=evaluation_protocol == 'frozen',
        )
        if evaluation_protocol == 'official':
            print('[MiniWAM] Preloading official online text encoder before server readiness.',
                  flush=True)
            self.instruction_cache.ensure_encoder_loaded()
        print(f'[MiniWAM] evaluation_protocol={evaluation_protocol}', flush=True)
        print(f'[MiniWAM] text_cache_miss={miss_policy}', flush=True)
        print(f'[MiniWAM] text_encoder_sha256={self.instruction_cache.contract["encoder_sha256"]}',
              flush=True)
        self.text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        inference = self.config.get("inference", {})
        self.action_horizon = int(action_horizon or self.protocol.action_horizon)
        if self.action_horizon != self.protocol.action_horizon:
            raise ValueError(
                f"Runtime action_horizon={self.action_horizon} must equal checkpoint "
                f"horizon={self.protocol.action_horizon}."
            )
        self.replan_steps = resolve_robotwin_replan_steps(
            self.action_horizon,
            replan_steps if replan_steps is not None else inference.get("replan_steps"),
        )
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None else inference.get("num_inference_steps", 20)
        )
        self.seed = int(seed if seed is not None else inference.get("seed", 42))
        self.inference_calls = 0

        if os.environ.get('MINIWAM_CASE_MODE') == 'replay':
            from scripts.eval.robotwin_runtime_identity import bind_frozen_runtime
            source_files = sorted((self.project_root/'miniwam').rglob('*.py'))
            source_files += [Path(__file__), self.project_root/'integrations/xpolicylab/policy/MiniWAM/model.py']
            identity = dict(schema_version=1,
                checkpoint_sha256=file_sha256(self.checkpoint_path / WEIGHTS_FILE),
                dataset_stats_sha256=file_sha256(self.checkpoint_path / STATISTICS_FILE),
                training_config_sha256=file_sha256(self.checkpoint_path / CONFIG_FILE),
                eval_manifest_sha256=file_sha256(Path(os.environ['MINIWAM_EVAL_MANIFEST'])),
                text_cache_sha256=file_sha256(Path(os.environ['MINIWAM_FROZEN_TEXT_INDEX'])),
                action_horizon=self.action_horizon, replan_steps=self.replan_steps,
                num_inference_steps=self.num_inference_steps,
                seed=self.seed, mixed_precision=mixed_precision,
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(self.device),
                renderer_profile=os.environ.get('MINIWAM_ROBOTWIN_RENDER_PROFILE', 'upstream'),
                policy_source_sha256={str(p.resolve().relative_to(self.project_root)): file_sha256(p)
                                      for p in source_files})
            bind_frozen_runtime(os.environ['MINIWAM_CASE_OUTPUT'], identity)

    def _resolve_project_path(self, value: str | Path | None) -> Path:
        if value is None:
            raise ValueError("Required MiniWAM asset path is absent from the training config.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MiniWAM asset does not exist: {path}")
        return path

    def _text_features(self, instructions: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        contexts, masks = [], []
        for instruction in instructions:
            key = instruction
            if not key:
                raise ValueError("RoboTwin instruction must be non-empty.")
            if key not in self.text_cache:
                self.text_cache[key] = self.instruction_cache.get(key)
            context, mask = self.text_cache[key]
            contexts.append(context)
            masks.append(mask)
        return (
            torch.stack(contexts).to(device=self.device, dtype=self.dtype),
            torch.stack(masks).to(device=self.device, dtype=torch.bool),
        )

    @torch.no_grad()
    def infer_normalized(
        self,
        *,
        head_rgb: Sequence[np.ndarray],
        left_wrist_rgb: Sequence[np.ndarray],
        right_wrist_rgb: Sequence[np.ndarray],
        proprio: np.ndarray,
        instructions: Sequence[str],
    ) -> np.ndarray:
        batch_size = len(instructions)
        if batch_size == 0:
            raise ValueError("RoboTwin inference batch must be non-empty.")
        if not (len(head_rgb) == len(left_wrist_rgb) == len(right_wrist_rgb) == batch_size):
            raise ValueError("RoboTwin camera and instruction batch sizes differ.")

        def encode(images: Sequence[np.ndarray], camera: str) -> torch.Tensor:
            source = f"runtime obs vision/{camera}/color"
            stacked = np.stack([np.asarray(image) for image in images])
            if self.visual_encoder_type == "dinov3":
                rgb = dino_rgb_uint8_to_bchw(stacked, source=source).to(device=self.device)
                return self.visual_encoder.encode_for_miniwam(rgb)
            rgb = wan_vae_rgb_uint8_to_bchw(
                stacked, resolution=self.vae_resolution, source=source,
            ).unsqueeze(2).to(device=self.device, dtype=self.dtype)
            return self.visual_encoder.single_encode(rgb, device=self.device)

        # Slot order is fixed; both wrist slots pass through one shared wrist E_phi.
        z_t_main = encode(head_rgb, "cam_head").unsqueeze(1)
        z_t_wrist = torch.stack(
            (
                encode(left_wrist_rgb, "cam_left_wrist"),
                encode(right_wrist_rgb, "cam_right_wrist"),
            ),
            dim=1,
        )
        main_present = torch.ones((batch_size, 1), dtype=torch.bool, device=self.device)
        wrist_present = torch.ones((batch_size, 2), dtype=torch.bool, device=self.device)
        proprio = np.asarray(proprio, dtype=np.float32)
        if proprio.shape != (batch_size, 14) or not np.isfinite(proprio).all():
            raise ValueError(f"Runtime packed proprio must be finite [{batch_size},14], got {proprio.shape}.")
        proprio_norm = torch.from_numpy(
            normalize_proprio(proprio, self.statistics["proprio"]),
        ).to(device=self.device, dtype=self.dtype)
        text_embeds, text_mask = self._text_features(instructions)
        predicted = self.model.infer(
            z_t_main=z_t_main,
            main_present=main_present,
            z_t_wrist=z_t_wrist,
            wrist_present=wrist_present,
            text_embeds=text_embeds,
            text_mask=text_mask,
            action_horizon=self.action_horizon,
            proprio=proprio_norm,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed + self.inference_calls,
        )
        self.inference_calls += 1
        result = np.asarray(predicted, dtype=np.float32)
        expected = (batch_size, self.action_horizon, 14)
        if result.shape != expected or not np.isfinite(result).all():
            raise ValueError(f"MiniWAM normalized action output must be finite {expected}, got {result.shape}.")
        return np.clip(result, -5.0, 5.0)

    def infer_actions(self, **kwargs) -> np.ndarray:
        normalized = self.infer_normalized(**kwargs)
        return np.asarray(denormalize_actions(normalized, self.statistics), dtype=np.float32)

    def reset(self) -> None:
        self.inference_calls = 0
