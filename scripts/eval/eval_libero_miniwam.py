#!/usr/bin/env python
"""
eval_libero_miniwam.py

Evaluation script for MiniWAM on LIBERO simulation suites.
Loads a MiniWAM checkpoint folder and its frozen visual encoder (DINOv3, or the
Wan2.1 VAE for Wan-VAE checkpoints), reads the instruction embedding from the
checkpoint's cache (or encodes it online with UMT5-XXL), and rolls out policy
trials in the specified LIBERO environment.

Usage:
    python scripts/eval/eval_libero_miniwam.py \
        --checkpoint checkpoints/libero \
        --task_suite_name libero_spatial \
        --task_id 0 \
        --num_trials 50 \
        --gpu_id 0
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Setup sys.path to find miniwam and LIBERO
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

libero_path = Path(os.environ.get("MINIWAM_LIBERO_ROOT", project_root / "LIBERO")).expanduser()
if libero_path.exists():
    sys.path.insert(0, str(libero_path))

scripts_eval_path = project_root / "scripts" / "eval"
if scripts_eval_path.exists():
    sys.path.insert(0, str(scripts_eval_path))

from gpu_environment import assign_egl_device
from libero_eval_recovery import (
    atomic_write_json,
    load_recoverable_trials,
    quarantine_superseded_results,
    write_trial_progress,
)
# Suite step limits of the common LIBERO protocol, shared with the LIBERO-Plus evaluator.
from libero_plus_protocol import MAX_STEPS as COMMON_MAX_STEPS


def ensure_libero_config() -> None:
    """Create a non-interactive LIBERO config before importing libero modules."""
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", project_root / ".libero")).expanduser()
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return

    benchmark_root = libero_path / "libero" / "libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"assets: {benchmark_root / 'assets'}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"benchmark_root: {benchmark_root}",
                f"datasets: {benchmark_root.parent / 'datasets'}",
                f"init_states: {benchmark_root / 'init_files'}",
                "",
            ]
        ),
        encoding="utf-8",
    )


ensure_libero_config()

# Now import the modules
try:
    from miniwam.model import MiniWAM
    from libero_utils import (
        LIBERO_ENV_RESOLUTION,
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        invert_gripper_action,
        quat2axisangle,
        save_rollout_video,
    )
    from miniwam.models.wan_video_vae import WanVideoVAE
    from miniwam.models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
    from miniwam.prompting import (
        encode_instruction_features,
        load_cached_instruction_features,
        resolve_text_feature_cache,
    )
    from miniwam.proprio import normalize_proprio
    from miniwam.action_normalization import denormalize_clipped_actions
    from miniwam.statistics import get_action_bounds, select_statistics
    from miniwam.checkpointing import load_policy
    from miniwam.models.dinov3 import DINOv3FrameEncoder
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_libero_miniwam")

def parse_args():
    parser = argparse.ArgumentParser(description="MiniWAM LIBERO Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="MiniWAM checkpoint folder (e.g. outputs/libero_stage2/latest)")

    # LIBERO task parameters
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
                        help="LIBERO task suite name")
    parser.add_argument("--task_id", type=int, default=0,
                        help="Task ID/index within the suite")
    parser.add_argument("--num_trials", type=int, default=50,
                        help="Number of evaluation trials")

    # Model configuration overrides
    parser.add_argument("--vae_path", type=str, default="pretrained/vision/Wan2.1_VAE.pth",
                        help="Path to Wan 2.1 VAE checkpoint")
    parser.add_argument("--text_encoder_path", type=str, default="pretrained/language/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to UMT5-XXL text encoder checkpoint")
    parser.add_argument("--tokenizer_id", type=str, default="google/umt5-xxl",
                        help="HuggingFace tokenizer ID")
    parser.add_argument("--context_len", type=int, default=128,
                        help="Text context length")
    parser.add_argument(
        "--text_feature_cache", type=str, default=None,
        help="LIBERO instruction cache (default: libero_instruction_features.pt in the checkpoint folder)",
    )
    parser.add_argument(
        "--online_text_encoding", action="store_true",
        help="Ignore any discovered cache and encode the instruction with UMT5-XXL online.",
    )

    # Evaluation hyperparameters
    parser.add_argument("--replan_steps", type=int, default=None,
                        help="Re-planning/execution horizon (number of steps to execute before query)")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Number of denoising steps. Defaults to config inference.num_inference_steps")
    parser.add_argument("--num_steps_wait", type=int, default=10,
                        help="Initial no-op settling steps (common LIBERO protocol: 10)")
    parser.add_argument("--binarize_gripper", action=argparse.BooleanOptionalAction, default=True,
                        help="Binarize gripper actions to -1 (open) or 1 (close)")
    parser.add_argument("--action_horizon", type=int, default=None,
                        help="Action horizon (overrides config data.action_horizon if specified)")

    # System
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to run evaluation on")
    parser.add_argument("--output_dir", type=str, default="./eval_results/libero",
                        help="Base output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Policy seed, reset at every episode (common LIBERO protocol: 42)")
    parser.add_argument("--env_seed", type=int, default=0,
                        help="LIBERO environment seed (common LIBERO protocol: 0)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"],
                        help="Inference data type")

    return parser.parse_args()


# ── Model & Feature Loaders ──────────────────────────────────────────────────

def load_state_dict(path, device="cpu"):
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=device)
    else:
        return torch.load(path, map_location=device, weights_only=True)


def load_libero_init_states(task):
    """Load trusted local LIBERO init states under PyTorch 2.6+."""
    from libero.libero import get_libero_path

    init_states_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    return torch.load(init_states_path, weights_only=False)

def load_vae_model(vae_path, device, dtype):
    """Load Wan 2.1 VAE from checkpoint."""
    logger.info("Loading Wan 2.1 VAE from %s", vae_path)
    vae = WanVideoVAE(z_dim=16)
    state_dict = load_state_dict(vae_path, device="cpu")
    state_dict = WanVideoVAE.state_dict_converter().from_civitai(state_dict)
    vae.load_state_dict(state_dict, strict=True)
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    return vae

def load_text_encoder_model(text_encoder_path, tokenizer_id, context_len, device, dtype):
    """Load T5 text encoder and tokenizer."""
    logger.info("Loading UMT5-XXL text encoder from %s", text_encoder_path)
    text_encoder = WanTextEncoder()
    state_dict = load_state_dict(text_encoder_path, device="cpu")
    text_encoder.load_state_dict(state_dict, strict=True)
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()

    if not os.path.isdir(tokenizer_id):
        try:
            from huggingface_hub import snapshot_download
            resolved_id = snapshot_download(repo_id=tokenizer_id, local_files_only=True)
            logger.info("Resolved tokenizer_id '%s' to local path: %s", tokenizer_id, resolved_id)
            tokenizer_id = resolved_id
        except Exception as e:
            raise RuntimeError(f"Could not resolve tokenizer_id offline: {tokenizer_id}") from e

    tokenizer = HuggingfaceTokenizer(name=tokenizer_id, seq_len=context_len, clean="whitespace")
    return text_encoder, tokenizer

def encode_text(text_encoder, tokenizer, instruction, device, dtype):
    """Encode a text instruction using UMT5-XXL."""
    return encode_instruction_features(text_encoder, tokenizer, instruction, device, dtype)


# ── Observation Processing ────────────────────────────────────────────────────

def process_observation(obs: dict, device, dtype):
    """
    Extract LIBERO main and wrist views as separate VAE inputs.
    """
    imgs = get_libero_image(obs)

    def to_vae_input(rgb: np.ndarray) -> torch.Tensor:
        image_t = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_t = F.interpolate(image_t, size=(224, 224), mode="bilinear", align_corners=False)
        return (image_t.unsqueeze(2).to(device=device, dtype=dtype) * 2.0) - 1.0

    view_inputs = {
        "main": to_vae_input(imgs["image"]),
        "wrist": to_vae_input(imgs["wrist_image"]),
    }
    return view_inputs, imgs


def extract_libero_proprio(obs: dict) -> np.ndarray:
    """Return the raw 8-D pre-action state used by the LIBERO datasets."""
    state = np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        np.asarray(quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float32),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    ])
    if state.shape != (8,):
        raise ValueError(f"LIBERO proprio must have shape (8,), got {state.shape}.")
    return state


# ── Action Inference & Rollout ────────────────────────────────────────────────

def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: MiniWAM,
    vae,
    text_embeds: torch.Tensor,
    text_mask: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
    action_constant_mask: torch.Tensor,
    proprio_stats,
    args,
    device,
    dtype,
    episode_idx: int,
    max_steps: int,
) -> tuple[bool, list]:
    """Execute a single evaluation trial/episode in LIBERO."""
    env.reset()
    obs = env.set_init_state(initial_state)

    replay_images = []
    pending_actions = []

    t = 0
    done = False
    pbar = tqdm(total=max_steps + args.num_steps_wait, desc=f"Episode {episode_idx + 1}", leave=False)

    while t < max_steps + args.num_steps_wait:
        pbar.update(1)

        # 1. Initial settling steps
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        # 2. Get action predictions if pending action buffer is empty
        if len(pending_actions) == 0:
            # Process the observation at the native visual-encoder precision.
            visual_type = getattr(model, "visual_encoder_type", "wan_vae")
            observation_dtype = torch.float32 if visual_type == "dinov3" else dtype
            view_inputs, imgs = process_observation(obs, device, observation_dtype)
            if getattr(args, "capture_video", True):
                replay_images.append(imgs)

            with torch.no_grad():
                if visual_type == "dinov3":
                    z_t_main = vae.encode_for_miniwam(view_inputs["main"].squeeze(2)).unsqueeze(1)
                    z_t_wrist = vae.encode_for_miniwam(view_inputs["wrist"].squeeze(2)).unsqueeze(1)
                else:
                    z_t_main = vae.single_encode(view_inputs["main"], device=device).unsqueeze(1)
                    z_t_wrist = vae.single_encode(view_inputs["wrist"], device=device).unsqueeze(1)
                main_present = torch.ones((1, 1), dtype=torch.bool, device=device)
                wrist_present = torch.ones((1, 1), dtype=torch.bool, device=device)
                proprio = None
                if proprio_stats is not None:
                    proprio = torch.from_numpy(
                        normalize_proprio(extract_libero_proprio(obs), proprio_stats)
                    ).unsqueeze(0).to(device=device, dtype=dtype)

                pred_actions = model.infer(
                    z_t_main=z_t_main,
                    main_present=main_present,
                    z_t_wrist=z_t_wrist,
                    wrist_present=wrist_present,
                    text_embeds=text_embeds,
                    text_mask=text_mask,
                    proprio=proprio,
                    action_horizon=args.action_horizon,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                )

            # Denormalize predicted actions using the checkpoint's declared bounds.
            # pred_actions shape: [1, action_horizon, action_dim]
            pred_actions = pred_actions.squeeze(0).to(device=device, dtype=dtype) # [action_horizon, action_dim]

            # Extraction clips normalized targets to [-1, 1]. Project unconstrained
            # flow predictions back to that learned support before denormalization.
            actions_denorm = denormalize_clipped_actions(
                pred_actions, action_lower, action_upper, action_constant_mask,
            )
            actions_denorm = actions_denorm.cpu().numpy()

            # Gripper post-processing
            # The gripper is index 6 (last dimension).
            # The training dataset has gripper actions normalized to [0, 1] (0=close, 1=open).
            # We map this back to [-1, 1], and invert the sign to match the environment (1=close, -1=open).
            actions_denorm[..., -1] = actions_denorm[..., -1] * 2.0 - 1.0
            actions_denorm = invert_gripper_action(actions_denorm)
            # If binarize is enabled, binarize gripper based on sign
            if args.binarize_gripper:
                actions_denorm[..., -1] = np.sign(actions_denorm[..., -1])

            # Queue the actions for execution
            pending_actions = actions_denorm[:args.replan_steps].tolist()
        else:
            imgs = get_libero_image(obs)
            if getattr(args, "capture_video", True):
                replay_images.append(imgs)

        # 3. Step the environment
        act = pending_actions.pop(0)
        obs, _, done, _ = env.step(act)

        if done:
            break
        t += 1

    pbar.close()
    return bool(done), replay_images


# ── Main Entry ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    assign_egl_device(os.environ, args.gpu_id)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MiniWAM evaluation; torch.cuda.is_available() is false.")
    device = torch.device(f"cuda:{args.gpu_id}")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    logger.info("Evaluation device: %s | Dtype: %s", device, dtype)

    output_dir = Path(args.output_dir) / args.task_suite_name / f"task_{args.task_id}"
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the policy from its checkpoint folder.
    policy = load_policy(args.checkpoint, device=device, dtype=dtype)
    model, cfg = policy.model, policy.config
    model_cfg = cfg["model"]
    if args.action_horizon is None:
        args.action_horizon = int(cfg["data"]["action_horizon"])
    if args.replan_steps is None:
        args.replan_steps = int(cfg["inference"]["replan_steps"])
    if args.num_inference_steps is None:
        args.num_inference_steps = int(cfg["inference"]["num_inference_steps"])
    logger.info(
        "Checkpoint %s | action horizon %d | %d denoising steps | replan every %d steps",
        policy.path, args.action_horizon, args.num_inference_steps, args.replan_steps,
    )

    # 2. Load the frozen visual encoder for online observations.
    if model.visual_encoder_type == "dinov3":
        visual_cfg = model_cfg["visual_encoder"]
        vae = DINOv3FrameEncoder(
            architecture=visual_cfg.get("architecture", "vitb16"),
            pretraining=visual_cfg.get("pretraining", "lvd1689m"),
            model_id=visual_cfg["model_id"],
            revision=visual_cfg.get("revision"),
            input_size=visual_cfg.get("input_size", 224), frozen=True,
        ).to(device=device, dtype=torch.float32).eval()
    else:
        vae = load_vae_model(args.vae_path, device, dtype)

    # 3. Normalization statistics stored with the checkpoint.
    dataset_name = f"{args.task_suite_name}_no_noops_lerobot"
    ds_stats = select_statistics(policy.statistics, dataset_name, policy.path)
    action_lower_np, action_upper_np, action_constant_np, action_norm_mode = get_action_bounds(
        ds_stats, policy.path,
    )
    action_lower = torch.tensor(action_lower_np, device=device, dtype=torch.float32)
    action_upper = torch.tensor(action_upper_np, device=device, dtype=torch.float32)
    action_constant_mask = torch.tensor(action_constant_np, device=device, dtype=torch.bool)
    proprio_stats = None
    if model_cfg.get("context_encoder", {}).get("proprio_dim") not in (None, 0):
        proprio_stats = ds_stats["proprio"]
    logger.info(
        "Action normalization %s | lower=%s | upper=%s",
        action_norm_mode, action_lower_np, action_upper_np,
    )

    # 4. Initialize LIBERO benchmark environment
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite name: {args.task_suite_name}")

    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = load_libero_init_states(task)

    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.env_seed)
    logger.info("Initialized LIBERO task %d: '%s'", args.task_id, task_description)

    # 5. Resolve task text once. Cached mode avoids constructing UMT5 in every worker.
    if args.online_text_encoding and args.text_feature_cache:
        raise ValueError("--online_text_encoding cannot be combined with --text_feature_cache.")
    cache_path = None
    if not args.online_text_encoding:
        cache_path = resolve_text_feature_cache(args.text_feature_cache, policy.path)
    if cache_path is not None:
        expected_text_dim = int(model_cfg.get("context_encoder", {}).get("text_dim", 4096))
        context, mask = load_cached_instruction_features(
            cache_path,
            dataset_name,
            task_description,
            expected_context_len=args.context_len,
            expected_text_dim=expected_text_dim,
        )
        text_feature_source = str(cache_path)
        logger.info("Loaded cached instruction features from %s", cache_path)
    else:
        logger.info("No LIBERO text cache selected; encoding instruction online.")
        text_encoder, tokenizer = load_text_encoder_model(
            args.text_encoder_path, args.tokenizer_id, args.context_len, device, dtype,
        )
        context, mask = encode_text(text_encoder, tokenizer, task_description, device, dtype)
        del text_encoder, tokenizer
        torch.cuda.empty_cache()
        text_feature_source = "online_umt5"
    text_embeds = context.unsqueeze(0).to(device=device, dtype=dtype) # [1, L, D]
    text_mask = mask.unsqueeze(0).to(device=device, dtype=torch.bool) # [1, L]

    # Suite step limits shared with the common comparison protocol.
    max_steps = int(COMMON_MAX_STEPS[args.task_suite_name])

    # 6. Recover completed trials, then evaluate only missing trial IDs.
    evaluation_protocol = {
        "protocol": "common_libero_v1",
        "max_steps": int(max_steps),
        "env_seed": int(args.env_seed),
        "policy_seed": int(args.seed),
        "num_inference_steps": int(args.num_inference_steps),
        "replan_steps": int(args.replan_steps),
        "action_horizon": int(args.action_horizon),
        "dtype": args.dtype,
        "num_steps_wait": int(args.num_steps_wait),
        "binarize_gripper": bool(args.binarize_gripper),
    }
    trial_results_by_id, recovery_metadata = load_recoverable_trials(
        output_dir,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        checkpoint=args.checkpoint,
        num_trials=args.num_trials,
        base_seed=args.seed,
        required_protocol=evaluation_protocol,
    )
    if recovery_metadata.get("ignored_incompatible_progress") or recovery_metadata.get("skipped_video_trials"):
        # Move the old progress and videos aside; otherwise a later resume would
        # import those videos next to the new protocol's progress.
        superseded = quarantine_superseded_results(output_dir)
        logger.warning(
            "Existing results in %s were recorded under a different evaluation protocol "
            "or without one; they were moved to %s and are not resumed.",
            output_dir, superseded,
        )
        trial_results_by_id = {}
    write_trial_progress(
        output_dir,
        trial_results_by_id,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        checkpoint=args.checkpoint,
        num_trials=args.num_trials,
        base_seed=args.seed,
        evaluation_protocol=evaluation_protocol,
    )
    duplicate_ids = recovery_metadata["duplicate_trial_ids"]
    if duplicate_ids:
        logger.warning(
            "Duplicate rollout videos for trial IDs: %s (latest video wins).",
            duplicate_ids,
        )
    missing_trial_ids = [
        trial_idx for trial_idx in range(args.num_trials)
        if trial_idx not in trial_results_by_id
    ]
    logger.info(
        "Evaluation recovery: %d/%d trials complete; %d missing.",
        len(trial_results_by_id), args.num_trials, len(missing_trial_ids),
    )

    for trial_idx in missing_trial_ids:
        # Recovery identity only; the policy seed is args.seed for every episode.
        trial_seed = args.seed + trial_idx
        # Reset host RNGs to the policy seed at every episode, as the common
        # protocol's model server does.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        success, replay_images = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            vae=vae,
            text_embeds=text_embeds,
            text_mask=text_mask,
            action_lower=action_lower,
            action_upper=action_upper,
            action_constant_mask=action_constant_mask,
            proprio_stats=proprio_stats,
            args=args,
            device=device,
            dtype=dtype,
            episode_idx=trial_idx,
            max_steps=max_steps,
        )

        # Save rollout video
        save_rollout_video(
            rollout_dir=str(video_dir),
            rollout_images=replay_images,
            idx=trial_idx,
            success=success,
            task_description=task_description,
            fps=24,
        )

        # Persist only after the rollout video has been closed successfully. If the
        # process dies between these operations, the next run imports the MP4 name.
        trial_results_by_id[trial_idx] = {
            "trial_idx": trial_idx,
            "success": bool(success),
            "seed": trial_seed,
            "source": "evaluation",
        }
        write_trial_progress(
            output_dir,
            trial_results_by_id,
            task_suite_name=args.task_suite_name,
            task_id=args.task_id,
            checkpoint=args.checkpoint,
            num_trials=args.num_trials,
            base_seed=args.seed,
            evaluation_protocol=evaluation_protocol,
        )
        success_count = sum(result["success"] for result in trial_results_by_id.values())
        logger.info(
            "Trial %d/%d: Success = %s | Recovered progress = %d/%d | Success Rate = %.1f%%",
            trial_idx + 1, args.num_trials, success, len(trial_results_by_id), args.num_trials,
            (success_count / len(trial_results_by_id)) * 100,
        )

    trial_results = [trial_results_by_id[index] for index in range(args.num_trials)]
    success_count = sum(result["success"] for result in trial_results)
    final_success_rate = (success_count / args.num_trials) * 100
    logger.info("=========================================")
    logger.info("Task Evaluation Complete!")
    logger.info("Task: %s", task_description)
    logger.info("Success Rate: %d/%d (%.1f%%)", success_count, args.num_trials, final_success_rate)
    logger.info("=========================================")

    # Save final results summary json
    summary = {
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": task_description,
        "text_feature_source": text_feature_source,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "num_trials": args.num_trials,
        "base_seed": args.seed,
        "evaluation_protocol": evaluation_protocol,
        "success_count": success_count,
        "success_rate": final_success_rate,
        "trials": trial_results,
    }
    atomic_write_json(output_dir / "summary.json", summary)

    logger.info("Results summary saved to %s", output_dir / "summary.json")
    env.close()

if __name__ == "__main__":
    main()
