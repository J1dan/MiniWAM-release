#!/usr/bin/env python
"""Train MiniWAM on pre-extracted DINOv3 or Wan-VAE features.

Usage:
    # Stage 1: learn the PRISM encoder
    python scripts/train.py --config configs/libero/stage1.yaml

    # Stage 2: freeze the encoder and train the world-action model
    python scripts/train.py --config configs/libero/stage2.yaml

    # Ablations stack overlays on a stage config (use the same overlays in both stages)
    python scripts/train.py \
        --config configs/libero/stage1.yaml configs/ablations/capacity/grid2x2_d64.yaml

    # Several GPUs: DDP by default, or FSDP
    torchrun --standalone --nproc_per_node=4 scripts/train.py \
        --config configs/libero/stage1.yaml --fsdp --batch_size 64
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniwam.checkpointing import (
    build_cpu_model_with_seed,
    build_model_protocol,
    initialize_stage2,
    load_checkpoint,
    load_training_state,
    restore_model_forward_step,
    save_checkpoint,
    validate_model_protocol,
)
from miniwam.config import load_training_config
from miniwam.dataset import MiniWAMCollator, MiniWAMPreextractedDataset
from miniwam.factory import build_miniwam_model, is_native_config
from miniwam.protocol import MiniWAMRuntimeProtocol, resolve_miniwam_protocol
from miniwam.robotwin import (
    ROBOTWIN_ACTION_DIM_INFO,
    ROBOTWIN_DATASET_KEY,
    assert_robotwin_data_config,
    assert_robotwin_joint_contract,
)
from miniwam.robotwin_features import bind_training_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniWAM")
    parser.add_argument("--config", nargs="+", default=["configs/libero/stage1.yaml"],
                        help="Stage config, optionally followed by ablation overlays")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--features_root", type=str, default=None,
                        help="Comma-separated feature roots (overrides data.features_root)")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Stage-1 checkpoint folder (overrides checkpoint.stage1_path)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint folder of this run to resume from")
    parser.add_argument("--fsdp", action="store_true",
                        help="With several GPUs, shard gradients and optimizer state (FSDP) instead of DDP")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=str, default=None,
                        help="Integer steps (e.g. 50000) or epochs (e.g. '10epoch')")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=str, default=None,
                        help="Integer steps or epochs (e.g. '5epoch')")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    """Load the base, stage and overlay configs, then apply CLI overrides."""
    cfg = load_training_config(args.base_config, args.config)
    training = cfg.setdefault("training", {})
    overrides = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
    }
    training.update({key: value for key, value in overrides.items() if value is not None})
    if args.stage1_checkpoint is not None:
        cfg.setdefault("checkpoint", {})["stage1_path"] = args.stage1_checkpoint
    return cfg


def get_lr_scheduler(optimizer, total_steps, cfg):
    """Linear warmup followed by cosine decay, or a constant rate."""
    training_cfg = cfg.get("training", {})
    scheduler_type = training_cfg.get("lr_scheduler_type", "cosine")
    warmup_steps = int(total_steps * training_cfg.get("warmup_ratio", 0.05))
    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    if scheduler_type == "constant":
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(optimizer, lambda step: 1.0)
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def resolve_per_device_batch_size(training_cfg, world_size: int) -> int:
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    configured = int(training_cfg.get("per_device_batch_size", 64))
    global_batch = training_cfg.get("global_batch_size")
    if global_batch is None:
        return configured
    global_batch = int(global_batch)
    denominator = world_size * grad_accum
    if global_batch % denominator:
        raise ValueError(
            f"global_batch_size={global_batch} is not divisible by world_size({world_size}) "
            f"* gradient_accumulation_steps({grad_accum})."
        )
    resolved = global_batch // denominator
    if configured != resolved:
        raise ValueError(
            f"per_device_batch_size={configured} yields global batch "
            f"{configured * denominator}, expected {global_batch}."
        )
    return resolved


def parse_interval(value):
    """Return (steps, epochs) for an integer step count or an '<N>epoch' string."""
    if isinstance(value, str) and "epoch" in value.lower():
        return None, int(value.lower().replace("epoch", "").strip() or "1")
    return int(value), None


def train(args, cfg, runtime_protocol: MiniWAMRuntimeProtocol):
    training_cfg = cfg.setdefault("training", {})
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    stage = int(cfg["stage"])
    native = is_native_config(model_cfg)
    if data_cfg.get("dataset_name") == ROBOTWIN_DATASET_KEY:
        assert_robotwin_data_config(data_cfg)
        assert_robotwin_joint_contract(
            action_type=data_cfg.get("action_type"),
            robot_action_dim_info=ROBOTWIN_ACTION_DIM_INFO,
            action_dim=data_cfg.get("action_dim"),
            proprio_dim=data_cfg.get("proprio_dim"),
            action_representation=data_cfg.get("action_representation"),
            action_transform=data_cfg.get("action_transform"),
            delta_action_mask=data_cfg.get("delta_action_mask"),
        )

    # ── Devices ──
    import torch.distributed as dist
    is_dist = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MiniWAM training.")
    if is_dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device("cuda")
    is_rank0 = rank == 0
    if not is_rank0:
        logging.getLogger().setLevel(logging.WARNING)
    use_fsdp = is_dist and args.fsdp

    seed = training_cfg.get("seed", 42)
    torch.manual_seed(seed + rank)
    dtype = torch.bfloat16 if training_cfg.get("mixed_precision") == "bf16" else torch.float32
    output_dir = Path(training_cfg.get("output_dir", "outputs/miniwam"))
    if is_rank0:
        output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = resolve_per_device_batch_size(training_cfg, world_size)
    grad_accum = training_cfg.get("gradient_accumulation_steps", 1)
    learning_rate = training_cfg.get("learning_rate", 1e-4)
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
    log_interval = training_cfg.get("log_interval", 10)

    features_root = args.features_root or data_cfg.get("features_root")
    if not features_root:
        raise ValueError("Training requires data.features_root or --features_root.")
    bind_training_features(cfg, features_root)
    logger.info("Features: %s", features_root)

    # ── Model ──
    logger.info("Building %s model for Stage %d", model_cfg["architecture"], stage)
    model = build_cpu_model_with_seed(lambda: build_miniwam_model(model_cfg, runtime_protocol), seed)
    if stage == 2 and not native:
        checkpoint_cfg = cfg["checkpoint"]
        if not checkpoint_cfg.get("stage1_path"):
            raise ValueError("Stage 2 needs checkpoint.stage1_path or --stage1_checkpoint.")
        logger.info(
            "Initializing Stage 2 from %s (policy_init=%s)",
            checkpoint_cfg["stage1_path"], checkpoint_cfg["policy_init"],
        )
        initialize_stage2(
            model, load_checkpoint(checkpoint_cfg["stage1_path"]),
            checkpoint_cfg["policy_init"], runtime_protocol,
        )

    loss_cfg = cfg.get("loss", {})
    model.lambda_action = loss_cfg.get("lambda_action", 1.0)
    if native:
        model.lambda_video = loss_cfg.get("lambda_video", 1.0)
    else:
        model.lambda_latent = loss_cfg.get("lambda_latent", 0.0 if stage == 1 else 1.0)
        model.kl_beta = loss_cfg.get("kl_beta", 1e-4 if stage == 1 else 0.0)
        model.kl_warmup_steps = loss_cfg.get("kl_warmup_steps", 1000)
    model = model.to(device)
    raw_model = model
    model_protocol = build_model_protocol(raw_model, runtime_protocol)
    if int(data_cfg.get("action_dim", 7)) != int(raw_model.action_dit.action_dim):
        raise ValueError("data.action_dim and model.action_dit.action_dim must match.")

    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

        # The whole model is one FSDP unit: the Mixture-of-Transformers runs the
        # expert blocks' parameters directly, so they must be gathered together.
        precision = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
        model = FSDP(
            model,
            auto_wrap_policy=None,
            mixed_precision=MixedPrecision(
                param_dtype=precision, reduce_dtype=precision, buffer_dtype=precision,
            ),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )
        dist.barrier()
    elif is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Params: %.1fM total, %.1fM trainable", total_params / 1e6, train_params / 1e6)

    # ── Logging ──
    logging_cfg = cfg.get("logging", {})
    use_wandb = is_rank0 and (args.wandb or logging_cfg.get("use_wandb", False))
    if use_wandb:
        import wandb

        wandb_name = args.wandb_run_name or logging_cfg.get("wandb_run_name", f"stage{stage}")
        wandb_kwargs = {
            "project": args.wandb_project or logging_cfg.get("wandb_project", "miniwam"),
            "entity": logging_cfg.get("wandb_entity"),
            "name": wandb_name,
            "config": cfg,
        }
        if args.resume:
            wandb_kwargs.update(resume="allow", id=wandb_name.replace("/", "_").replace(" ", "_"))
        wandb.init(**wandb_kwargs)

    # ── Data ──
    dataset = MiniWAMPreextractedDataset(
        features_root=features_root,
        runtime_protocol=runtime_protocol,
        action_dim=data_cfg.get("action_dim", 7),
        proprio_dim=model_cfg.get("context_encoder", {}).get("proprio_dim"),
        visual_encoder_config=model_cfg["visual_encoder"],
    )
    if is_rank0:
        (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    sampler = (
        torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
        )
        if is_dist else None
    )
    shuffle_generator = torch.Generator() if sampler is None else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=shuffle_generator,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=MiniWAMCollator(),
        drop_last=True,
    )
    logger.info("Dataset: %d samples, %d batches/epoch", len(dataset), len(dataloader))

    batches_per_epoch = len(dataloader)
    max_steps, num_epochs = parse_interval(training_cfg.get("max_steps", 50000))
    if max_steps is None:
        max_steps = max((num_epochs * batches_per_epoch) // grad_accum, 1)
    else:
        num_epochs = (max_steps * grad_accum + batches_per_epoch - 1) // batches_per_epoch
    save_interval_steps, save_every_n_epochs = parse_interval(
        training_cfg.get("save_interval", "1epoch")
    )
    logger.info("Training for %d steps (%d epochs)", max_steps, num_epochs)

    # ── Optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=training_cfg.get("weight_decay", 0.0),
        betas=(0.9, 0.999),
    )
    scheduler = get_lr_scheduler(optimizer, max_steps, cfg)

    def full_state():
        """Gather the model and optimizer state on rank 0."""
        if use_fsdp:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, policy):
                return model.state_dict(), FSDP.optim_state_dict(model, optimizer)
        return raw_model.state_dict(), optimizer.state_dict()

    last_saved_step = None

    def save(step):
        nonlocal last_saved_step
        if step == last_saved_step:  # e.g. the final save right after an interval save
            return
        last_saved_step = step
        if not (use_fsdp or is_rank0):
            return
        model_state, optimizer_state = full_state()
        if not is_rank0:
            return
        step_dir = save_checkpoint(
            output_dir / f"step_{step:06d}",
            state_dict=model_state,
            config=cfg,
            statistics=dataset.statistics,
            global_step=step,
            model_protocol=model_protocol,
            training_state={
                "optimizer": optimizer_state,
                "scheduler": scheduler.state_dict(),
                "model_forward_step": int(getattr(raw_model, "_global_step", step * grad_accum)),
            },
        )
        latest = output_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(step_dir.name)
        logger.info("Saved checkpoint to %s", step_dir)

    # ── Resume ──
    global_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        validate_model_protocol(checkpoint.model_protocol, model_protocol, context="resume")
        state = load_training_state(args.resume)
        if use_fsdp:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, policy):
                model.load_state_dict(checkpoint.state_dict, strict=True)
                optimizer.load_state_dict(
                    FSDP.optim_state_dict_to_load(model, optimizer, state["optimizer"])
                )
        else:
            raw_model.load_state_dict(checkpoint.state_dict, strict=True)
            optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(checkpoint.metadata["global_step"])
        restore_model_forward_step(
            raw_model, state, global_step=global_step, gradient_accumulation_steps=grad_accum,
        )
        logger.info("Resumed from step %d", global_step)
    forward_step = global_step * grad_accum
    # Continue the sampler's epoch sequence and skip the batches already consumed
    # in the interrupted epoch, so a resumed run does not replay its data order.
    resume_epochs_done, resume_skip_batches = resume_position(global_step, grad_accum, batches_per_epoch)

    # ── Training loop ──
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    loss_dict_accum = {}
    steps_in_window = 0
    epoch = resume_epochs_done

    from tqdm import tqdm

    with tqdm(
        total=max_steps,
        initial=global_step,
        desc=f"Stage {stage} | epoch 0/{num_epochs}",
        dynamic_ncols=True,
        disable=not is_rank0,
    ) as pbar:
        while global_step < max_steps and epoch < num_epochs:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            else:
                # Same per-epoch order as the distributed sampler's convention, so a
                # single-GPU resume replays exactly the batches it skips.
                shuffle_generator.manual_seed(seed + epoch)
            pbar.set_description(f"Stage {stage} | epoch {epoch}/{num_epochs}")

            for batch_idx, batch in enumerate(dataloader):
                if global_step >= max_steps:
                    break
                if resume_skip_batches:
                    resume_skip_batches -= 1
                    continue

                features = {
                    key: batch[key].to(device=device, dtype=dtype)
                    for key in ("z_t_main", "z_fwd_main", "z_t_wrist", "z_fwd_wrist", "text_embeds")
                }
                with torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    loss, loss_dict = model(
                        actions=batch["actions"].to(device=device, dtype=dtype),
                        text_mask=batch["text_mask"].to(device=device),
                        proprio=(
                            batch["proprio"].to(device=device, dtype=dtype)
                            if "proprio" in batch else None
                        ),
                        action_is_pad=batch["action_is_pad"].to(device=device),
                        main_present=batch["main_present"].to(device=device),
                        wrist_present=batch["wrist_present"].to(device=device),
                        **features,
                    )

                (loss / grad_accum).backward()
                loss_accum += loss.item()
                steps_in_window += 1
                for key, value in loss_dict.items():
                    loss_dict_accum[key] = loss_dict_accum.get(key, 0.0) + value

                if (forward_step + 1) % grad_accum == 0 or grad_accum == 1:
                    if max_grad_norm > 0:
                        if use_fsdp:
                            model.clip_grad_norm_(max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    pbar.update(1)

                    if global_step % log_interval == 0 and steps_in_window > 0:
                        if use_wandb:
                            import wandb

                            metrics = {
                                "train/loss": loss_accum / steps_in_window,
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/epoch": epoch - 1 + (batch_idx + 1) / batches_per_epoch,
                                "train/step": global_step,
                            }
                            metrics.update({
                                f"train/{key}": value / steps_in_window
                                for key, value in loss_dict_accum.items() if key != "loss_total"
                            })
                            wandb.log(metrics, step=global_step)
                        loss_accum = 0.0
                        loss_dict_accum = {}
                        steps_in_window = 0

                    if save_interval_steps and global_step % save_interval_steps == 0:
                        save(global_step)

                forward_step += 1
                if is_rank0 and steps_in_window > 0:
                    postfix = {
                        "loss": f"{loss_accum / steps_in_window:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                    postfix.update({
                        key: f"{value / steps_in_window:.4f}"
                        for key, value in loss_dict_accum.items() if key != "loss_total"
                    })
                    pbar.set_postfix(postfix, refresh=False)

            if save_every_n_epochs and epoch % save_every_n_epochs == 0:
                save(global_step)
                logger.info("End of epoch %d/%d, step %d/%d", epoch, num_epochs, global_step, max_steps)

    save(global_step)
    logger.info("Training complete: %d/%d steps, %d/%d epochs", global_step, max_steps, epoch, num_epochs)
    if use_wandb:
        import wandb

        wandb.finish()
    if is_dist:
        dist.destroy_process_group()


def resume_position(global_step, grad_accum, batches_per_epoch):
    """Return (completed epochs, batches to skip) for an exact data-order resume."""
    forward_step = int(global_step) * int(grad_accum)
    return forward_step // int(batches_per_epoch), forward_step % int(batches_per_epoch)


def main():
    args = parse_args()
    cfg, runtime_protocol = resolve_miniwam_protocol(load_config(args))
    train(args, cfg, runtime_protocol)


if __name__ == "__main__":
    main()
