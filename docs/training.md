# Training

Prepare features first ([data](data.md)). Every run merges `configs/base.yaml`,
one stage config and, for ablations, any number of overlays:

```bash
python scripts/train.py --config <stage config> [overlay ...] [options]
```

| Benchmark | Stage 1 | Stage 2 |
| --- | --- | --- |
| LIBERO | `configs/libero/stage1.yaml`: 16,260 steps (15 epochs) | `configs/libero/stage2.yaml`: 54,200 steps (50 epochs) |
| RoboTwin 2.0 | `configs/robotwin/stage1.yaml`: 12,864 steps (6 epochs) | `configs/robotwin/stage2.yaml`: 53,600 steps (25 epochs) |

Both use the 0.25B model, AdamW with peak learning rate 5e-4, cosine decay with
5% warmup, BF16, seed 42 and a global batch of 256. LIBERO uses action and
visual horizons of 16; RoboTwin uses 24. RoboTwin lengths are set in epochs, so
its step count can differ by one with the GPU layout.

## The two stages

**Stage 1** trains the PRISM encoder together with an inverse-dynamics action
loss and a feature-reconstruction loss (weight 1); world tokens are clean
inputs. **Stage 2** loads the Stage-1 checkpoint named by
`checkpoint.stage1_path` (default `outputs/<benchmark>_stage1/latest`), freezes
the encoder and jointly denoises world tokens and actions.

`checkpoint.policy_init` chooses how the Stage-2 policy starts:

- `fresh` (default): load only the frozen PRISM encoder from Stage 1.
- `stage1` (benchmark models): copy all Stage-1 weights; only the new
  proprioception projection is freshly initialized.

Override the Stage-1 folder with `--stage1_checkpoint outputs/<run>/step_016260`.

## Batch size and GPUs

```text
GPUs × per-device batch (--batch_size) × --gradient_accumulation_steps = 256
```

On one GPU the configs use a per-device batch of 256 (32 × 8 accumulation for
the native-feature ablations). To fit a smaller GPU,
trade batch for accumulation, e.g. `--batch_size 16 --gradient_accumulation_steps 16`.
On several GPUs launch with `torchrun`; `--fsdp` shards gradients and optimizer
state, otherwise DDP is used:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --config configs/libero/stage1.yaml --fsdp --batch_size 64
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --config configs/robotwin/stage1.yaml --fsdp --batch_size 16 --gradient_accumulation_steps 4
```

The trainer refuses a combination that does not give the configured global batch.

## Outputs, resuming and logging

A run writes `outputs/<run>/step_<N>/` checkpoint folders and a `latest` link.
Each folder holds `model.safetensors`, `config.yaml`, `statistics.json`,
`metadata.json` and `training_state.pt` (optimizer, scheduler and KL warmup
progress); the first four are all that evaluation needs. Resume with the same
command plus `--resume outputs/<run>/step_<N>`.

Logs go to the console. Add `--wandb --wandb_project <project>` after
`pip install -e '.[logging]'` to log to Weights & Biases.

## Ablations

Ablations stack overlays from `configs/ablations/` on the stage configs. Apply
the same overlays to both stages; Stage 2 then finds its matching Stage-1 run
automatically. The Stage-2-only overlays `no_cotrain.yaml` and
`fresh_policy.yaml` go on Stage 2 alone, which then uses the default Stage-1 run. See [configs/ablations](../configs/ablations/README.md).
