# Ablations

Every ablation in the paper is the default LIBERO recipe plus one or more
overlays from this folder. Pass the same overlays to both stages, except the
Stage-2-only overlays [`no_cotrain.yaml`](no_cotrain.yaml) and [`fresh_policy.yaml`](fresh_policy.yaml), which go on
Stage 2 alone:

```bash
python scripts/train.py --config configs/libero/stage1.yaml <overlays>
python scripts/train.py --config configs/libero/stage2.yaml <overlays>
```

A stacked run is named after its files, e.g.
`outputs/libero_stage1__grid2x2_d64__recon_only`, and Stage 2 loads the
matching Stage-1 run automatically. Every ablation arm starts Stage 2 from a
seeded policy; only the PRISM encoder comes from the arm's own Stage 1.
Evaluate on LIBERO and on the fixed LIBERO-Plus-2K subset
([evaluation](../../docs/evaluation.md#libero-plus-2k)).

| Overlay | Effect |
| --- | --- |
| [`capacity/grid2x2_d64.yaml`](capacity/grid2x2_d64.yaml) | pool the PRISM grid to 2×2 locations, 64 channels |
| [`capacity/grid1x1_d16.yaml`](capacity/grid1x1_d16.yaml) | pool to 1×1, 16 channels |
| [`objective/recon_only.yaml`](objective/recon_only.yaml) | Stage 1 trains the encoder by feature reconstruction only |
| [`objective/inverse_only.yaml`](objective/inverse_only.yaml) | Stage 1 uses inverse dynamics only (no decoder) |
| [`no_cotrain.yaml`](no_cotrain.yaml) | Stage 2 sets the world-token loss weight to 0 |
| [`fresh_policy.yaml`](fresh_policy.yaml) | Stage 2 of the default recipe starts from a seeded policy (the reference arm) |

[`libero/`](libero/) and [`robotwin/`](robotwin/) hold full stage configs rather than overlays.

## Table 1: world-modeling targets

Evaluated on LIBERO and LIBERO-Plus-2K.

| Features | Target | Stage 1 | Stage 2 |
| --- | --- | --- | --- |
| DINOv3 | PRISM | [`libero/stage1.yaml`](../libero/stage1.yaml) | [`libero/stage2.yaml`](../libero/stage2.yaml) + [`ablations/fresh_policy.yaml`](fresh_policy.yaml) |
| DINOv3 | native co-train | — | [`ablations/libero/native_dinov3.yaml`](libero/native_dinov3.yaml) |
| DINOv3 | w/o co-train | [`libero/stage1.yaml`](../libero/stage1.yaml) | [`libero/stage2.yaml`](../libero/stage2.yaml) + [`ablations/no_cotrain.yaml`](no_cotrain.yaml) |
| Wan2.1-VAE | PRISM | [`ablations/libero/wan_vae_stage1.yaml`](libero/wan_vae_stage1.yaml) | [`ablations/libero/wan_vae_stage2.yaml`](libero/wan_vae_stage2.yaml) |
| Wan2.1-VAE | native co-train | — | [`ablations/libero/native_wan_vae.yaml`](libero/native_wan_vae.yaml) |
| Wan2.1-VAE | w/o co-train | [`ablations/libero/wan_vae_stage1.yaml`](libero/wan_vae_stage1.yaml) | [`ablations/libero/wan_vae_stage2.yaml`](libero/wan_vae_stage2.yaml) + [`ablations/no_cotrain.yaml`](no_cotrain.yaml) |

Paths are relative to `configs/`. A cell with two files is one run: pass both
to `--config`, stage config first, e.g.
`--config configs/libero/stage2.yaml configs/ablations/no_cotrain.yaml`.

The native co-training controls have no PRISM encoder: one stage jointly
denoises the actions and all native future features (16 DINOv3 patch grids, or
4 Wan-VAE latent frames).

## Table 4: PRISM objective and capacity

Evaluated on LIBERO-Plus-2K. Each cell is the default LIBERO recipe plus the
overlays below, passed to both stages; [`fresh_policy.yaml`](fresh_policy.yaml) goes on Stage 2 only.

| Cell | Overlays |
| --- | --- |
| 4×4 Both | [`fresh_policy.yaml`](fresh_policy.yaml) (Stage 2 only) |
| 4×4 Recon. / Inv. dyn. | [`objective/recon_only.yaml`](objective/recon_only.yaml) / [`objective/inverse_only.yaml`](objective/inverse_only.yaml) |
| 2×2 Both | [`capacity/grid2x2_d64.yaml`](capacity/grid2x2_d64.yaml) |
| 2×2 Recon. / Inv. dyn. | [`capacity/grid2x2_d64.yaml`](capacity/grid2x2_d64.yaml) + [`objective/recon_only.yaml`](objective/recon_only.yaml) / [`objective/inverse_only.yaml`](objective/inverse_only.yaml) |
| 1×1 Both | [`capacity/grid1x1_d16.yaml`](capacity/grid1x1_d16.yaml) |
| 1×1 Recon. / Inv. dyn. | [`capacity/grid1x1_d16.yaml`](capacity/grid1x1_d16.yaml) + [`objective/recon_only.yaml`](objective/recon_only.yaml) / [`objective/inverse_only.yaml`](objective/inverse_only.yaml) |

## RoboTwin controls

[`ablations/robotwin/`](robotwin/) holds RoboTwin versions of the Wan-VAE and
native-feature controls (horizon 16). [`robotwin/stage2.yaml`](../robotwin/stage2.yaml) +
[`ablations/no_cotrain.yaml`](no_cotrain.yaml) gives the RoboTwin model without co-training.
