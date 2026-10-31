# Data preparation

Training reads pre-extracted features: frozen DINOv3 patch tokens (or Wan2.1-VAE
latents) for every frame and camera, UMT5 instruction embeddings, actions,
proprioception and normalization statistics. Evaluating a released checkpoint
needs none of this.

## LIBERO

MiniWAM trains on the four LIBERO suites (Spatial, Object, Goal, LIBERO-10) as
converted for FastWAM: [`yuanty/LIBERO-fastwam`](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
(CC BY 4.0), LeRobot 2.1 archives at revision `ee018b9`, 1,712 episodes and
277,713 frames in total.

Download, verify, then extract:

```bash
mkdir -p datasets/LIBERO-fastwam && cd datasets/LIBERO-fastwam
for suite in libero_10 libero_spatial libero_object libero_goal; do
  hf download yuanty/LIBERO-fastwam ${suite}_no_noops_lerobot.tar.gz --repo-type dataset \
    --revision ee018b997c430bb12b5bf3c892d744798c5a2f91 --local-dir .
done
sha256sum -c <<'EOF'
eb5af5f46d9348f8ba0146cf90de718ce102f261348983c0fd2df08d50dd1059  libero_10_no_noops_lerobot.tar.gz
7c374be75133e1a6779a0e07b9452c16294777be577b89553f1cf5350eafd8f9  libero_spatial_no_noops_lerobot.tar.gz
732405d5deaf3019c244bacd3f26c9617f25c871750586c9968747b6cfefedda  libero_object_no_noops_lerobot.tar.gz
a21ae10171535585fb43e6405d9efa09ff38ef34689e4176428ca005af3a39ea  libero_goal_no_noops_lerobot.tar.gz
EOF
for suite in libero_10 libero_spatial libero_object libero_goal; do
  tar -xzf ${suite}_no_noops_lerobot.tar.gz
done
cd ../..
```

Extract features for all four suites and build the instruction cache:

```bash
VISUAL_ENCODER=dinov3 bash scripts/data/extract_dataset_feature_libero.sh
```

This writes `datasets/dino_features_libero_aggregate_minmax/<suite>/` and
`libero_instruction_features.pt`. Actions and proprioception are normalized to
[-1, 1] with min/max statistics computed over all four suites together.
DINOv3 runs in FP32 and features are stored in BF16, one 14×14 patch grid per
frame and camera (main and wrist), so any future horizon can be windowed from
the same cache. For the Wan-VAE ablations use `VISUAL_ENCODER=wan_vae`; the
output goes to `datasets/vae_features_libero_aggregate_minmax/`. Set
`LIBERO_DATA_ROOT` if the archives are elsewhere. Re-running skips episodes that
are already complete.

## RoboTwin 2.0

RoboTwin trains on the official `demo_clean` data: 50 tasks × 50 demonstrations
(episodes `0000000`–`0000049`), three cameras (head, left wrist, right wrist),
14-D absolute joint actions and proprioception, normalized with global z-score
statistics clipped at ±5.

```bash
bash scripts/setup_robotwin.sh
bash scripts/download_robotwin_training_data.sh
python scripts/data/extract_dataset_feature_robotwin.py --robotwin-root RoboTwin \
  --output-dir datasets/dino_features_robotwin2_demo_clean_50x50 --audit-only
python scripts/data/extract_dataset_feature_robotwin.py --robotwin-root RoboTwin \
  --output-dir datasets/dino_features_robotwin2_demo_clean_50x50
```

The audit checks that exactly the 2,500 selected episodes are present (548,985
transitions). Run the extractor with `--help` for Wan-VAE features and sharding
across GPUs.
