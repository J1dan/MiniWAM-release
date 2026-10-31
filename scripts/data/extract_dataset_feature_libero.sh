#!/usr/bin/env bash
# Four-suite extraction with shared statistics; one GPU by default.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${PYTHON:-python}"
VISUAL_ENCODER="${VISUAL_ENCODER:-dinov3}"
DATA_ROOT="${LIBERO_DATA_ROOT:-datasets/LIBERO-fastwam}"
case "$VISUAL_ENCODER" in
  dinov3) FEATURES_ROOT="${FEATURES_ROOT:-datasets/dino_features_libero_aggregate_minmax}" ;;
  wan_vae) FEATURES_ROOT="${FEATURES_ROOT:-datasets/vae_features_libero_aggregate_minmax}" ;;
  *) echo 'VISUAL_ENCODER must be dinov3 or wan_vae' >&2; exit 2 ;;
esac
DATASETS=(libero_10_no_noops_lerobot libero_spatial_no_noops_lerobot libero_object_no_noops_lerobot libero_goal_no_noops_lerobot)
STATS_MIX="$(IFS=,; echo "${DATASETS[*]}")"
for suite in "${DATASETS[@]}"; do
  "$PYTHON" scripts/data/extract_dataset_feature_libero.py \
    --data_root "$DATA_ROOT" --data_mix "$suite" --stats_data_mix "$STATS_MIX" \
    --visual_encoder "$VISUAL_ENCODER" --action_normalization minmax \
    --output_dir "$FEATURES_ROOT/$suite" --future_rgb_horizon 16 \
    --per_view_resolution 224 224 --num_main_slots 1 --num_wrist_slots 1 \
    --main_camera_keys image --wrist_camera_keys wrist_image \
    --device cuda --dtype bf16 --batch_size "${BATCH_SIZE:-8}"
done
"$PYTHON" scripts/data/build_libero_text_feature_cache.py --features_root "$FEATURES_ROOT"
