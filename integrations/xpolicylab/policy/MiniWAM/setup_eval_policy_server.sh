#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
MINIWAM_ROOT="${MINIWAM_ROOT:-$(cd "$(readlink -f "${SCRIPT_DIR}")/../../../.." && pwd)}"
CKPT_ROOT="${MINIWAM_CKPT_ROOT:-${MINIWAM_ROOT}/outputs}"
yaml_file="${SCRIPT_DIR}/deploy.yml"

resolve_checkpoint_candidate() {
    local requested=$1
    if [[ "${requested}" == /* ]]; then
        printf '%s\n' "${requested}"
    elif [[ -e "${MINIWAM_ROOT}/${requested}" ]]; then
        printf '%s\n' "${MINIWAM_ROOT}/${requested}"
    elif [[ -e "${CKPT_ROOT}/${requested}" ]]; then
        printf '%s\n' "${CKPT_ROOT}/${requested}"
    elif [[ "${requested}" == outputs/* ]]; then
        printf '%s\n' "${MINIWAM_ROOT}/${requested}"
    else
        printf '%s\n' "${CKPT_ROOT}/${requested}"
    fi
}

if [[ -n "${MINIWAM_CHECKPOINT_PATH:-}" ]]; then
    checkpoint_path=$(resolve_checkpoint_candidate "${MINIWAM_CHECKPOINT_PATH}")
else
    checkpoint_path=$(resolve_checkpoint_candidate "${ckpt_name}")
fi
# Accept a checkpoint folder, a run folder (its latest step) or a run's parent.
if [[ ! -f "${checkpoint_path}/metadata.json" ]]; then
    if [[ -f "${checkpoint_path}/latest/metadata.json" ]]; then
        checkpoint_path="${checkpoint_path}/latest"
    else
        checkpoint_path=$(find "${checkpoint_path}" -mindepth 2 -maxdepth 2 -name metadata.json \
            -path '*/step_*' 2>/dev/null | sort -V | tail -n 1 | xargs -r dirname)
    fi
fi
if [[ ! -f "${checkpoint_path:-}/metadata.json" ]]; then
    echo "[MiniWAM][ERROR] Checkpoint folder not found: ${checkpoint_path:-<empty>}" >&2
    exit 1
fi
# Pin the immutable step folder rather than a moving latest link.
checkpoint_path=$(readlink -f "${checkpoint_path}")
dataset_stats_path="${checkpoint_path}/statistics.json"
checkpoint_sha256=$(sha256sum "${checkpoint_path}/model.safetensors" | awk '{print $1}')
if [[ -n "${MINIWAM_EXPECTED_CHECKPOINT_SHA256:-}" && \
      "${checkpoint_sha256}" != "${MINIWAM_EXPECTED_CHECKPOINT_SHA256}" ]]; then
    echo "[MiniWAM][ERROR] Pinned checkpoint contents changed during evaluation." >&2
    exit 1
fi
statistics_sha256=$(sha256sum "${dataset_stats_path}" | awk '{print $1}')
echo "[MiniWAM] checkpoint_path=${checkpoint_path}"
echo "[MiniWAM] checkpoint_sha256=${checkpoint_sha256}"
echo "[MiniWAM] dataset_stats_path=${dataset_stats_path}"
echo "[MiniWAM] dataset_stats_sha256=${statistics_sha256}"

# Resolve and fingerprint artifacts without importing either conda environment.
# This is useful for validating evaluator launch commands before a long run.
if [[ "${MINIWAM_RESOLVE_ONLY:-0}" == "1" ]]; then
    exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"
if ! python - <<'PY'
import msgpack
import msgpack_numpy
import pydantic
import websockets
PY
then
    echo "[MiniWAM][ERROR] XPolicyLab websocket dependencies are missing from '${policy_conda_env}'." >&2
    echo "Install them with: bash ${SCRIPT_DIR}/install.sh ${policy_conda_env}" >&2
    exit 1
fi
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
if [[ "${action_dim}" != "14" || "${action_type}" != "joint" ]]; then
    echo "[MiniWAM][ERROR] RoboTwin requires joint action_dim=14; got ${action_type}/${action_dim}." >&2
    exit 1
fi
source "${SCRIPT_DIR}/compat/configure_nvidia_runtime.sh" policy

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${MINIWAM_ROOT}:${BENCH_ROOT}:${PYTHONPATH:-}" \
    python -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name=MiniWAM \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            project_root="${MINIWAM_ROOT}" \
            checkpoint_path="${checkpoint_path}" \
            ${MINIWAM_REPLAN_STEPS:+replan_steps="${MINIWAM_REPLAN_STEPS}"}
