#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=$9
eval_env_conda_env=${10}

# Conda records prefix environments by canonical path. Accept convenient paths
# such as RoboTwin/../.robotwin-env from the public multitask launcher.
if [[ "${eval_env_conda_env}" == */* ]] && [[ -d "${eval_env_conda_env}" ]]; then
    eval_env_conda_env=$(cd "${eval_env_conda_env}" && pwd)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
export MINIWAM_ROOT="${MINIWAM_ROOT:-$(cd "$(readlink -f "${SCRIPT_DIR}")/../../../.." && pwd)}"

pin=$(python3 "${SCRIPT_DIR}/compat/prepare_job.py" "$@")
export MINIWAM_CHECKPOINT_PATH="${pin%%$'\n'*}"
export MINIWAM_EXPECTED_CHECKPOINT_SHA256="${pin##*$'\n'}"
echo "[MiniWAM] Isolated task worker; pinned checkpoint=${MINIWAM_CHECKPOINT_PATH}"

# Stop the policy server when this task attempt ends, keeping its exit status.
stop_policy_server() {
    local evaluation_status=$?
    trap - EXIT
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    exit "${evaluation_status}"
}
trap stop_policy_server EXIT

# The scheduler resolves configs/robotwin/eval_*.yaml into this internal switch.
# Refuse an implicit default so official and frozen results can never mix.
: "${MINIWAM_EVAL_PROTOCOL:?The multitask scheduler must select miniwam.evaluation_protocol}"
case "${MINIWAM_EVAL_PROTOCOL}" in
    frozen)
        if [[ "${MINIWAM_CASE_MODE:-}" != replay ]]; then
            echo '[MiniWAM][ERROR] Frozen evaluation requires internal replay mode.' >&2
            exit 1
        fi
        : "${MINIWAM_EVAL_MANIFEST:?Frozen evaluation requires a sealed manifest}"
        : "${MINIWAM_FROZEN_TEXT_INDEX:?Frozen evaluation requires frozen_index.json}"
        if [[ "${MINIWAM_TEXT_CACHE_MISS:-}" != error ]]; then
            echo '[MiniWAM][ERROR] Frozen evaluation must be cache-only.' >&2
            exit 1
        fi
        ;;
    official)
        if [[ -n "${MINIWAM_CASE_MODE:-}" || -n "${MINIWAM_EVAL_MANIFEST:-}" || \
              -n "${MINIWAM_FROZEN_TEXT_INDEX:-}" || -n "${MINIWAM_CASE_OUTPUT:-}" ]]; then
            echo '[MiniWAM][ERROR] Official evaluation was contaminated by frozen-case state.' >&2
            exit 1
        fi
        if [[ "${MINIWAM_TEXT_CACHE_MISS:-}" != build ]]; then
            echo '[MiniWAM][ERROR] Official evaluation requires explicit online text encoding.' >&2
            exit 1
        fi
        ;;
    *)
        echo "[MiniWAM][ERROR] Unsupported evaluation protocol: ${MINIWAM_EVAL_PROTOCOL}" >&2
        exit 1
        ;;
esac
echo "[MiniWAM] evaluation_protocol=${MINIWAM_EVAL_PROTOCOL}"
project_root=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve().parents[3])' "${SCRIPT_DIR}")
if [[ "${MINIWAM_EVAL_PROTOCOL}" == frozen ]]; then
    manifest_identity=$(sha256sum -- "${MINIWAM_EVAL_MANIFEST}")
    manifest_identity=${manifest_identity%% *}
    default_case_output="${project_root}/eval_results/robotwin/frozen_case_ledgers/${MINIWAM_EXPECTED_CHECKPOINT_SHA256}/${manifest_identity}"
    if [[ "${MINIWAM_ROBOTWIN_RENDER_PROFILE:-upstream}" != upstream ]]; then
        default_case_output="${default_case_output}/renderer-${MINIWAM_ROBOTWIN_RENDER_PROFILE}"
    fi
    export MINIWAM_CASE_OUTPUT="${MINIWAM_CASE_OUTPUT:-${default_case_output}}"
fi

# Exit 10 means an incomplete/missing task must run; all other errors are fatal.
resume_status=0
python3 "${SCRIPT_DIR}/compat/resume_task.py" "$@" || resume_status=$?
if [[ "${resume_status}" == "0" ]]; then
    exit 0
elif [[ "${resume_status}" != "10" ]]; then
    exit "${resume_status}"
fi

require_conda_env() {
    local env_name=$1
    local role=$2
    if ! (
        # Conda compiler activation hooks are not nounset-safe.
        set +u
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${env_name}"
    ) >/dev/null 2>&1; then
        echo "[MiniWAM][ERROR] ${role} Conda environment not found: ${env_name}" >&2
        if [[ "${role}" == "RoboTwin simulator" ]]; then
            echo "Run from the MiniWAM root:" >&2
            echo "  bash scripts/setup_robotwin_eval_env.sh --download-assets" >&2
            echo "Then pass: --eval-env-conda-env \"\$PWD/.robotwin-env\"" >&2
        fi
        exit 1
    fi
}

require_conda_env "${policy_conda_env}" "MiniWAM policy"
require_conda_env "${eval_env_conda_env}" "RoboTwin simulator"

if ! (
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${eval_env_conda_env}"
    python - <<'PY'
import torch
import torch._C
import mplib
import sapien
PY
) >/dev/null 2>&1; then
    echo "[MiniWAM][ERROR] RoboTwin simulator environment is incomplete: ${eval_env_conda_env}" >&2
    echo "PyTorch, SAPIEN, and MPLib must import before evaluation." >&2
    echo "Run: bash scripts/setup_robotwin_eval_env.sh" >&2
    exit 1
fi

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip=localhost
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" \
    "${policy_server_port}" "${policy_server_ip}" &
SERVER_PID=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
    "${additional_info}" "${policy_server_port}" "${policy_server_ip}"
echo "[MiniWAM] task_completed=true"
