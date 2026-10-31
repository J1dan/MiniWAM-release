#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# Ray-traced SAPIEN evaluation and a saturated training process on the same
# device can stop making progress without an exception. Warn, but do not reject
# deliberate jobs_per_gpu sharing. The byte-level watchdog below is the hard
# safety net for a genuinely wedged evaluator.
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_utilization=$(nvidia-smi --id="${env_gpu_id}" \
        --query-gpu=utilization.gpu --format=csv,noheader,nounits \
        2>/dev/null | head -n 1 | tr -d ' ' || true)
    if [[ "${gpu_utilization}" =~ ^[0-9]+$ ]] && (( gpu_utilization >= 80 )); then
        echo "[MiniWAM][WARN] GPU ${env_gpu_id} is already ${gpu_utilization}% utilized before" >&2
        echo "[MiniWAM][WARN] simulator startup. Use a GPU without concurrent training for" >&2
        echo "[MiniWAM][WARN] stable ray-traced leaderboard evaluation." >&2
    fi
fi

inactivity_timeout="${MINIWAM_EVAL_INACTIVITY_TIMEOUT_S:-900}"
watchdog="${SCRIPT_DIR}/compat/run_with_inactivity_timeout.py"
source "${SCRIPT_DIR}/compat/configure_nvidia_runtime.sh" simulator
source "${SCRIPT_DIR}/compat/configure_renderer_runtime.sh" "${env_gpu_id}"

# SAPIEN catches some OIDN loader exceptions internally and can otherwise keep
# running with noisy path-traced RGB. Execute the selected denoiser for every
# isolated task job and require OIDN's own device/version diagnostics before
# the official evaluator starts. This validates but does not alter upstream.
if [[ "${eval_env_conda_env}" == */* ]]; then
    renderer_conda_args=(--prefix "$(cd "${eval_env_conda_env}" && pwd)")
else
    renderer_conda_args=(--name "${eval_env_conda_env}")
fi
set +e
renderer_preflight_output=$(env OIDN_VERBOSE=4 CUDA_VISIBLE_DEVICES="${env_gpu_id}" \
    conda run --no-capture-output "${renderer_conda_args[@]}" \
    python "${SCRIPT_DIR}/compat/validate_renderer.py" 2>&1)
renderer_preflight_status=$?
set -e
printf '%s\n' "${renderer_preflight_output}"
if [[ "${renderer_preflight_status}" != 0 ]] || \
   grep -Eqi 'OIDN Error:|unsupported device type: CUDA|Error: invalid handle|unsupported compute capability|OIDN CUDA device error' \
       <<<"${renderer_preflight_output}" || \
   ! grep -Fq "Loaded module: '${MINIWAM_EXPECTED_OIDN_CUDA_LIBRARY}'" \
       <<<"${renderer_preflight_output}" || \
   ! grep -Fq "Intel(R) Open Image Denoise ${MINIWAM_EXPECTED_OIDN_VERSION}" \
       <<<"${renderer_preflight_output}" || \
   ! grep -Fq '[MiniWAM] renderer_preflight=passed' \
       <<<"${renderer_preflight_output}"; then
    echo '[MiniWAM][ERROR] Renderer preflight failed; evaluation will not start.' >&2
    exit 125
fi

# Use the XPolicyLab client dispatcher. Its simulator implementation,
# utils/run_sim_env_client.sh, invokes RoboTwin/scripts/eval_policy.sh; the
# multitask scheduler's ROBOTWIN_EVAL_ARGS_FILE is inherited by that process.
exec python3 "${watchdog}" --timeout-seconds "${inactivity_timeout}" -- \
    env CUDA_VISIBLE_DEVICES="${env_gpu_id}" \
    bash "${UTILS_DIR}/setup_env_client.sh" \
        "${UTILS_DIR}" \
        "${yaml_file}" \
        "${eval_env_conda_env}" \
        "${policy_server_port}" \
        "${bench_name}" \
        "${task_name}" \
        "${env_cfg_type}" \
        "${policy_name}" \
        "${additional_info}" \
        "${BENCH_ROOT}" \
        "${seed}" \
        "${env_gpu_id}" \
        "${policy_server_ip}"
