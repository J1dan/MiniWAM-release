#!/bin/bash
# Optional, validated host-driver overrides. Correctly configured hosts need none.
miniwam_runtime_role=${1:?runtime role must be simulator or policy}
if [[ "${miniwam_runtime_role}" != simulator && "${miniwam_runtime_role}" != policy ]]; then
    echo "[MiniWAM][ERROR] Unknown NVIDIA runtime role: ${miniwam_runtime_role}" >&2
    return 1
fi

if [[ -n "${MINIWAM_NVIDIA_LIBRARY_PATH:-}" ]]; then
    if [[ ":${MINIWAM_NVIDIA_LIBRARY_PATH}:" == *::* ]]; then
        echo '[MiniWAM][ERROR] MINIWAM_NVIDIA_LIBRARY_PATH contains an empty entry.' >&2
        return 1
    fi
    IFS=: read -r -a miniwam_nvidia_dirs <<< "${MINIWAM_NVIDIA_LIBRARY_PATH}"
    for miniwam_nvidia_dir in "${miniwam_nvidia_dirs[@]}"; do
        if [[ -z "${miniwam_nvidia_dir}" || ! -d "${miniwam_nvidia_dir}" ]]; then
            echo "[MiniWAM][ERROR] Invalid MINIWAM_NVIDIA_LIBRARY_PATH entry: ${miniwam_nvidia_dir:-<empty>}" >&2
            return 1
        fi
    done
    export LD_LIBRARY_PATH="${MINIWAM_NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "[MiniWAM][WARN] Using explicit NVIDIA library override: ${MINIWAM_NVIDIA_LIBRARY_PATH}" >&2
fi

# Apply an explicit repair before testing the NVML library used by nvidia-smi.
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo '[MiniWAM][ERROR] NVIDIA driver is not usable in this process.' >&2
    echo '[MiniWAM][ERROR] If driver packages were just upgraded, reboot the host before evaluation.' >&2
    return 1
fi

if [[ "${miniwam_runtime_role}" == simulator && -n "${MINIWAM_VULKAN_ICD_FILENAMES:-}" ]]; then
    if [[ ":${MINIWAM_VULKAN_ICD_FILENAMES}:" == *::* ]]; then
        echo '[MiniWAM][ERROR] MINIWAM_VULKAN_ICD_FILENAMES contains an empty entry.' >&2
        return 1
    fi
    IFS=: read -r -a miniwam_vulkan_files <<< "${MINIWAM_VULKAN_ICD_FILENAMES}"
    for miniwam_vulkan_file in "${miniwam_vulkan_files[@]}"; do
        if [[ -z "${miniwam_vulkan_file}" || ! -f "${miniwam_vulkan_file}" ]]; then
            echo "[MiniWAM][ERROR] Invalid MINIWAM_VULKAN_ICD_FILENAMES entry: ${miniwam_vulkan_file:-<empty>}" >&2
            return 1
        fi
    done
    export VK_ICD_FILENAMES="${MINIWAM_VULKAN_ICD_FILENAMES}"
    echo "[MiniWAM][WARN] Using explicit Vulkan ICD override: ${MINIWAM_VULKAN_ICD_FILENAMES}" >&2
fi

unset miniwam_runtime_role miniwam_nvidia_dirs miniwam_nvidia_dir miniwam_vulkan_files miniwam_vulkan_file
