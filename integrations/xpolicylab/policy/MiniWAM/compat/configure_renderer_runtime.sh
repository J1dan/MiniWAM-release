#!/bin/bash
# Opt-in renderer compatibility. Omitting the profile preserves upstream SAPIEN.
miniwam_renderer_gpu=${1:?simulator GPU id is required}
miniwam_renderer_profile=${MINIWAM_ROBOTWIN_RENDER_PROFILE:-upstream}

case "${miniwam_renderer_profile}" in
    upstream)
        if [[ -n "${MINIWAM_OIDN_ROOT:-}" ]]; then
            echo '[MiniWAM][ERROR] MINIWAM_OIDN_ROOT is set while renderer profile is upstream.' >&2
            return 1
        fi
        export MINIWAM_EXPECTED_OIDN_VERSION=2.0.1
        export MINIWAM_EXPECTED_OIDN_CUDA_LIBRARY=libOpenImageDenoise_device_cuda.so.2.0.1
        echo '[MiniWAM] renderer_profile=upstream (SAPIEN bundled OIDN)'
        ;;
    oidn_2_3_3_blackwell)
        : "${MINIWAM_OIDN_ROOT:?oidn_2_3_3_blackwell requires MINIWAM_OIDN_ROOT}"
        miniwam_oidn_lib="${MINIWAM_OIDN_ROOT}/lib"
        miniwam_oidn_core="${miniwam_oidn_lib}/libOpenImageDenoise_core.so.2.3.3"
        miniwam_oidn_api="${miniwam_oidn_lib}/libOpenImageDenoise.so.2.3.3"
        miniwam_oidn_cuda="${miniwam_oidn_lib}/libOpenImageDenoise_device_cuda.so.2.3.3"
        miniwam_expected_hashes=(
            eadb2437d6ac4a88e5746980495ba1091f7d14745e3000b82a9382093d22f351
            9ac9dcd18318cae879efb01c14cff72c2c4275428a15e5ab6e94c8ff8a43c4dc
            12b3ff87cfa4b6a90481b757f5a56ee69380a45e3dcaeadf8dd22cb73d4f714a
        )
        miniwam_oidn_files=("${miniwam_oidn_core}" "${miniwam_oidn_api}" "${miniwam_oidn_cuda}")
        for miniwam_index in 0 1 2; do
            if [[ ! -f "${miniwam_oidn_files[${miniwam_index}]}" ]]; then
                echo "[MiniWAM][ERROR] Missing pinned OIDN library: ${miniwam_oidn_files[${miniwam_index}]}" >&2
                return 1
            fi
            miniwam_actual_hash=$(sha256sum "${miniwam_oidn_files[${miniwam_index}]}" | awk '{print $1}')
            if [[ "${miniwam_actual_hash}" != "${miniwam_expected_hashes[${miniwam_index}]}" ]]; then
                echo "[MiniWAM][ERROR] OIDN library hash mismatch: ${miniwam_oidn_files[${miniwam_index}]}" >&2
                return 1
            fi
        done
        miniwam_gpu_capability=$(nvidia-smi --id="${miniwam_renderer_gpu}" \
            --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 || true)
        if [[ ! "${miniwam_gpu_capability%%.*}" =~ ^[0-9]+$ ]] || \
           (( ${miniwam_gpu_capability%%.*} < 12 )); then
            echo "[MiniWAM][ERROR] oidn_2_3_3_blackwell requires an sm_120+ GPU; got ${miniwam_gpu_capability:-unknown}." >&2
            return 1
        fi
        export LD_LIBRARY_PATH="${miniwam_oidn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export LD_PRELOAD="${miniwam_oidn_core}:${miniwam_oidn_api}${LD_PRELOAD:+:${LD_PRELOAD}}"
        export MINIWAM_EXPECTED_OIDN_VERSION=2.3.3
        export MINIWAM_EXPECTED_OIDN_CUDA_LIBRARY=libOpenImageDenoise_device_cuda.so.2.3.3
        echo "[MiniWAM] renderer_profile=${miniwam_renderer_profile} gpu_sm=${miniwam_gpu_capability}"
        echo "[MiniWAM] oidn_version=2.3.3 oidn_root=${MINIWAM_OIDN_ROOT}"
        ;;
    *)
        echo "[MiniWAM][ERROR] Unsupported renderer profile: ${miniwam_renderer_profile}" >&2
        return 1
        ;;
esac

unset miniwam_renderer_gpu miniwam_renderer_profile miniwam_oidn_lib miniwam_oidn_core
unset miniwam_oidn_api miniwam_oidn_cuda miniwam_expected_hashes miniwam_oidn_files
unset miniwam_index miniwam_actual_hash miniwam_gpu_capability
