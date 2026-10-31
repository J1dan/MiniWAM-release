#!/bin/bash
set -euo pipefail

# Keep ~/.local packages out of both installation and simulator subprocesses.
# They can otherwise satisfy pip dependencies or conflict with upstream pins.
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${MINIWAM_ROOT}/RoboTwin}"
ENV_PREFIX="${ROBOTWIN_EVAL_ENV_PREFIX:-${MINIWAM_ROOT}/.robotwin-env}"
CACHE_ROOT="${ROBOTWIN_INSTALL_CACHE:-${MINIWAM_ROOT}/.cache/robotwin-install}"
CUROBO_REV="d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
DOWNLOAD_ASSETS=0
INSTALL_PYTORCH3D=0
TORCH_PROFILE="${ROBOTWIN_TORCH_PROFILE:-auto}"

usage() {
    cat <<EOF
Usage: scripts/setup_robotwin_eval_env.sh [--prefix PATH] [--download-assets] [--with-pytorch3d]

Create the official RoboTwin simulator/client environment on the workspace
filesystem. The default prefix is:
  ${ENV_PREFIX}

Package caches and temporary build files are also kept below the MiniWAM
workspace so installations do not exhaust a small root filesystem.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            ENV_PREFIX=${2:?--prefix requires a path}
            shift 2
            ;;
        --download-assets)
            DOWNLOAD_ASSETS=1
            shift
            ;;
        --with-pytorch3d)
            INSTALL_PYTORCH3D=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "${ROBOTWIN_ROOT}/.git" ]]; then
    echo "[MiniWAM][ERROR] RoboTwin checkout not found: ${ROBOTWIN_ROOT}" >&2
    echo "Run bash scripts/setup_robotwin.sh first." >&2
    exit 1
fi

expected_xpl_revision=$(git -C "${ROBOTWIN_ROOT}" ls-tree HEAD XPolicyLab | awk '{print $3}')
actual_xpl_revision=$(git -C "${ROBOTWIN_ROOT}/XPolicyLab" rev-parse HEAD)
if [[ "${actual_xpl_revision}" != "${expected_xpl_revision}" ]]; then
    echo "[MiniWAM][ERROR] XPolicyLab is ${actual_xpl_revision}, expected pinned ${expected_xpl_revision}." >&2
    echo "Run bash scripts/setup_robotwin.sh to restore the pinned submodule." >&2
    exit 1
fi

mkdir -p "$(dirname "${ENV_PREFIX}")"
available_kb=$(df -Pk "$(dirname "${ENV_PREFIX}")" | awk 'NR == 2 {print $4}')
required_kb=$((30 * 1024 * 1024))
if [[ ! -x "${ENV_PREFIX}/bin/python" ]] && (( available_kb < required_kb )); then
    echo "[MiniWAM][ERROR] RoboTwin installation requires at least 30 GiB free." >&2
    echo "Target filesystem has $((available_kb / 1024 / 1024)) GiB: ${ENV_PREFIX}" >&2
    exit 1
fi

mkdir -p "${CACHE_ROOT}/conda-pkgs" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/tmp"
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda-pkgs"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${CACHE_ROOT}/tmp"

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    conda create --prefix "${ENV_PREFIX}" python=3.10 -y
fi
conda env config vars set --prefix "${ENV_PREFIX}" PYTHONNOUSERSITE=1
conda activate "${ENV_PREFIX}"
set -u

# Upstream RoboTwin pins torch 2.4.1/cu121. That wheel has no sm_120 kernels,
# so Blackwell requires the closest XPolicyLab-tested pairing: torch
# 2.7.1/cu128. ROBOTWIN_TORCH_PROFILE can force "official" or "blackwell".
if [[ "${TORCH_PROFILE}" == "auto" ]]; then
    gpu_capability=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | sort -V | tail -n 1 || true)
    fi
    if [[ "${gpu_capability%%.*}" =~ ^[0-9]+$ ]] && \
       (( ${gpu_capability%%.*} >= 12 )); then
        TORCH_PROFILE="blackwell"
    else
        TORCH_PROFILE="official"
    fi
fi

case "${TORCH_PROFILE}" in
    official)
        TORCH_VERSION="2.4.1"
        TORCHVISION_VERSION="0.19.1"
        CUDA_FAMILY="12.1"
        PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
        ;;
    blackwell)
        TORCH_VERSION="2.7.1"
        TORCHVISION_VERSION="0.22.1"
        CUDA_FAMILY="12.8"
        PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
        export TORCH_CUDA_ARCH_LIST="12.0"
        ;;
    *)
        echo "[MiniWAM][ERROR] ROBOTWIN_TORCH_PROFILE must be auto, official, or blackwell." >&2
        exit 2
        ;;
esac
echo "[MiniWAM] torch_profile=${TORCH_PROFILE} torch=${TORCH_VERSION} cuda=${CUDA_FAMILY}"

# Compile CuRobo/PyTorch3D with a matching workspace-local NVCC instead of
# whichever /usr/bin/nvcc the host happens to expose.
set +u
conda install --prefix "${ENV_PREFIX}" -c nvidia \
    "cuda-nvcc=${CUDA_FAMILY}" \
    "cuda-cudart=${CUDA_FAMILY}" \
    "cuda-cudart-dev=${CUDA_FAMILY}" \
    "cuda-cccl=${CUDA_FAMILY}" \
    "cuda-version=${CUDA_FAMILY}" \
    -y
set -u
export CUDA_HOME="${ENV_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# Install the upstream requirements without its two torch lines; the selected
# compatible pair is installed explicitly below and cannot be silently
# downgraded on an idempotent rerun.
filtered_requirements=$(mktemp "${TMPDIR}/robotwin-requirements.XXXXXX.txt")
trap 'rm -f "${filtered_requirements}"' EXIT
awk '!/^torch(==|vision)/' "${ROBOTWIN_ROOT}/scripts/requirements.txt" > "${filtered_requirements}"
python -m pip install -r "${filtered_requirements}"
python -m pip install ninja

# RoboTwin records evaluation videos by launching an `ffmpeg` executable. Its
# requirements install imageio-ffmpeg (and an unrelated Python package named
# `ffmpeg`) but do not guarantee an executable on PATH. Reuse ImageIO's pinned
# binary inside this self-contained prefix.
if ! command -v ffmpeg >/dev/null 2>&1; then
    imageio_ffmpeg_exe=$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
    if [[ ! -x "${imageio_ffmpeg_exe}" ]]; then
        echo "[MiniWAM][ERROR] ImageIO FFmpeg binary not found: ${imageio_ffmpeg_exe}" >&2
        exit 1
    fi
    ln -s "${imageio_ffmpeg_exe}" "${ENV_PREFIX}/bin/ffmpeg"
fi
ffmpeg -version >/dev/null

if ! EXPECTED_TORCH="${TORCH_VERSION}" EXPECTED_CUDA="${CUDA_FAMILY}" python - <<'PY'
import os
import torch
import torch._C

assert torch.__version__.split("+")[0] == os.environ["EXPECTED_TORCH"]
assert (torch.version.cuda or "").startswith(os.environ["EXPECTED_CUDA"])
PY
then
    echo "[MiniWAM][WARN] Installing the ${TORCH_PROFILE} PyTorch stack." >&2
    python -m pip install --force-reinstall \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${PYTORCH_INDEX_URL}"
fi
EXPECTED_TORCH="${TORCH_VERSION}" EXPECTED_CUDA="${CUDA_FAMILY}" python - <<'PY'
import os
import torch
import torch._C

if torch.__version__.split("+")[0] != os.environ["EXPECTED_TORCH"]:
    raise RuntimeError(f"Unexpected torch version: {torch.__version__}")
if not (torch.version.cuda or "").startswith(os.environ["EXPECTED_CUDA"]):
    raise RuntimeError(f"Unexpected torch CUDA runtime: {torch.version.cuda}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
PY
nvcc_release=$(nvcc --version | awk '/release/ {gsub(/,/, "", $5); print $5}')
if [[ "${nvcc_release}" != "${CUDA_FAMILY}" ]]; then
    echo "[MiniWAM][ERROR] Expected CUDA ${CUDA_FAMILY} NVCC, found ${nvcc_release:-unknown}." >&2
    exit 1
fi
echo "nvcc=${nvcc_release} CUDA_HOME=${CUDA_HOME}"

# PyTorch3D is not needed by the RGB/state leaderboard evaluator. Install it
# only for workflows that consume 3-D data; it is a substantial CUDA build.
if [[ "${INSTALL_PYTORCH3D}" -eq 1 ]] && ! python -m pip install \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
        --no-build-isolation; then
    echo "[MiniWAM][WARN] PyTorch3D installation failed; continuing without 3-D support." >&2
fi

python -m pip install -e "${ROBOTWIN_ROOT}/XPolicyLab"

sapien_root=$(python -m pip show sapien | awk '/^Location:/ {print $2}')/sapien
urdf_loader="${sapien_root}/wrapper/urdf_loader.py"
if [[ ! -f "${urdf_loader}" ]]; then
    echo "[MiniWAM][ERROR] SAPIEN URDF loader not found: ${urdf_loader}" >&2
    exit 1
fi
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${urdf_loader}"
sed -i 's/urdf_file\[:-4\] + "srdf"/urdf_file[:-4] + ".srdf"/' "${urdf_loader}"

mplib_root=$(python -m pip show mplib | awk '/^Location:/ {print $2}')/mplib
planner="${mplib_root}/planner.py"
if [[ ! -f "${planner}" ]]; then
    echo "[MiniWAM][ERROR] MPLib planner not found: ${planner}" >&2
    exit 1
fi
sed -i -E \
    's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
    "${planner}"

curobo_root="${ROBOTWIN_ROOT}/envs/curobo"
if [[ ! -e "${curobo_root}" ]]; then
    git clone --branch v0.7.8 --depth 1 \
        https://github.com/NVlabs/curobo.git "${curobo_root}"
elif [[ ! -d "${curobo_root}/.git" ]]; then
    echo "[MiniWAM][ERROR] Existing CuRobo path is not a Git checkout: ${curobo_root}" >&2
    exit 1
fi
curobo_revision=$(git -C "${curobo_root}" rev-parse HEAD)
if [[ "${curobo_revision}" != "${CUROBO_REV}" ]] || \
   [[ -n "$(git -C "${curobo_root}" status --porcelain)" ]]; then
    echo "[MiniWAM][ERROR] CuRobo must be clean at ${CUROBO_REV}; found ${curobo_revision}." >&2
    exit 1
fi
python -m pip install -e "${curobo_root}" --no-build-isolation
python -m pip install \
    "warp-lang==1.12.0" \
    "setuptools==69.5.1" \
    "pillow==11.3.0" \
    "scipy==1.10.1" \
    "scikit-image==0.21.0"

python - <<'PY'
import importlib
import torch
import torch._C
import curobo
import mplib
import sapien
import scipy
import skimage
import websockets

assert scipy.__version__ == "1.10.1", scipy.__version__
assert skimage.__version__ == "0.21.0", skimage.__version__
for extension in (
    "geom_cu",
    "kinematics_fused_cu",
    "lbfgs_step_cu",
    "line_search_cu",
    "tensor_step_cu",
):
    importlib.import_module(f"curobo.curobolib.{extension}")
if torch.cuda.is_available():
    # This catches wheels that import but do not contain kernels for the GPU
    # (notably upstream torch 2.4.1/cu121 on Blackwell sm_120).
    value = torch.sign(torch.tensor([-1.0, 1.0], device="cuda")).cpu().tolist()
    assert value == [-1.0, 1.0], value
print("RoboTwin simulator imports: OK")
PY
python -m pip check

if [[ "${DOWNLOAD_ASSETS}" -eq 1 ]]; then
    (
        cd "${ROBOTWIN_ROOT}"
        bash scripts/_download_assets.sh
    )
fi

echo
echo "RoboTwin simulator environment ready: ${ENV_PREFIX}"
echo "Evaluate with: --eval-env-conda-env ${ENV_PREFIX}"
if [[ "${DOWNLOAD_ASSETS}" -eq 0 ]]; then
    echo "Assets were not requested. Download them with:"
    echo "  cd ${ROBOTWIN_ROOT} && bash scripts/_download_assets.sh"
fi
