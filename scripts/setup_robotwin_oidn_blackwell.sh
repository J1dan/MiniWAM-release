#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_ROOT="${MINIWAM_OIDN_INSTALL_ROOT:-${MINIWAM_ROOT}/.robotwin-compat}"
VERSION=2.3.3
ARCHIVE="oidn-${VERSION}.x86_64.linux.tar.gz"
ARCHIVE_SHA256=3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f
URL="https://github.com/RenderKit/oidn/releases/download/v${VERSION}/${ARCHIVE}"

usage() {
    echo "Usage: scripts/setup_robotwin_oidn_blackwell.sh [--prefix DIRECTORY]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) INSTALL_ROOT=${2:?--prefix requires a directory}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

TARGET="${INSTALL_ROOT}/oidn-${VERSION}.x86_64.linux"
mkdir -p "${INSTALL_ROOT}/downloads"
archive_path="${INSTALL_ROOT}/downloads/${ARCHIVE}"
if [[ ! -f "${archive_path}" ]]; then
    curl -fL --retry 3 -o "${archive_path}" "${URL}"
fi
actual_archive_hash=$(sha256sum "${archive_path}" | awk '{print $1}')
if [[ "${actual_archive_hash}" != "${ARCHIVE_SHA256}" ]]; then
    echo "[MiniWAM][ERROR] OIDN archive SHA256 mismatch: ${actual_archive_hash}" >&2
    exit 1
fi
if [[ ! -d "${TARGET}" ]]; then
    tar -xzf "${archive_path}" -C "${INSTALL_ROOT}"
fi

MINIWAM_ROBOTWIN_RENDER_PROFILE=oidn_2_3_3_blackwell \
MINIWAM_OIDN_ROOT="${TARGET}" \
bash -c 'source "$1" 0' _ \
    "${MINIWAM_ROOT}/integrations/xpolicylab/policy/MiniWAM/compat/configure_renderer_runtime.sh"

echo "OIDN Blackwell compatibility profile installed: ${TARGET}"
echo "Archive SHA256: ${ARCHIVE_SHA256}"
