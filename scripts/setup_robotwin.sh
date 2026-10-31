#!/bin/bash
# Check out the pinned RoboTwin 2.0 revision, apply MiniWAM's evaluator patch and
# link the MiniWAM XPolicyLab policy adapter into it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${MINIWAM_ROOT}/RoboTwin}"
ROBOTWIN_REPOSITORY="${ROBOTWIN_REPOSITORY:-https://github.com/RoboTwin-Platform/RoboTwin.git}"
ROBOTWIN_REVISION="96c1feab536306b50c26af200044fcdf126e8904"
PATCH="${MINIWAM_ROOT}/integrations/robotwin/robotwin.patch"
# The patched official evaluator must match the one that captured the frozen cases.
EVALUATOR_SHA256="b76f04ad2a88c99292632c2965bc0687edf3ef2bb66c264c0d8800d673824ec9"

if [[ ! -d "${ROBOTWIN_ROOT}/.git" ]]; then
    git clone --recurse-submodules "${ROBOTWIN_REPOSITORY}" "${ROBOTWIN_ROOT}"
    git -C "${ROBOTWIN_ROOT}" checkout --detach "${ROBOTWIN_REVISION}"
elif [[ "$(git -C "${ROBOTWIN_ROOT}" rev-parse HEAD)" != "${ROBOTWIN_REVISION}" ]]; then
    echo "[MiniWAM][ERROR] ${ROBOTWIN_ROOT} is not at ${ROBOTWIN_REVISION}." >&2
    exit 1
fi
git -C "${ROBOTWIN_ROOT}" submodule update --init --recursive

if git -C "${ROBOTWIN_ROOT}" apply --reverse --check "${PATCH}" 2>/dev/null; then
    echo "MiniWAM evaluator patch already applied."
elif ! git -C "${ROBOTWIN_ROOT}" apply "${PATCH}"; then
    echo "[MiniWAM][ERROR] The evaluator patch does not apply; the checkout may carry an older one." >&2
    echo "Restore the patched files, then rerun this script:" >&2
    echo "  git -C ${ROBOTWIN_ROOT} checkout -- scripts/eval_policy.sh scripts/eval_policy_multitask.py scripts/eval_policy_xpolicylab.py" >&2
    exit 1
fi
actual_sha256=$(sha256sum "${ROBOTWIN_ROOT}/scripts/eval_policy_xpolicylab.py" | awk '{print $1}')
if [[ "${actual_sha256}" != "${EVALUATOR_SHA256}" ]]; then
    echo "[MiniWAM][ERROR] Patched evaluator hash ${actual_sha256} != ${EVALUATOR_SHA256}." >&2
    exit 1
fi

link() {  # link SOURCE TARGET, refusing to replace anything else
    local source="$1" target="$2"
    if [[ -L "${target}" ]]; then
        if [[ "$(readlink -f "${target}")" != "$(readlink -f "${source}")" ]]; then
            echo "[MiniWAM][ERROR] ${target} points elsewhere; remove it first." >&2
            exit 1
        fi
    elif [[ -e "${target}" ]]; then
        echo "[MiniWAM][ERROR] Refusing to replace existing ${target}." >&2
        exit 1
    else
        ln -s "${source}" "${target}"
    fi
}
adapter="${MINIWAM_ROOT}/integrations/xpolicylab/policy/MiniWAM"
link "${adapter}" "${ROBOTWIN_ROOT}/XPolicyLab/policy/MiniWAM"
link "${adapter}/compat/frozen_cases.py" "${ROBOTWIN_ROOT}/scripts/frozen_cases.py"

echo "RoboTwin ready: ${ROBOTWIN_ROOT}"
echo "RoboTwin revision: $(git -C "${ROBOTWIN_ROOT}" rev-parse HEAD)"
echo "XPolicyLab revision: $(git -C "${ROBOTWIN_ROOT}/XPolicyLab" rev-parse HEAD)"
