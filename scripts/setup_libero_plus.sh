#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_PLUS_REPO="https://github.com/sylvestf/LIBERO-plus.git"
LIBERO_PLUS_REV="4976dc30028e805ff8094b55501d532c48fec182"
LIBERO_PLUS_DIR="${LIBERO_PLUS_ROOT:-$ROOT_DIR/LIBERO-plus}"

command -v git >/dev/null 2>&1 || {
    echo "git is required but was not found on PATH." >&2
    exit 1
}

if [[ ! -e "$LIBERO_PLUS_DIR" ]]; then
    git clone "$LIBERO_PLUS_REPO" "$LIBERO_PLUS_DIR"
    git -C "$LIBERO_PLUS_DIR" checkout --detach "$LIBERO_PLUS_REV"
elif [[ ! -d "$LIBERO_PLUS_DIR/.git" ]]; then
    echo "$LIBERO_PLUS_DIR exists but is not a Git checkout." >&2
    exit 1
else
    current_rev="$(git -C "$LIBERO_PLUS_DIR" rev-parse HEAD)"
    if [[ -n "$(git -C "$LIBERO_PLUS_DIR" status --porcelain)" ]]; then
        echo "$LIBERO_PLUS_DIR contains local changes; refusing to modify it." >&2
        exit 1
    fi
    if [[ "$current_rev" != "$LIBERO_PLUS_REV" ]]; then
        echo "LIBERO-Plus is at $current_rev, expected $LIBERO_PLUS_REV." >&2
        exit 1
    fi
fi

benchmark_root="$LIBERO_PLUS_DIR/libero/libero"
mkdir -p "$ROOT_DIR/.libero_plus"
cat > "$ROOT_DIR/.libero_plus/config.yaml" <<EOF
assets: $benchmark_root/assets
bddl_files: $benchmark_root/bddl_files
benchmark_root: $benchmark_root
datasets: $LIBERO_PLUS_DIR/libero/datasets
init_states: $benchmark_root/init_files
EOF

cat <<EOF
Pinned LIBERO-Plus checkout ready at $LIBERO_PLUS_DIR.

The 6.4 GB perturbation assets are intentionally not downloaded automatically.
Follow docs/evaluation.md (LIBERO-Plus), then verify with:

  python scripts/verify_miniwam_environment.py --check-libero-plus
EOF
