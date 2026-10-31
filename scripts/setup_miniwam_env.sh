#!/usr/bin/env bash
set -euo pipefail

# Install dependencies into this environment, even when ~/.local has a matching
# version; another environment's installer must not be able to remove them.
export PYTHONNOUSERSITE=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="miniwam"
SKIP_LIBERO=0
LIBERO_REPO="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_REV="8f1084e3132a39270c3a13ebe37270a43ece2a01"

usage() {
    cat <<'EOF'
Usage: scripts/setup_miniwam_env.sh [--skip-libero]

Create or update the MiniWAM Conda environment, install this source checkout,
and clone the pinned LIBERO checkout. Existing dirty or
unpinned LIBERO checkouts are never modified automatically.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-libero)
            SKIP_LIBERO=1
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

command -v conda >/dev/null 2>&1 || {
    echo "conda is required but was not found on PATH." >&2
    exit 1
}
command -v git >/dev/null 2>&1 || {
    echo "git is required but was not found on PATH." >&2
    exit 1
}

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Updating existing Conda environment '$ENV_NAME'..."
    conda env update --name "$ENV_NAME" --file "$ROOT_DIR/environment.yml" --prune
else
    echo "Creating Conda environment '$ENV_NAME'..."
    conda env create --name "$ENV_NAME" --file "$ROOT_DIR/environment.yml"
fi

# Install this checkout as a standard editable package. Dependencies are pinned above.
conda env config vars set --name "$ENV_NAME" PYTHONNOUSERSITE=1
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --no-deps -e "$ROOT_DIR"

if [[ "$SKIP_LIBERO" -eq 0 ]]; then
    LIBERO_DIR="$ROOT_DIR/LIBERO"
    if [[ ! -e "$LIBERO_DIR" ]]; then
        git clone "$LIBERO_REPO" "$LIBERO_DIR"
        git -C "$LIBERO_DIR" checkout --detach "$LIBERO_REV"
    elif [[ ! -d "$LIBERO_DIR/.git" ]]; then
        echo "$LIBERO_DIR exists but is not a Git checkout." >&2
        exit 1
    else
        current_rev="$(git -C "$LIBERO_DIR" rev-parse HEAD)"
        if [[ -n "$(git -C "$LIBERO_DIR" status --porcelain)" ]]; then
            echo "$LIBERO_DIR contains local changes; refusing to modify it." >&2
            exit 1
        fi
        if [[ "$current_rev" != "$LIBERO_REV" ]]; then
            echo "LIBERO is at $current_rev, expected $LIBERO_REV." >&2
            echo "Move/remove the checkout or select the pinned revision manually." >&2
            exit 1
        fi
    fi

    mkdir -p "$ROOT_DIR/.libero"
    cat > "$ROOT_DIR/.libero/config.yaml" <<EOF
assets: $LIBERO_DIR/libero/libero/assets
bddl_files: $LIBERO_DIR/libero/libero/bddl_files
benchmark_root: $LIBERO_DIR/libero/libero
datasets: $LIBERO_DIR/libero/datasets
init_states: $LIBERO_DIR/libero/libero/init_files
EOF
fi

verify_args=()
if [[ "$SKIP_LIBERO" -eq 0 ]]; then
    verify_args+=(--check-libero)
fi
conda run --no-capture-output -n "$ENV_NAME" \
    python "$ROOT_DIR/scripts/verify_miniwam_environment.py" "${verify_args[@]}"

echo
echo "Environment '$ENV_NAME' is ready. Activate it with: conda activate $ENV_NAME"
echo "On a GPU node, run: python scripts/verify_miniwam_environment.py --require-cuda --check-libero"
