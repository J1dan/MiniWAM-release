#!/bin/bash
set -euo pipefail

policy_conda_env=${1:-miniwam}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# Install XPolicyLab's websocket transport and its evaluation scheduler's rich
# dependency. The rest of XPolicyLab is imported from the pinned submodule via
# PYTHONPATH, keeping simulator packages out of the MiniWAM policy environment.
python -m pip install \
    "websockets==15.0.1" \
    "msgpack==1.1.1" \
    "msgpack-numpy==0.4.8" \
    "pydantic==2.13.4" \
    "rich==15.0.0"

python - <<'PY'
import msgpack
import msgpack_numpy
import pydantic
import websockets
import rich.console

print(f"websockets={websockets.__version__}")
print(f"msgpack={msgpack.version}")
print("msgpack_numpy=ok")
print(f"pydantic={pydantic.__version__}")
PY

echo "MiniWAM XPolicyLab transport and scheduler dependencies are ready in '${policy_conda_env}'."
