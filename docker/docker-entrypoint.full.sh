#!/usr/bin/env bash
set -euo pipefail

export ROOT_DIR="${ROOT_DIR:-/workspace/hunyuan3.0_assets_creation}"
export ISAACSIM_ROOT="${ISAACSIM_ROOT:-/isaac-sim}"
export ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAACSIM_ROOT}/python.sh}"
export BLENDER_BIN="${BLENDER_BIN:-/opt/blender/blender}"

if [ -d "${ROOT_DIR}" ]; then
    cd "${ROOT_DIR}"
fi

exec "$@"
