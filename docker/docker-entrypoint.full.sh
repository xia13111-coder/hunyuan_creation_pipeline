#!/usr/bin/env bash
set -euo pipefail

export ROOT_DIR="${ROOT_DIR:-/workspace/hunyuan3.0_assets_creation}"
export ISAACSIM_ROOT="${ISAACSIM_ROOT:-/isaac-sim}"
export ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAACSIM_ROOT}/python.sh}"
export BLENDER_BIN="${BLENDER_BIN:-/opt/blender/blender}"
export PIPELINE_ASSET_ROOT="${PIPELINE_ASSET_ROOT:-/workspace/assets}"

if [ -d "${ROOT_DIR}" ]; then
    cd "${ROOT_DIR}"
fi

preflight_profile="${PIPELINE_PREFLIGHT_PROFILE:-}"
if [ -n "${preflight_profile}" ] && [ "${preflight_profile}" != "off" ]; then
    preflight_args=(--profile "${preflight_profile}")
    if [ "${PIPELINE_PREFLIGHT_SKIP_PYTHON_PROBES:-0}" = "1" ]; then
        preflight_args+=(--skip-python-probes)
    fi
    python -m asset_pipeline.docker_preflight "${preflight_args[@]}"
fi

exec "$@"
