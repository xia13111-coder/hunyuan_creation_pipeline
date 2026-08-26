#!/usr/bin/env bash
set -euo pipefail

# Build the isolated Qwen3.5 runtime and fetch the two Hugging Face checkpoints
# used by the visual retrieval path.  The production pipeline is still launched
# from hunyuan_sam3d; its isolated multimodel material subprocess uses this
# interpreter while SAM3, MVInverse, Isaac Sim, and Blender keep theirs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODEL_VOLUME="${ASSET_MODEL_VOLUME:-${XDG_CACHE_HOME:-${HOME}/.cache}/hunyuan_asset_pipeline}"
RUNTIME_ROOT="${QWEN35_RUNTIME_ROOT:-${MODEL_VOLUME}/qwen35_4b_runtime}"
ENV_DIR="${QWEN35_ENV_DIR:-${RUNTIME_ROOT}/env}"
MODEL_DIR="${QWEN35_MODEL_DIR:-${RUNTIME_ROOT}/model}"
SIGLIP2_DIR="${SIGLIP2_MODEL_DIR:-${MODEL_VOLUME}/models/siglip2-base-patch16-224}"
CONDA_BIN="${CONDA_BIN:-${HOME}/miniconda3/bin/conda}"
FETCH_SCRIPT="${SCRIPT_DIR}/fetch_verified_models.py"
SMOKE_SCRIPT="${SCRIPT_DIR}/smoke_qwen35_runtime.py"

mkdir -p \
  "${RUNTIME_ROOT}/conda_pkgs" \
  "${RUNTIME_ROOT}/pip_cache" \
  "${RUNTIME_ROOT}/hf" \
  "${RUNTIME_ROOT}/modelscope" \
  "${RUNTIME_ROOT}/tmp" \
  "$(dirname -- "${MODEL_DIR}")" \
  "$(dirname -- "${SIGLIP2_DIR}")"

exec 9>"${RUNTIME_ROOT}/setup.lock"
flock 9

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  CONDA_PKGS_DIRS="${RUNTIME_ROOT}/conda_pkgs" \
    "${CONDA_BIN}" create -y -p "${ENV_DIR}" python=3.11 pip
fi

TMPDIR="${RUNTIME_ROOT}/tmp" \
PIP_CACHE_DIR="${RUNTIME_ROOT}/pip_cache" \
  "${ENV_DIR}/bin/python" -m pip install \
    --disable-pip-version-check --quiet \
    torch==2.12.1 torchvision==0.27.1 \
    --index-url https://download.pytorch.org/whl/cu126

TMPDIR="${RUNTIME_ROOT}/tmp" \
PIP_CACHE_DIR="${RUNTIME_ROOT}/pip_cache" \
  "${ENV_DIR}/bin/python" -m pip install \
    --disable-pip-version-check --quiet \
    --requirement "${PACKAGE_ROOT}/requirements-qwen35.txt"

download_checkpoint() {
  local model_key="$1"
  local destination="$2"
  local timestamp
  local staging_root
  local huggingface_staging
  local huggingface_failed_marker
  local huggingface_ready
  local modelscope_staging
  local selected_staging
  local quarantine

  if "${ENV_DIR}/bin/python" "${FETCH_SCRIPT}" \
    --model "${model_key}" \
    --destination "${destination}" \
    --source verify; then
    return 0
  fi

  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  staging_root="${destination}.incomplete"
  huggingface_staging="${staging_root}/huggingface"
  huggingface_failed_marker="${huggingface_staging}/.transport-failed"
  modelscope_staging="${staging_root}/modelscope"
  mkdir -p "${huggingface_staging}"
  huggingface_ready=0

  if [[ "${QWEN35_PREFER_HUGGINGFACE:-1}" == "1" ]] \
    && [[ ! -f "${huggingface_failed_marker}" ]]; then
    if TMPDIR="${RUNTIME_ROOT}/tmp" \
      HF_HOME="${RUNTIME_ROOT}/hf" \
      HF_HUB_ETAG_TIMEOUT="30" \
      HF_HUB_DOWNLOAD_TIMEOUT="120" \
      NO_PROXY="${NO_PROXY:+${NO_PROXY},}huggingface.co,.huggingface.co,hf.co,.hf.co,xethub.hf.co,.xethub.hf.co" \
      no_proxy="${no_proxy:+${no_proxy},}huggingface.co,.huggingface.co,hf.co,.hf.co,xethub.hf.co,.xethub.hf.co" \
        "${ENV_DIR}/bin/python" "${FETCH_SCRIPT}" \
          --model "${model_key}" \
          --destination "${huggingface_staging}" \
          --source huggingface \
          --attempts 2 \
          --max-workers 4; then
      huggingface_ready=1
    else
      touch "${huggingface_failed_marker}"
    fi
  fi

  if [[ "${huggingface_ready}" == "1" ]]; then
    selected_staging="${huggingface_staging}"
  else
    echo "${model_key}: Hugging Face transport unavailable or disabled; using the pinned ModelScope mirror and verifying every runtime file." >&2
    TMPDIR="${RUNTIME_ROOT}/tmp" \
    MODELSCOPE_CACHE="${RUNTIME_ROOT}/modelscope" \
      "${ENV_DIR}/bin/python" "${FETCH_SCRIPT}" \
        --model "${model_key}" \
        --destination "${modelscope_staging}" \
        --source modelscope \
        --attempts 3 \
        --max-workers 2
    selected_staging="${modelscope_staging}"
  fi

  "${ENV_DIR}/bin/python" "${FETCH_SCRIPT}" \
    --model "${model_key}" \
    --destination "${selected_staging}" \
    --source verify

  if [[ -e "${destination}" ]]; then
    quarantine="${destination}.invalid-${timestamp}-$$"
    mv -- "${destination}" "${quarantine}"
    echo "${model_key}: quarantined prior incomplete directory at ${quarantine}" >&2
  fi
  mv -- "${selected_staging}" "${destination}"
  if [[ -d "${staging_root}" ]]; then
    rm -rf -- "${staging_root}"
  fi
  "${ENV_DIR}/bin/python" "${FETCH_SCRIPT}" \
    --model "${model_key}" \
    --destination "${destination}" \
    --source verify
}

download_checkpoint "qwen3_5_4b" "${MODEL_DIR}"
download_checkpoint "siglip2_base" "${SIGLIP2_DIR}"

"${ENV_DIR}/bin/python" -m pip check

QWEN35_RUNTIME_ROOT="${RUNTIME_ROOT}" \
QWEN35_ENV_DIR="${ENV_DIR}" \
QWEN35_MODEL_DIR="${MODEL_DIR}" \
SIGLIP2_MODEL_DIR="${SIGLIP2_DIR}" \
  "${ENV_DIR}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

import cv2
import torch
import transformers

runtime = Path(os.environ["QWEN35_RUNTIME_ROOT"])
environment = Path(os.environ["QWEN35_ENV_DIR"])
model = Path(os.environ["QWEN35_MODEL_DIR"])
config = json.loads((model / "config.json").read_text(encoding="utf-8"))
if config.get("model_type") != "qwen3_5":
    raise RuntimeError(f"unexpected Qwen model_type: {config.get('model_type')!r}")
siglip = Path(os.environ["SIGLIP2_MODEL_DIR"])
siglip_config = json.loads((siglip / "config.json").read_text(encoding="utf-8"))
if siglip_config.get("model_type") != "siglip":
    raise RuntimeError(
        f"unexpected SigLIP2 model_type: {siglip_config.get('model_type')!r}"
    )
if cv2.__version__ != "4.14.0":
    raise RuntimeError(f"unexpected OpenCV version: {cv2.__version__!r}")
if transformers.__version__ != "5.14.1":
    raise RuntimeError(
        f"unexpected Transformers version: {transformers.__version__!r}"
    )
if torch.__version__ != "2.12.1+cu126":
    raise RuntimeError(f"unexpected Torch version: {torch.__version__!r}")
print(f"Qwen3.5 runtime root: {runtime}")
print(f"Qwen3.5 Python ready: {environment / 'bin' / 'python'}")
print(f"Qwen3.5 checkpoint provisioned and verified: {model}")
print(f"SigLIP2 checkpoint provisioned and verified: {siglip}")
PY

if [[ "${QWEN35_RUN_SMOKE:-0}" == "1" ]]; then
  PYTHONPATH="$(dirname -- "${PACKAGE_ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
    "${ENV_DIR}/bin/python" "${SMOKE_SCRIPT}" \
      --model-path "${MODEL_DIR}" \
      --cycles 2
else
  echo "CUDA inference smoke skipped; set QWEN35_RUN_SMOKE=1 to run it."
fi
