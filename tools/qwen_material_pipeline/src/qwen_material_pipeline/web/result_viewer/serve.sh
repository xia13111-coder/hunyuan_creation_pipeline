#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  serve.sh [PORT]
  serve.sh --port PORT

Serve the isolated qwen_material_pipeline result site.

Environment variables:
  BIND_ADDRESS  Address to bind (default: 0.0.0.0)
  PYTHON_BIN    Python executable (default: python3)
  DELIVERY_DIR  Validated delivery directory to expose (required)
EOF
}

port="8088"
case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  -p|--port)
    if [[ $# -lt 2 ]]; then
      echo "error: --port requires a value" >&2
      usage >&2
      exit 2
    fi
    port="$2"
    shift 2
    ;;
  "")
    ;;
  *)
    port="$1"
    shift
    ;;
esac

if [[ $# -ne 0 ]]; then
  echo "error: unexpected argument: $1" >&2
  usage >&2
  exit 2
fi

if [[ ! "$port" =~ ^[0-9]+$ ]]; then
  echo "error: port must be an integer between 1 and 65535: $port" >&2
  exit 2
fi
port_number=$((10#$port))
if (( port_number < 1 || port_number > 65535 )); then
  echo "error: port must be between 1 and 65535: $port" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
viewer="$script_dir/index.html"
manifest_builder="$script_dir/build_manifest.py"
: "${DELIVERY_DIR:?Set DELIVERY_DIR to a validated delivery directory}"
delivery="$DELIVERY_DIR"

if [[ ! -f "$viewer" ]]; then
  echo "error: result viewer does not exist: $viewer" >&2
  exit 1
fi
if [[ ! -f "$manifest_builder" ]]; then
  echo "error: result viewer manifest builder does not exist: $manifest_builder" >&2
  exit 1
fi
if [[ ! -d "$delivery" ]]; then
  echo "error: validated delivery is unavailable: $delivery" >&2
  echo "set DELIVERY_DIR to the result directory you want to inspect" >&2
  exit 1
fi
delivery="$(cd -- "$delivery" && pwd -P)"

runtime_root="$(mktemp -d)"
cleanup() {
  rm -rf -- "$runtime_root"
}
trap cleanup EXIT INT TERM
mkdir -p "$runtime_root/result_viewer"
cp "$viewer" "$runtime_root/result_viewer/index.html"
ln -s "$delivery" "$runtime_root/result_viewer/delivery"

bind_address="${BIND_ADDRESS:-0.0.0.0}"
python_bin="${PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "error: Python executable not found: $python_bin" >&2
  exit 1
fi
"$python_bin" "$manifest_builder" \
  --delivery "$delivery" \
  --output "$runtime_root/result_viewer/viewer_manifest.json"

echo "Serving validated delivery: $delivery"
echo "Viewer:  http://127.0.0.1:${port_number}/result_viewer/"
echo "Bind:    ${bind_address}:${port_number}"
echo "Warning: this development server has no authentication."

exec "$python_bin" -m http.server "$port_number" \
  --bind "$bind_address" \
  --directory "$runtime_root"
