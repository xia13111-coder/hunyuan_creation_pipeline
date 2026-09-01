#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ENV_FILE="${PIPELINE_DOCKER_ENV_FILE:-${SCRIPT_DIR}/.env.runtime}"
COMPOSE_FILE="${SCRIPT_DIR}/compose.full.yaml"

usage() {
    printf '%s\n' \
        "Usage: docker/run-full.sh init|preflight|up|build|replace|down|status|logs|shell|smoke" \
        "" \
        "  init       create docker/.env.runtime from the safe template" \
        "  preflight  validate Docker, GPU, disk, models, materials and mounts" \
        "  up         start the verified existing image" \
        "  build      build the current source, verify it, then start" \
        "  replace    remove the old named pipeline container, then start" \
        "  down       stop this Compose stack" \
        "  status     show container and health status" \
        "  logs       follow pipeline logs" \
        "  shell      open a shell in the pipeline container" \
        "  smoke      repeat full container checks and start Isaac headless"
}

require_env_file() {
    if [ ! -f "${ENV_FILE}" ]; then
        printf 'Missing %s. Run: docker/run-full.sh init\n' "${ENV_FILE}" >&2
        exit 2
    fi
}

compose() {
    docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"
}

compose_image() {
    local image="$1"
    shift
    env PIPELINE_IMAGE="${image}" \
        docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"
}

preflight() {
    python3 "${SCRIPT_DIR}/preflight.py" --env-file "${ENV_FILE}" "$@"
}

env_value() {
    python3 - "${ENV_FILE}" "$1" "${SCRIPT_DIR}" <<'PY'
import sys
from pathlib import Path

env_file, name, script_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(script_dir))
from preflight import load_environment  # noqa: E402

value = load_environment(env_file).get(name, "")
if not value:
    raise SystemExit(f"{name} is not configured")
print(value)
PY
}

env_value_or_default() {
    python3 - "${ENV_FILE}" "$1" "$2" "${SCRIPT_DIR}" <<'PY'
import sys
from pathlib import Path

env_file, name, default, script_dir = (
    Path(sys.argv[1]),
    sys.argv[2],
    sys.argv[3],
    Path(sys.argv[4]),
)
sys.path.insert(0, str(script_dir))
from preflight import load_environment  # noqa: E402

print(load_environment(env_file).get(name, "").strip() or default)
PY
}

preflight_image() {
    local image="$1"
    shift
    env PIPELINE_IMAGE="${image}" \
        python3 "${SCRIPT_DIR}/preflight.py" --env-file "${ENV_FILE}" "$@"
}

verify_candidate_image() {
    local image="$1"
    compose_image "${image}" run --rm --no-deps --entrypoint /bin/bash pipeline -lc \
        'python -m asset_pipeline.docker_preflight --profile complete && bash docker/isaac-smoke.sh'
}

start_verified_service() {
    local timeout
    timeout="$(env_value_or_default PIPELINE_START_TIMEOUT_SECONDS 300)"
    compose up -d "$@" --wait --wait-timeout "${timeout}"
}

require_docker_build_space() {
    local docker_root minimum_gb
    docker_root="$(docker info --format '{{.DockerRootDir}}')"
    minimum_gb="$(env_value_or_default PIPELINE_DOCKER_MIN_BUILD_FREE_GB 15)"
    python3 - "${docker_root}" "${minimum_gb}" <<'PY'
import shutil
import sys
from pathlib import Path

docker_root = Path(sys.argv[1])
try:
    minimum_gb = float(sys.argv[2])
except ValueError as exc:
    raise SystemExit(
        "PIPELINE_DOCKER_MIN_BUILD_FREE_GB must be numeric, "
        f"got {sys.argv[2]!r}"
    ) from exc
if minimum_gb < 0:
    raise SystemExit("PIPELINE_DOCKER_MIN_BUILD_FREE_GB must not be negative")

free_gb = shutil.disk_usage(docker_root).free / 1024**3
print(f"Docker root free space: {free_gb:.1f} GiB ({docker_root})")
if free_gb < minimum_gb:
    raise SystemExit(
        f"Docker build requires at least {minimum_gb:g} GiB free under "
        f"{docker_root}; only {free_gb:.1f} GiB is available"
    )
PY
}

ensure_hub_cache() {
    local name image uid gid cache_root actual_cache
    name="$(env_value HUB_CONTAINER_NAME)"
    image="$(env_value HUB_IMAGE)"
    uid="$(env_value PIPELINE_UID)"
    gid="$(env_value PIPELINE_GID)"
    cache_root="$(env_value MODEL_CACHE_ROOT)/ov-hub"
    if docker container inspect "${name}" >/dev/null 2>&1; then
        actual_cache="$(docker container inspect "${name}" --format \
            '{{range .Mounts}}{{if eq .Destination "/var/cache/hub"}}{{.Source}}{{end}}{{end}}')"
        if [ "${actual_cache}" = "${cache_root}" ]; then
            docker start "${name}" >/dev/null
            return
        fi
        printf 'Recreating %s because its cache mount changed.\n' "${name}"
        docker rm -f "${name}" >/dev/null
    fi
    docker run -d \
        --name "${name}" \
        --network host \
        --user "${uid}:${gid}" \
        --restart unless-stopped \
        --volume "${cache_root}:/var/cache/hub:rw" \
        "${image}" >/dev/null
}

command="${1:-}"
case "${command}" in
    init)
        if [ -e "${ENV_FILE}" ]; then
            printf 'Refusing to overwrite existing runtime file: %s\n' "${ENV_FILE}" >&2
            exit 2
        fi
        install -m 600 "${SCRIPT_DIR}/env.runtime.example" "${ENV_FILE}"
        printf 'Created %s\nEdit every REQUIRED path, then run docker/run-full.sh preflight.\n' "${ENV_FILE}"
        ;;
    preflight)
        require_env_file
        preflight --prepare-runtime
        ;;
    up)
        require_env_file
        preflight --prepare-runtime
        ensure_hub_cache
        start_verified_service
        compose ps
        ;;
    build)
        require_env_file
        require_docker_build_space
        target_image="$(env_value PIPELINE_IMAGE)"
        candidate_image="${target_image}-candidate-$(date +%Y%m%d%H%M%S)"
        previous_image_id="$(docker image inspect "${target_image}" \
            --format '{{.Id}}' 2>/dev/null || true)"
        preflight --prepare-runtime --skip-image
        ensure_hub_cache
        printf 'Building candidate image: %s\n' "${candidate_image}"
        compose_image "${candidate_image}" build pipeline
        preflight_image "${candidate_image}"
        verify_candidate_image "${candidate_image}"
        docker image tag "${candidate_image}" "${target_image}"
        if ! start_verified_service --force-recreate; then
            printf 'New service failed health checks; restoring the previous image.\n' >&2
            if [ -n "${previous_image_id}" ]; then
                docker image tag "${previous_image_id}" "${target_image}"
                start_verified_service --force-recreate || true
            fi
            exit 1
        fi
        docker image rm "${candidate_image}" >/dev/null 2>&1 || true
        compose ps
        ;;
    replace)
        require_env_file
        container_name="$({ sed -n 's/^PIPELINE_CONTAINER_NAME=//p' "${ENV_FILE}" || true; } | tail -1)"
        container_name="${container_name:-hunyuan-pipeline-601}"
        preflight --prepare-runtime
        ensure_hub_cache
        if docker container inspect "${container_name}" >/dev/null 2>&1; then
            docker rm -f "${container_name}"
        fi
        start_verified_service
        compose ps
        ;;
    down)
        require_env_file
        compose down
        ;;
    status)
        require_env_file
        compose ps
        ;;
    logs)
        require_env_file
        compose logs --tail 200 -f pipeline
        ;;
    shell)
        require_env_file
        compose exec pipeline bash
        ;;
    smoke)
        require_env_file
        compose exec -T pipeline \
            python -m asset_pipeline.docker_preflight --profile complete
        compose exec -T pipeline bash docker/isaac-smoke.sh
        ;;
    *)
        usage
        exit 2
        ;;
esac
