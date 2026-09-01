#!/usr/bin/env bash
set -uo pipefail

isaac_python="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
max_attempts="${ISAAC_SMOKE_MAX_ATTEMPTS:-3}"

if ! [[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ISAAC_SMOKE_MAX_ATTEMPTS must be a positive integer, got: %s\n' \
        "${max_attempts}" >&2
    exit 2
fi

for attempt in $(seq 1 "${max_attempts}"); do
    printf 'Isaac Sim headless smoke attempt %s/%s\n' "${attempt}" "${max_attempts}"
    "${isaac_python}" -c \
        'from isaacsim import SimulationApp; app = SimulationApp({"headless": True}); print("Isaac Sim OK"); app.close()'
    exit_code=$?
    if [ "${exit_code}" -eq 0 ]; then
        exit 0
    fi
    if [ "${attempt}" -eq "${max_attempts}" ]; then
        printf 'Isaac Sim smoke failed after %s attempts (last exit %s).\n' \
            "${max_attempts}" "${exit_code}" >&2
        exit "${exit_code}"
    fi
    printf 'Isaac Sim exited with %s; retrying in a clean process.\n' \
        "${exit_code}" >&2
    sleep $((attempt * 2))
done
