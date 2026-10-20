#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_planning.sh [options] [-- PLANNING_ARGS...]

Run the released slot_planning.py client in a temporary Docker container.
The model server and simulation server must already be running on the host.

Options:
  --task-id IDS          Task id or comma-separated ids. Default: 0.
  --task-variant NAME    Task variant or "all". Default: default.
  --suite NAME           Run reach, pick, rotate, or place paper tasks.
  --sim-host HOST        Simulation server host. Default: 127.0.0.1.
  --sim-port PORT        Simulation server port. Default: 8021.
  --model-port PORT      Slot-JEPA model server port. Default: 8031.
  --image IMAGE          Docker image. Default: REALM_DOCKER_IMAGE or realm:latest.
  --dry-run              Validate inputs and print the Docker command only.
  -h, --help             Show this help.

Additional slot_planning.py arguments can be passed after --.

Examples:
  scripts/run_planning.sh
  scripts/run_planning.sh --task-id 0 --task-variant default
  scripts/run_planning.sh --suite reach
  scripts/run_planning.sh --suite pick -- --retries 1
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ROBOT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
RELEASE_ROOT="$(cd -- "${ROBOT_ROOT}/../.." >/dev/null 2>&1 && pwd -P)"

TASK_IDS="0"
TASK_VARIANT="default"
TASK_VARIANT_SET=0
SUITE=""
SIM_HOST="${REALM_SIM_SERVER_HOST:-127.0.0.1}"
SIM_PORT="${REALM_SIM_SERVER_PORT:-8021}"
MODEL_PORT="${SLOT_JEPA_MODEL_PORT:-8031}"
DOCKER_IMAGE="${REALM_DOCKER_IMAGE:-realm:latest}"
DRY_RUN=0
PLANNING_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id|--task_id)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value." >&2; exit 2; }
      TASK_IDS="$2"
      shift 2
      ;;
    --task-id=*|--task_id=*) TASK_IDS="${1#*=}"; shift ;;
    --task-variant)
      [[ $# -ge 2 ]] || { echo "ERROR: --task-variant requires a value." >&2; exit 2; }
      TASK_VARIANT="$2"
      TASK_VARIANT_SET=1
      shift 2
      ;;
    --task-variant=*) TASK_VARIANT="${1#*=}"; TASK_VARIANT_SET=1; shift ;;
    --suite)
      [[ $# -ge 2 ]] || { echo "ERROR: --suite requires a value." >&2; exit 2; }
      SUITE="$2"
      shift 2
      ;;
    --suite=*) SUITE="${1#*=}"; shift ;;
    --sim-host)
      [[ $# -ge 2 ]] || { echo "ERROR: --sim-host requires a value." >&2; exit 2; }
      SIM_HOST="$2"
      shift 2
      ;;
    --sim-host=*) SIM_HOST="${1#*=}"; shift ;;
    --sim-port)
      [[ $# -ge 2 ]] || { echo "ERROR: --sim-port requires a value." >&2; exit 2; }
      SIM_PORT="$2"
      shift 2
      ;;
    --sim-port=*) SIM_PORT="${1#*=}"; shift ;;
    --model-port)
      [[ $# -ge 2 ]] || { echo "ERROR: --model-port requires a value." >&2; exit 2; }
      MODEL_PORT="$2"
      shift 2
      ;;
    --model-port=*) MODEL_PORT="${1#*=}"; shift ;;
    --image)
      [[ $# -ge 2 ]] || { echo "ERROR: --image requires a value." >&2; exit 2; }
      DOCKER_IMAGE="$2"
      shift 2
      ;;
    --image=*) DOCKER_IMAGE="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      PLANNING_ARGS=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown wrapper option: $1" >&2
      echo "Pass native slot_planning.py options after --." >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${SUITE}" in
  "") ;;
  reach)
    TASK_IDS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17"
    (( TASK_VARIANT_SET )) || TASK_VARIANT="default"
    ;;
  pick)
    TASK_IDS="24,25,26,27,28,29"
    (( TASK_VARIANT_SET )) || TASK_VARIANT="all"
    ;;
  rotate)
    TASK_IDS="30,31,32,33,34,35"
    (( TASK_VARIANT_SET )) || TASK_VARIANT="all"
    ;;
  place)
    TASK_IDS="18,19,20,21,22,23"
    (( TASK_VARIANT_SET )) || TASK_VARIANT="all"
    ;;
  *)
    echo "ERROR: --suite must be reach, pick, rotate, or place (got ${SUITE})." >&2
    exit 2
    ;;
esac

for port_spec in "sim:${SIM_PORT}" "model:${MODEL_PORT}"; do
  port_name="${port_spec%%:*}"
  port_value="${port_spec#*:}"
  if [[ ! "${port_value}" =~ ^[0-9]+$ ]] || (( port_value < 1 || port_value > 65535 )); then
    echo "ERROR: ${port_name} port must be an integer from 1 to 65535 (got ${port_value})." >&2
    exit 2
  fi
done

SUCCESS_MANIFEST="${RELEASE_ROOT}/artifacts/data/robot/successful_case/put_green_block_into_bowl_Default_camera_frame/manifest.json"
[[ -f "${SUCCESS_MANIFEST}" ]] || { echo "ERROR: missing released goal targets: ${SUCCESS_MANIFEST}" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is not installed or not in PATH." >&2; exit 1; }
docker info >/dev/null 2>&1 || {
  echo "ERROR: Docker daemon is not reachable by user $(id -un)." >&2
  echo "Check Docker service status and /var/run/docker.sock permissions." >&2
  exit 1
}
docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1 || {
  echo "ERROR: Docker image not found: ${DOCKER_IMAGE}" >&2
  echo "Build it first with: ./setup.sh --docker" >&2
  exit 1
}

DOCKER_ARGS=(
  run --rm
  --network=host
  --userns=host
  --user "0:$(id -g)"
  -e OMNIGIBSON_HEADLESS=1
  -e OMNI_KIT_ALLOW_ROOT=1
  -e "REALM_HOST_UID=$(id -u)"
  -e "REALM_HOST_GID=$(id -g)"
  -e REALM_OUTPUT_UMASK=0002
  -e PYTHONDONTWRITEBYTECODE=1
  --mount "type=bind,src=${ROBOT_ROOT},dst=/app"
  --mount "type=bind,src=${RELEASE_ROOT}/artifacts,dst=/artifacts,readonly"
)
if [[ -t 0 && -t 1 ]]; then
  DOCKER_ARGS+=(-it)
fi

CONTAINER_COMMAND=(
  python /app/runtime/slot_planning.py
  --sim-server-host "${SIM_HOST}"
  --sim-server-port "${SIM_PORT}"
  --port "${MODEL_PORT}"
  --task_id "${TASK_IDS}"
  --task-variant "${TASK_VARIANT}"
  "${PLANNING_ARGS[@]}"
)

echo "Starting Slot-JEPA planning client"
echo "  image       : ${DOCKER_IMAGE}"
echo "  task ids    : ${TASK_IDS}"
echo "  variant     : ${TASK_VARIANT}"
echo "  sim server  : ${SIM_HOST}:${SIM_PORT}"
echo "  model server: 127.0.0.1:${MODEL_PORT}"
echo "  GPU         : none requested (planning client is RPC-only)"

if (( DRY_RUN )); then
  printf 'docker'
  printf ' %q' "${DOCKER_ARGS[@]}" "${DOCKER_IMAGE}" "${CONTAINER_COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec docker "${DOCKER_ARGS[@]}" "${DOCKER_IMAGE}" "${CONTAINER_COMMAND[@]}"
