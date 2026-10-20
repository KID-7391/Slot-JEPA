#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/start_sim_server.sh [options] [-- SIM_SERVER_ARGS...]

Start the released REALM OmniGibson simulation server in Docker.

Options:
  --data-path PATH   Override the REALM_DATA_PATH configured during setup.
  --gpu ID           Physical host GPU. Default: REALM_SIM_GPU or 1.
  --host HOST        Simulation server bind address. Default: 0.0.0.0.
  --port PORT        Simulation server port. Default: 8021.
  --image IMAGE      Docker image. Default: REALM_DOCKER_IMAGE or realm:latest.
  --shell            Open an interactive REALM shell instead of starting the server.
  --dry-run          Validate inputs and print the Docker command without running it.
  -h, --help         Show this help.

Examples:
  export REALM_DATA_PATH=/data/WM-data
  scripts/start_sim_server.sh
  scripts/start_sim_server.sh --gpu 2 --port 8022
  scripts/start_sim_server.sh --shell
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ROBOT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
RELEASE_ROOT="$(cd -- "${ROBOT_ROOT}/../.." >/dev/null 2>&1 && pwd -P)"

DATA_PATH="${REALM_DATA_PATH:-}"
GPU_ID="${REALM_SIM_GPU:-1}"
SERVER_HOST="0.0.0.0"
SERVER_PORT="8021"
DOCKER_IMAGE="${REALM_DOCKER_IMAGE:-realm:latest}"
OPEN_SHELL=0
DRY_RUN=0
SIM_SERVER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path)
      [[ $# -ge 2 ]] || { echo "ERROR: --data-path requires a value." >&2; exit 2; }
      DATA_PATH="$2"
      shift 2
      ;;
    --data-path=*) DATA_PATH="${1#*=}"; shift ;;
    --gpu)
      [[ $# -ge 2 ]] || { echo "ERROR: --gpu requires a value." >&2; exit 2; }
      GPU_ID="$2"
      shift 2
      ;;
    --gpu=*) GPU_ID="${1#*=}"; shift ;;
    --host)
      [[ $# -ge 2 ]] || { echo "ERROR: --host requires a value." >&2; exit 2; }
      SERVER_HOST="$2"
      shift 2
      ;;
    --host=*) SERVER_HOST="${1#*=}"; shift ;;
    --port)
      [[ $# -ge 2 ]] || { echo "ERROR: --port requires a value." >&2; exit 2; }
      SERVER_PORT="$2"
      shift 2
      ;;
    --port=*) SERVER_PORT="${1#*=}"; shift ;;
    --image)
      [[ $# -ge 2 ]] || { echo "ERROR: --image requires a value." >&2; exit 2; }
      DOCKER_IMAGE="$2"
      shift 2
      ;;
    --image=*) DOCKER_IMAGE="${1#*=}"; shift ;;
    --shell) OPEN_SHELL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      SIM_SERVER_ARGS=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${DATA_PATH}" ]]; then
  echo "ERROR: pass --data-path PATH or set REALM_DATA_PATH." >&2
  usage >&2
  exit 2
fi
if [[ ! "${SERVER_PORT}" =~ ^[0-9]+$ ]] || (( SERVER_PORT < 1 || SERVER_PORT > 65535 )); then
  echo "ERROR: --port must be an integer from 1 to 65535 (got ${SERVER_PORT})." >&2
  exit 2
fi
if [[ "${GPU_ID}" == "all" || "${GPU_ID}" == *,* ]]; then
  echo "ERROR: --gpu must select one physical GPU, not '${GPU_ID}'." >&2
  exit 2
fi
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "ERROR: REALM data path does not exist: ${DATA_PATH}" >&2
  exit 1
fi
DATA_PATH="$(cd -- "${DATA_PATH}" >/dev/null 2>&1 && pwd -P)"

DATASET_ROOT="${DATA_PATH}/datasets"
SUCCESS_MANIFEST="${RELEASE_ROOT}/artifacts/data/robot/successful_case/put_green_block_into_bowl_Default_camera_frame/manifest.json"
[[ -d "${DATASET_ROOT}" ]] || { echo "ERROR: missing REALM datasets: ${DATASET_ROOT}" >&2; exit 1; }
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

CACHE_ROOT="${DATA_PATH}/isaac-sim/cache"
LOCAL_SHARE_ROOT="${DATA_PATH}/isaac-sim/local_share_ov"
mkdir -p \
  "${CACHE_ROOT}/kit" \
  "${CACHE_ROOT}/ov" \
  "${CACHE_ROOT}/pip" \
  "${CACHE_ROOT}/glcache" \
  "${CACHE_ROOT}/computecache" \
  "${CACHE_ROOT}/kit_user" \
  "${CACHE_ROOT}/carb" \
  "${CACHE_ROOT}/omniverse" \
  "${DATA_PATH}/isaac-sim/logs" \
  "${DATA_PATH}/isaac-sim/config" \
  "${LOCAL_SHARE_ROOT}" \
  "${DATA_PATH}/isaac-sim/data" \
  "${DATA_PATH}/isaac-sim/documents"

DOCKER_ARGS=(
  run --rm
  --gpus "device=${GPU_ID}"
  --ipc=host
  --network=host
  --userns=host
  --user "0:$(id -g)"
  -e "NVIDIA_VISIBLE_DEVICES=${GPU_ID}"
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e OMNIVERSE_EULA_ACCEPTED=1
  -e OMNIGIBSON_HEADLESS=1
  -e OMNI_KIT_ALLOW_ROOT=1
  -e OMNI_KIT_DISABLE_EXT_REGISTRY=1
  -e OMNI_KIT_DISABLE_EXTENSIONS_REGISTRY=1
  -e OV_DISABLE_EXT_REGISTRY=1
  -e OMNI_KIT_REGISTRY_ENABLED=0
  -e REALM_DISABLE_OMNI_PARTICLE_EXTENSIONS=1
  -e "REALM_HOST_UID=$(id -u)"
  -e "REALM_HOST_GID=$(id -g)"
  -e REALM_OUTPUT_UMASK=0002
  -e PYTHONDONTWRITEBYTECODE=1
  --mount "type=bind,src=${ROBOT_ROOT},dst=/app"
  --mount "type=bind,src=${RELEASE_ROOT}/artifacts,dst=/artifacts,readonly"
  --mount "type=bind,src=${DATASET_ROOT},dst=/data"
  --mount "type=bind,src=${CACHE_ROOT}/kit,dst=/isaac-sim/kit/cache/Kit"
  --mount "type=bind,src=${CACHE_ROOT}/ov,dst=/root/.cache/ov"
  --mount "type=bind,src=${CACHE_ROOT}/pip,dst=/root/.cache/pip"
  --mount "type=bind,src=${CACHE_ROOT}/glcache,dst=/root/.cache/nvidia/GLCache"
  --mount "type=bind,src=${CACHE_ROOT}/computecache,dst=/root/.nv/ComputeCache"
  --mount "type=bind,src=${CACHE_ROOT}/kit_user,dst=/root/.cache/Kit"
  --mount "type=bind,src=${CACHE_ROOT}/carb,dst=/root/.cache/carb"
  --mount "type=bind,src=${CACHE_ROOT}/omniverse,dst=/root/.cache/omniverse"
  --mount "type=bind,src=${DATA_PATH}/isaac-sim/logs,dst=/root/.nvidia-omniverse/logs"
  --mount "type=bind,src=${DATA_PATH}/isaac-sim/config,dst=/root/.nvidia-omniverse/config"
  --mount "type=bind,src=${LOCAL_SHARE_ROOT},dst=/root/.local/share/ov"
  --mount "type=bind,src=${DATA_PATH}/isaac-sim/data,dst=/root/.local/share/ov/data"
  --mount "type=bind,src=${DATA_PATH}/isaac-sim/documents,dst=/root/Documents"
)

if [[ -f /usr/share/nvidia/nvoptix.bin ]]; then
  DOCKER_ARGS+=(--mount "type=bind,src=/usr/share/nvidia/nvoptix.bin,dst=/usr/share/nvidia/nvoptix.bin,readonly")
fi
if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  DOCKER_ARGS+=(--mount "type=bind,src=/usr/share/vulkan/icd.d/nvidia_icd.json,dst=/usr/share/vulkan/icd.d/nvidia_icd.json,readonly")
fi
if [[ -t 0 && -t 1 ]]; then
  DOCKER_ARGS+=(-it)
fi

if (( OPEN_SHELL )); then
  CONTAINER_COMMAND=(/bin/bash --login)
else
  CONTAINER_COMMAND=(
    python /app/runtime/slot_sim_server.py
    --host "${SERVER_HOST}"
    --port "${SERVER_PORT}"
    "${SIM_SERVER_ARGS[@]}"
  )
fi

echo "Starting REALM simulation"
echo "  image       : ${DOCKER_IMAGE}"
echo "  host GPU    : ${GPU_ID} (appears as cuda:0 in the container)"
echo "  data root   : ${DATA_PATH}"
if (( OPEN_SHELL )); then
  echo "  mode        : interactive shell"
else
  echo "  server      : ${SERVER_HOST}:${SERVER_PORT}"
fi
echo "  privileged  : disabled (required for single-GPU isolation)"

if (( DRY_RUN )); then
  printf 'docker'
  printf ' %q' "${DOCKER_ARGS[@]}" "${DOCKER_IMAGE}" "${CONTAINER_COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec docker "${DOCKER_ARGS[@]}" "${DOCKER_IMAGE}" "${CONTAINER_COMMAND[@]}"
