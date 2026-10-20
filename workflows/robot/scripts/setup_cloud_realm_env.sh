#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_cloud_realm_env.sh [--data-path PATH] [--no-install]

This script is for cloud containers where Docker is unavailable. It installs
only the lightweight Python dependencies needed by REALM planning/model
servers, creates Isaac/Omniverse cache directories, and writes a source-able
environment file.

Options:
  --data-path PATH   Cache / dataset root. Defaults to REALM_DATA_PATH or
                     ~/shared-nvme/wenpeisong/WM-data if present, otherwise
                     ./WM-data.
  --no-install       Only write environment files and run diagnostics.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REALM_ROOT=$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)
REQ_FILE="${SCRIPT_DIR}/requirements_cloud_planning.txt"
INSTALL_DEPS=1

if [[ -n "${REALM_DATA_PATH:-}" ]]; then
  DATA_PATH="${REALM_DATA_PATH}"
elif [[ -d "${HOME}/shared-nvme/wenpeisong/WM-data" ]]; then
  DATA_PATH="${HOME}/shared-nvme/wenpeisong/WM-data"
else
  DATA_PATH="${REALM_ROOT}/WM-data"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path)
      DATA_PATH="$2"
      shift 2
      ;;
    --data-path=*)
      DATA_PATH="${1#*=}"
      shift
      ;;
    --no-install)
      INSTALL_DEPS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "${REALM_ROOT}"

PYTHON_BIN=$(command -v python || true)
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not found in PATH. Activate your conda/env first." >&2
  exit 1
fi

echo "REALM root: ${REALM_ROOT}"
echo "Data/cache root: ${DATA_PATH}"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import sys
print("Python version:", sys.version.replace("\n", " "))
try:
    import torch
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
except Exception as exc:
    print("Torch check failed:", exc)
PY

mkdir -p \
  "${DATA_PATH}/isaac-sim/cache/kit" \
  "${DATA_PATH}/isaac-sim/cache/ov" \
  "${DATA_PATH}/isaac-sim/cache/pip" \
  "${DATA_PATH}/isaac-sim/cache/glcache" \
  "${DATA_PATH}/isaac-sim/cache/computecache" \
  "${DATA_PATH}/isaac-sim/cache/kit_user" \
  "${DATA_PATH}/isaac-sim/cache/carb" \
  "${DATA_PATH}/isaac-sim/cache/omniverse" \
  "${DATA_PATH}/isaac-sim/cache/matplotlib" \
  "${DATA_PATH}/isaac-sim/logs" \
  "${DATA_PATH}/isaac-sim/config" \
  "${DATA_PATH}/isaac-sim/local_share_ov" \
  "${DATA_PATH}/isaac-sim/data" \
  "${DATA_PATH}/isaac-sim/documents" \
  "${DATA_PATH}/datasets"

if [[ ! -f "${DATA_PATH}/isaac-sim/config/omniverse.toml" ]]; then
  cat > "${DATA_PATH}/isaac-sim/config/omniverse.toml" <<'EOF'
[privacy]
privacyConsent = true

[cache]
enabled = true
EOF
fi

ENV_FILE="${REALM_ROOT}/.realm_cloud_env.sh"
cat > "${ENV_FILE}" <<EOF
# Source this file before running REALM planning in a cloud container:
#   source ${ENV_FILE}
export REALM_ROOT="${REALM_ROOT}"
export REALM_DATA_PATH="${DATA_PATH}"
export OMNIGIBSON_HEADLESS=1
export OMNI_KIT_ALLOW_ROOT=1
export OMNI_KIT_DISABLE_EXT_REGISTRY=1
export OMNI_KIT_DISABLE_EXTENSIONS_REGISTRY=1
export OV_DISABLE_EXT_REGISTRY=1
export OMNI_KIT_REGISTRY_ENABLED=0
export REALM_DISABLE_OMNI_PARTICLE_EXTENSIONS=1
export MPLCONFIGDIR="${DATA_PATH}/isaac-sim/cache/matplotlib"
export OMNIGIBSON_DATASET_PATH="${DATA_PATH}/datasets/og_dataset"
export OMNIGIBSON_ASSET_PATH="${DATA_PATH}/datasets/assets"
export GIBSON_DATASET_PATH="${DATA_PATH}/datasets/g_dataset"
export OMNIGIBSON_KEY_PATH="${DATA_PATH}/datasets/omnigibson.key"
export PYTHONPATH="${REALM_ROOT}/../../src:${REALM_ROOT}:\${PYTHONPATH:-}"
EOF

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "Installing minimal Python dependencies into the current environment..."
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"
else
  echo "Skipping pip install because --no-install was set."
fi

echo "Running dependency diagnostics..."
"${PYTHON_BIN}" - <<'PY'
mods = [
    "msgpack", "zmq", "yaml", "PIL", "cv2", "numpy", "scipy", "pandas",
    "matplotlib", "timm", "einops", "omegaconf", "decord",
]
missing = []
for name in mods:
    try:
        __import__(name)
    except Exception as exc:
        missing.append((name, str(exc)))
if missing:
    print("Missing/problematic modules:")
    for name, exc in missing:
        print(f"  {name}: {exc}")
else:
    print("Core Python dependencies: OK")

try:
    import omnigibson
    print("OmniGibson: OK", getattr(omnigibson, "__version__", "unknown"))
except Exception as exc:
    print("OmniGibson check failed:", exc)
    print("REALM sim server cannot run in this Python environment without OmniGibson.")
    print("Use an image/env that already contains IsaacSim + OmniGibson, or run via Docker/Apptainer on a host that supports it.")

for name in ("isaacsim", "omni", "carb"):
    try:
        __import__(name)
        print(f"{name}: OK")
    except Exception as exc:
        print(f"{name} check failed: {exc}")
PY

echo
echo "Environment file written:"
echo "  ${ENV_FILE}"
echo
echo "Next:"
echo "  source ${ENV_FILE}"
echo "  python serve_vjepa_droid.py --port 8010 --ckpt-path outputs/vjepa-droid/vjepa_droid/run/latest.pt"
