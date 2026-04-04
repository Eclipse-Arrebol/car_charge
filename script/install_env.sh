#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-}"
TORCH_GEOMETRIC_VERSION="${TORCH_GEOMETRIC_VERSION:-}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
CUDA_FLAVOR="${CUDA_FLAVOR:-cu128}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PIP_FALLBACK_INDEX_URL="${PIP_FALLBACK_INDEX_URL:-https://pypi.org/simple}"
VENV_PYTHON=""

resolve_venv_python() {
  local candidates=(
    "$VENV_DIR/bin/python"
    "$VENV_DIR/bin/python3"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      VENV_PYTHON="$candidate"
      return 0
    fi
  done

  echo "Virtual environment Python not found under $VENV_DIR/bin" >&2
  exit 1
}

pip_install() {
  local extra_args=("$@")
  local pip_args=()

  if [[ -n "$PIP_INDEX_URL" ]]; then
    pip_args+=(--index-url "$PIP_INDEX_URL")
  fi

  if "$VENV_PYTHON" -m pip install "${extra_args[@]}" "${pip_args[@]}"; then
    return 0
  fi

  if [[ -n "$PIP_INDEX_URL" ]]; then
    echo "Primary pip index failed, retrying with fallback: $PIP_FALLBACK_INDEX_URL"
    "$VENV_PYTHON" -m pip install "${extra_args[@]}" --index-url "$PIP_FALLBACK_INDEX_URL"
    return $?
  fi

  return 1
}

usage() {
  cat <<'EOF'
Usage:
  bash script/install_env.sh [--cuda cpu|cu118|cu121|cu126|cu128] [--python python3.10] [--venv /path/to/venv]

Examples:
  bash script/install_env.sh
  bash script/install_env.sh --cuda cu128
  bash script/install_env.sh --python python3.10 --venv /data/ev-demo-venv

Environment variables:
  TORCH_VERSION   Default: 2.7.1
  TORCHVISION_VERSION Optional override, auto-matched from TORCH_VERSION when unset
  TORCHAUDIO_VERSION Optional override, auto-matched from TORCH_VERSION when unset
  TORCH_GEOMETRIC_VERSION Optional override, auto-matched from TORCH_VERSION when unset
  NUMPY_VERSION   Default: 1.26.4
  CUDA_FLAVOR     Default: cu128
  PYTHON_BIN      Default: python3
  VENV_DIR        Default: <repo>/.venv
  PIP_INDEX_URL   Optional primary pip index URL
  PIP_FALLBACK_INDEX_URL Default: https://pypi.org/simple
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      CUDA_FLAVOR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script currently targets Ubuntu/Debian servers with apt-get." >&2
  exit 1
fi

APT_GET_PREFIX=()
if command -v sudo >/dev/null 2>&1; then
  APT_GET_PREFIX=(sudo)
elif [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "apt-get requires root privileges, but sudo is not available." >&2
  echo "Please run this script as root, or install sudo in the container first." >&2
  exit 1
fi

echo "[1/6] Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
"${APT_GET_PREFIX[@]}" apt-get update
"${APT_GET_PREFIX[@]}" apt-get install -y \
  build-essential \
  git \
  curl \
  ca-certificates \
  pkg-config \
  "$PYTHON_BIN" \
  python3-pip \
  python3-venv

echo "[2/6] Creating virtual environment at $VENV_DIR ..."
mkdir -p "$(dirname "$VENV_DIR")"
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv --copies "$VENV_DIR"
resolve_venv_python

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[3/6] Upgrading pip tooling..."
if ! pip_install --upgrade pip setuptools wheel; then
  echo "Warning: failed to upgrade pip tooling. Continuing with the existing pip version."
fi

if [[ -z "$TORCHVISION_VERSION" || -z "$TORCHAUDIO_VERSION" || -z "$TORCH_GEOMETRIC_VERSION" ]]; then
  case "$TORCH_VERSION" in
    2.3.*)
      : "${TORCHVISION_VERSION:=0.18.1}"
      : "${TORCHAUDIO_VERSION:=2.3.1}"
      : "${TORCH_GEOMETRIC_VERSION:=2.5.3}"
      ;;
    2.7.*)
      : "${TORCHVISION_VERSION:=0.22.1}"
      : "${TORCHAUDIO_VERSION:=2.7.1}"
      : "${TORCH_GEOMETRIC_VERSION:=2.7.0}"
      ;;
    *)
      echo "Unsupported TORCH_VERSION for automatic torchvision/torchaudio matching: $TORCH_VERSION" >&2
      echo "Please set TORCHVISION_VERSION, TORCHAUDIO_VERSION, and TORCH_GEOMETRIC_VERSION explicitly." >&2
      exit 1
      ;;
  esac
fi

if [[ "$CUDA_FLAVOR" == "cpu" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
else
  TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_FLAVOR}"
fi

echo "[4/6] Installing PyTorch ${TORCH_VERSION} / torchvision ${TORCHVISION_VERSION} / torchaudio ${TORCHAUDIO_VERSION} (${CUDA_FLAVOR})..."
pip_install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "$TORCH_INDEX_URL"

TORCH_MAJOR_MINOR="$("$VENV_PYTHON" - <<'PY'
import torch
parts = torch.__version__.split('+', 1)[0].split('.')
print(f"{parts[0]}.{parts[1]}.0")
PY
)"

PYG_WHL_URL="https://data.pyg.org/whl/torch-${TORCH_MAJOR_MINOR}+${CUDA_FLAVOR}.html"

echo "[5/6] Installing PyTorch Geometric and project dependencies..."
# Reinstall PyG native extensions from the exact wheel index that matches the
# selected PyTorch/CUDA combo. Mixing wheel builds across torch versions can
# crash the interpreter at import time.
pip_install \
  --force-reinstall \
  pyg_lib \
  torch_scatter \
  torch_sparse \
  torch_cluster \
  torch_spline_conv \
  -f "$PYG_WHL_URL"

pip_install \
  --force-reinstall \
  "torch_geometric==${TORCH_GEOMETRIC_VERSION}" \
  "numpy==${NUMPY_VERSION}" \
  networkx \
  matplotlib \
  cvxpy \
  osmnx \
  scipy \
  pandas \
  shapely

echo "[6/6] Running sanity checks..."
"$VENV_PYTHON" - <<'PY'
import torch
import torch_geometric
import cvxpy
import osmnx
from env.Traffic import TrafficPowerEnv

env = TrafficPowerEnv()
state = env.get_graph_state()

print("torch:", torch.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("cvxpy:", cvxpy.__version__)
print("osmnx:", osmnx.__version__)
print("state.x shape:", tuple(state.x.shape))
assert state.x.shape[1] == 15, state.x.shape
PY

cat <<EOF

Environment is ready.

Activate it with:
  source "$VENV_DIR/bin/activate"

Typical commands:
  python main.py train
  python main.py train-real
  python evaluation/run_evaluation.py
EOF
