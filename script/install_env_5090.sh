#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RTX 5090 profile: use CUDA 12.8 wheels that include SM 12.0 support.
export VENV_DIR="${VENV_DIR:-${HOME:-/root}/.venvs/ev-demo-5090}"
export CUDA_FLAVOR="${CUDA_FLAVOR:-cu128}"
export TORCH_VERSION="${TORCH_VERSION:-2.7.1}"

bash "$SCRIPT_DIR/install_env.sh" "$@"
