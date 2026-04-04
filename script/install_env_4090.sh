#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RTX 4090 profile: keep a broadly compatible CUDA 11.8 stack.
export VENV_DIR="${VENV_DIR:-${HOME:-/root}/.venvs/ev-demo-4090}"
export CUDA_FLAVOR="${CUDA_FLAVOR:-cu118}"
export TORCH_VERSION="${TORCH_VERSION:-2.3.1}"

bash "$SCRIPT_DIR/install_env.sh" "$@"
