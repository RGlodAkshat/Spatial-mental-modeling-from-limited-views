#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/phase_a/bootstrap.sh [--skip-install]"
  exit 0
fi

SKIP_INSTALL="false"
if [[ "${1:-}" == "--skip-install" ]]; then
  SKIP_INSTALL="true"
fi

cd "${PROJECT_ROOT}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ "${SKIP_INSTALL}" == "false" ]]; then
  "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
  "${VENV_PYTHON}" -m pip install -r requirements.txt
  "${VENV_PYTHON}" -m pip install pypdf huggingface_hub
fi
