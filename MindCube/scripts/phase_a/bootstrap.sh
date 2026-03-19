#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<USAGE
Usage: bash scripts/phase_a/bootstrap.sh [--skip-install]

Options:
  --skip-install   Skip dependency installation and only run environment checks
USAGE
  exit 0
fi

SKIP_INSTALL="false"
if [[ "${1:-}" == "--skip-install" ]]; then
  SKIP_INSTALL="true"
fi

cd "${PROJECT_ROOT}"
echo "[phase-a] Project root: ${PROJECT_ROOT}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[phase-a] Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
else
  echo "[phase-a] Reusing existing virtual environment at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ "${SKIP_INSTALL}" == "false" ]]; then
  echo "[phase-a] Installing dependencies"
  "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "[phase-a] macOS detected: installing requirements without decord"
    TMP_REQ="$(mktemp)"
    grep -v '^decord==0.6.0$' requirements.txt > "${TMP_REQ}"
    "${VENV_PYTHON}" -m pip install -r "${TMP_REQ}"
    rm -f "${TMP_REQ}"
  else
    "${VENV_PYTHON}" -m pip install -r requirements.txt
  fi

  "${VENV_PYTHON}" -m pip install pypdf huggingface_hub
else
  echo "[phase-a] Skipping dependency installation"
fi

echo "[phase-a] Running import checks"
for script in \
  scripts/data_processing.py \
  scripts/generate_prompts.py \
  scripts/convert_to_sft.py \
  scripts/run_inference.py \
  scripts/run_evaluation.py; do
  "${VENV_PYTHON}" "${script}" --help >/dev/null
done

echo "[phase-a] Bootstrap complete"
