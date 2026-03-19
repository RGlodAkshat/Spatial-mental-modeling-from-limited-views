#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_DIR="${PROJECT_ROOT}/Qwen2.5-VL-MindCube"
QWEN_REPO_URL="${QWEN_REPO_URL:-https://github.com/QinengWang-Aiden/Qwen2.5-VL-MindCube.git}"
QWEN_SETUP_REQUIRED="${QWEN_SETUP_REQUIRED:-false}"

cd "${PROJECT_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[phase-a] No NVIDIA GPU detected. Skipping Qwen setup by default."
  if [[ "${QWEN_SETUP_REQUIRED}" != "true" ]]; then
    exit 0
  fi
  echo "[phase-a] Qwen setup was explicitly requested; continuing anyway."
fi

if [[ ! -d "${QWEN_DIR}" ]]; then
  echo "[phase-a] Cloning Qwen SFT repo"
  git clone "${QWEN_REPO_URL}" "${QWEN_DIR}"
else
  echo "[phase-a] Reusing existing Qwen repo at ${QWEN_DIR}"
fi

echo "[phase-a] Verifying MindCube patch status"
if "${PYTHON_BIN}" experiments/sft/patch_qwen_data.py verify; then
  echo "[phase-a] Patch already applied"
else
  echo "[phase-a] Applying patch"
  "${PYTHON_BIN}" experiments/sft/patch_qwen_data.py patch
  "${PYTHON_BIN}" experiments/sft/patch_qwen_data.py verify
fi

echo "[phase-a] Qwen SFT setup complete"
