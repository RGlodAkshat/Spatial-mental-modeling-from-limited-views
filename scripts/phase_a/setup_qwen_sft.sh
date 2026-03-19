#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_DIR="${PROJECT_ROOT}/Qwen2.5-VL-MindCube"
QWEN_REPO_URL="${QWEN_REPO_URL:-https://github.com/QinengWang-Aiden/Qwen2.5-VL-MindCube.git}"

cd "${PROJECT_ROOT}"
if [[ ! -d "${QWEN_DIR}" ]]; then
  git clone "${QWEN_REPO_URL}" "${QWEN_DIR}"
fi
"${PYTHON_BIN}" experiments/sft/patch_qwen_data.py patch
