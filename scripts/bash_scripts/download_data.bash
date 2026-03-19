#!/bin/bash
set -euo pipefail

REPO="${MINDCUBE_DATASET_REPO:-MLL-Lab/MindCube}"
ZIP_NAME="data.zip"
TMP_DIR="./temp_download"
EXTRACT_DIR="./temp_extract"

cleanup() {
    rm -rf "${TMP_DIR}" "${EXTRACT_DIR}" "${ZIP_NAME}" 2>/dev/null || true
}

trap cleanup EXIT

echo "Downloading MindCube dataset from Hugging Face: ${REPO}"

mkdir -p "${TMP_DIR}"

if command -v hf >/dev/null 2>&1; then
    hf download "${REPO}" "data.zip" --local-dir "${TMP_DIR}" --repo-type dataset
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${REPO}" "data.zip" --local-dir "${TMP_DIR}" --repo-type dataset
fi

if [[ -f "${TMP_DIR}/data.zip" ]]; then
    mv "${TMP_DIR}/data.zip" "${ZIP_NAME}"
fi

if [[ ! -f "${ZIP_NAME}" ]]; then
    URL="https://huggingface.co/datasets/${REPO}/resolve/main/data.zip"
    curl -L "${URL}" -o "${ZIP_NAME}" -sS
fi

mkdir -p "${EXTRACT_DIR}"
unzip -q "${ZIP_NAME}" -d "${EXTRACT_DIR}"

if [[ -d "${EXTRACT_DIR}/data" ]]; then
    mkdir -p data
    cp -a "${EXTRACT_DIR}/data/." data/
else
    mkdir -p data
    cp -a "${EXTRACT_DIR}/." data/
fi
