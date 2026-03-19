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
    echo "Attempting download with hf CLI..."
    hf download "${REPO}" "data.zip" --local-dir "${TMP_DIR}" --repo-type dataset
elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "Attempting download with huggingface-cli..."
    huggingface-cli download "${REPO}" "data.zip" --local-dir "${TMP_DIR}" --repo-type dataset
else
    echo "ℹ Hugging Face CLI not found. Skipping CLI-based download."
    echo "ℹ If the direct URL fallback fails, install one of: 'hf' or 'huggingface-cli' manually."
fi

if [[ -f "${TMP_DIR}/data.zip" ]]; then
    mv "${TMP_DIR}/data.zip" "${ZIP_NAME}"
    echo "✓ Download successful via Hugging Face CLI"
fi

if [[ ! -f "${ZIP_NAME}" ]]; then
    echo "Attempting direct download..."
    URL="https://huggingface.co/datasets/${REPO}/resolve/main/data.zip"
    curl -L "${URL}" -o "${ZIP_NAME}" -sS
fi

if [[ ! -f "${ZIP_NAME}" ]] || ! file "${ZIP_NAME}" | grep -q "Zip archive"; then
    echo ""
    echo "❌ Download failed. Potential reasons:"
    echo "1. Dataset is gated/private and needs auth"
    echo "2. Network issue"
    echo "3. Dataset repo path changed"
    echo ""
    echo "Try one of these manually:"
    echo "  hf auth login"
    echo "  huggingface-cli login"
    echo ""
    exit 1
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

rm -rf "${EXTRACT_DIR}" "${ZIP_NAME}" 2>/dev/null || true
rm -rf ./data/__MACOSX 2>/dev/null || true
rm -f ./data/.DS_Store 2>/dev/null || true

echo "🎉 Dataset successfully downloaded into existing ./data tree"
ls -la ./data || true
