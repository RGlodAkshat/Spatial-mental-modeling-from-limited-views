#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASK_NAME="${TASK_NAME:-plain_cgmap_ffr_out}"
if [[ "${TASK_NAME}" != "plain_cgmap_ffr_out" ]]; then
  echo "Unsupported TASK_NAME='${TASK_NAME}'. Phase A only supports plain_cgmap_ffr_out."
  exit 1
fi

TRAIN_SUBSET="${TRAIN_SUBSET:-data/raw/subsets/MindCube_train_50.jsonl}"
TINY_SUBSET="${TINY_SUBSET:-data/raw/subsets/MindCube_tinybench_50.jsonl}"
MODEL_TYPE="${MODEL_TYPE:-qwen2.5vl}"
PHASE_A_WRITE_FULL_CANONICAL="${PHASE_A_WRITE_FULL_CANONICAL:-false}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${TRAIN_SUBSET}" ]]; then
  echo "Missing train subset: ${TRAIN_SUBSET}"
  exit 1
fi
if [[ ! -f "${TINY_SUBSET}" ]]; then
  echo "Missing tinybench subset: ${TINY_SUBSET}"
  exit 1
fi

echo "[phase-a] Step 1: Scaffold generation"
"${PYTHON_BIN}" scripts/data_processing.py --input "${TRAIN_SUBSET}" --task full_pipeline
"${PYTHON_BIN}" scripts/data_processing.py --input "${TINY_SUBSET}" --task full_pipeline

train_base="$(basename "${TRAIN_SUBSET}" .jsonl)"
tiny_base="$(basename "${TINY_SUBSET}" .jsonl)"

train_scaffold="data/scaffold/all/${train_base}.jsonl"
tiny_scaffold="data/scaffold/all/${tiny_base}.jsonl"

if [[ ! -f "${train_scaffold}" || ! -f "${tiny_scaffold}" ]]; then
  echo "Scaffold generation failed: missing scaffold outputs"
  exit 1
fi

echo "[phase-a] Step 2: Prompt generation for ${TASK_NAME}"
train_prompt="data/prompts/general/${train_base}_${TASK_NAME}.jsonl"
tiny_prompt="data/prompts/general/${tiny_base}_${TASK_NAME}.jsonl"
smoke_train_prompt="data/prompts/general/${train_base}_${TASK_NAME}_smoke.jsonl"
smoke_tiny_prompt="data/prompts/general/${tiny_base}_${TASK_NAME}_smoke.jsonl"

"${PYTHON_BIN}" scripts/generate_prompts.py --input "${train_scaffold}" --task "${TASK_NAME}" --output "${smoke_train_prompt}"
"${PYTHON_BIN}" scripts/generate_prompts.py --input "${tiny_scaffold}" --task "${TASK_NAME}" --output "${smoke_tiny_prompt}"

if [[ "${PHASE_A_WRITE_FULL_CANONICAL}" == "true" ]]; then
  cp "${smoke_train_prompt}" "${train_prompt}"
  cp "${smoke_tiny_prompt}" "${tiny_prompt}"
fi

echo "[phase-a] Step 3: Convert train prompt to ${MODEL_TYPE} SFT format"
"${PYTHON_BIN}" scripts/convert_to_sft.py --input "${smoke_train_prompt}" --model "${MODEL_TYPE}"

converted_file="data/prompts/training/${MODEL_TYPE}/${train_base}_${TASK_NAME}_smoke_qwen_sft.json"
canonical_train_file="data/prompts/training/${MODEL_TYPE}/MindCube_train_${TASK_NAME}_smoke_qwen_sft.json"
canonical_tiny_prompt="data/prompts/general/MindCube_tinybench_${TASK_NAME}_smoke.jsonl"

if [[ ! -f "${converted_file}" ]]; then
  echo "Missing converted SFT file: ${converted_file}"
  exit 1
fi

mkdir -p "$(dirname "${canonical_train_file}")"
cp "${converted_file}" "${canonical_train_file}"
cp "${smoke_tiny_prompt}" "${canonical_tiny_prompt}"

if [[ "${PHASE_A_WRITE_FULL_CANONICAL}" == "true" ]]; then
  full_train_file="data/prompts/training/${MODEL_TYPE}/MindCube_train_${TASK_NAME}_qwen_sft.json"
  full_tiny_prompt="data/prompts/general/MindCube_tinybench_${TASK_NAME}.jsonl"
  cp "${converted_file}" "${full_train_file}"
  cp "${smoke_tiny_prompt}" "${full_tiny_prompt}"
fi

echo "[phase-a] Data pipeline complete"
echo "  Train prompt: ${smoke_train_prompt}"
echo "  Tinybench prompt: ${smoke_tiny_prompt}"
echo "  SFT data: ${converted_file}"
echo "  Smoke canonical SFT copy: ${canonical_train_file}"
