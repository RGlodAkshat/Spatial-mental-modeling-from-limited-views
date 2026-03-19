#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_NAME="${TASK_NAME:-plain_cgmap_ffr_out}"
TRAIN_SUBSET="${TRAIN_SUBSET:-data/raw/subsets/MindCube_train_50.jsonl}"
TINY_SUBSET="${TINY_SUBSET:-data/raw/subsets/MindCube_tinybench_50.jsonl}"
MODEL_TYPE="${MODEL_TYPE:-qwen2.5vl}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/data_processing.py --input "${TRAIN_SUBSET}" --task full_pipeline
"${PYTHON_BIN}" scripts/data_processing.py --input "${TINY_SUBSET}" --task full_pipeline

train_base="$(basename "${TRAIN_SUBSET}" .jsonl)"
tiny_base="$(basename "${TINY_SUBSET}" .jsonl)"
train_scaffold="data/scaffold/all/${train_base}.jsonl"
tiny_scaffold="data/scaffold/all/${tiny_base}.jsonl"
smoke_train_prompt="data/prompts/general/${train_base}_${TASK_NAME}_smoke.jsonl"
smoke_tiny_prompt="data/prompts/general/${tiny_base}_${TASK_NAME}_smoke.jsonl"

"${PYTHON_BIN}" scripts/generate_prompts.py --input "${train_scaffold}" --task "${TASK_NAME}" --output "${smoke_train_prompt}"
"${PYTHON_BIN}" scripts/generate_prompts.py --input "${tiny_scaffold}" --task "${TASK_NAME}" --output "${smoke_tiny_prompt}"
"${PYTHON_BIN}" scripts/convert_to_sft.py --input "${smoke_train_prompt}" --model "${MODEL_TYPE}"
