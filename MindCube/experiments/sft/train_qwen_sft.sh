#!/bin/bash
set -euo pipefail

# ======================================================================
# Qwen2.5-VL SFT Training Script
# Independent training script for different MindCube tasks
# ======================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

TASK_CONFIG_FILE=${1:-""}
HARDWARE_CONFIG_FILE="${SCRIPT_DIR}/config_hardware.sh"

if [[ -n "${TASK_CONFIG_FILE}" && ! "${TASK_CONFIG_FILE}" = /* ]]; then
    if [[ ! "${TASK_CONFIG_FILE}" == experiments/sft/* ]]; then
        TASK_CONFIG_FILE="${SCRIPT_DIR}/${TASK_CONFIG_FILE}"
    else
        TASK_CONFIG_FILE="${PROJECT_ROOT}/${TASK_CONFIG_FILE}"
    fi
fi

if [[ -f "${HARDWARE_CONFIG_FILE}" ]]; then
    echo "Loading hardware configuration from: ${HARDWARE_CONFIG_FILE}"
    source "${HARDWARE_CONFIG_FILE}"
else
    echo "❌ Hardware configuration file '${HARDWARE_CONFIG_FILE}' not found!"
    echo "Please create the hardware configuration file at: experiments/sft/config_hardware.sh"
    exit 1
fi

if [[ -n "${TASK_CONFIG_FILE}" && -f "${TASK_CONFIG_FILE}" ]]; then
    echo "Loading task configuration from: ${TASK_CONFIG_FILE}"
    source "${TASK_CONFIG_FILE}"
else
    echo "Using default task configuration"
    TASK_NAME="raw_qa"
    DATASET_NAME="raw_qa"
    MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
    LEARNING_RATE=1e-5
    NUM_EPOCHS=3
    OUTPUT_BASE_DIR="experiments/sft/results"
    RUN_NAME="qwen2vl-${TASK_NAME}_sft"
    MAX_PIXELS=90000
    MIN_PIXELS=784
    MODEL_MAX_LENGTH=8192
    SAVE_STEPS=5
    SAVE_TOTAL_LIMIT=12
    MAX_STEPS=-1
fi

MAX_STEPS=${MAX_STEPS:-"-1"}
USE_BF16=${USE_BF16:-"true"}
REPORT_TO=${REPORT_TO:-"none"}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-""}

if [[ -z "${GPU_DEVICES:-}" || -z "${NUM_PROCESSES:-}" || -z "${BATCH_SIZE:-}" || -z "${GRAD_ACCUM_STEPS:-}" ]]; then
    echo "❌ Hardware configuration incomplete!"
    echo "Missing parameters: GPU_DEVICES, NUM_PROCESSES, BATCH_SIZE, or GRAD_ACCUM_STEPS"
    echo "Please check your config_hardware.sh file."
    exit 1
fi

cd "${PROJECT_ROOT}"
echo "Project root: ${PROJECT_ROOT}"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$("${PYTHON_BIN}" - <<'PY'
import random
print(random.randint(20001, 29999))
PY
)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-${NUM_PROCESSES}}
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"

deepspeed="${PROJECT_ROOT}/Qwen2.5-VL-MindCube/qwen-vl-finetune/scripts/zero3.json"
entry_file="${PROJECT_ROOT}/Qwen2.5-VL-MindCube/qwen-vl-finetune/qwenvl/train/train_qwen.py"

if [[ ! -f "${entry_file}" ]]; then
    echo "❌ Qwen training entry file not found: ${entry_file}"
    echo "Run setup first: bash scripts/phase_a/setup_qwen_sft.sh"
    exit 1
fi

output_dir="${PROJECT_ROOT}/${OUTPUT_BASE_DIR}/${TASK_NAME}/"
mkdir -p "${output_dir}"

args=(
    --deepspeed "${deepspeed}"
    --model_name_or_path "${MODEL_NAME}"
    --dataset_use "${DATASET_NAME}"
    --data_flatten True
    --tune_mm_vision True
    --tune_mm_mlp True
    --tune_mm_llm True
)

if [[ "${USE_BF16}" == "true" ]]; then
    args+=(--bf16)
fi

args+=(
    --output_dir "${output_dir}"
    --num_train_epochs "${NUM_EPOCHS}"
    --per_device_train_batch_size "${BATCH_SIZE}"
    --per_device_eval_batch_size "$((BATCH_SIZE * 2))"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --max_pixels "${MAX_PIXELS}"
    --min_pixels "${MIN_PIXELS}"
    --eval_strategy no
    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length "${MODEL_MAX_LENGTH}"
    --gradient_checkpointing True
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --run_name "${RUN_NAME}"
    --report_to "${REPORT_TO}"
)

if [[ "${MAX_STEPS}" != "-1" ]]; then
    args+=(--max_steps "${MAX_STEPS}")
fi

if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
    # Allow shell-style extra flags from config files.
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_TRAIN_ARGS})
    args+=("${extra_args[@]}")
fi

echo "==============================================================================
Training Configuration:
- Task: ${TASK_NAME}
- Model: ${MODEL_NAME}
- Dataset: ${DATASET_NAME}
- Run name: ${RUN_NAME}
- Output directory: ${output_dir}

Hardware Configuration:
- GPUs: ${CUDA_VISIBLE_DEVICES}
- Number of processes: ${NPROC_PER_NODE}
- Batch size per device: ${BATCH_SIZE}
- Gradient accumulation steps: ${GRAD_ACCUM_STEPS}
- Total effective batch size: $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))

Training Hyperparameters:
- Learning rate: ${LEARNING_RATE}
- Number of epochs: ${NUM_EPOCHS}
- Max steps: ${MAX_STEPS}
- Max pixels: ${MAX_PIXELS}
- Model max length: ${MODEL_MAX_LENGTH}
=============================================================================="

echo "Verifying MindCube datasets are available..."
cd "${PROJECT_ROOT}/experiments/sft"
if ! "${PYTHON_BIN}" patch_qwen_data.py verify; then
    echo ""
    echo "❌ MindCube datasets not found in Qwen's data configuration!"
    echo "Please run the following command first to setup the environment:"
    echo "  cd ${PROJECT_ROOT}/experiments/sft"
    echo "  python patch_qwen_data.py patch"
    echo ""
    exit 1
fi

cd "${PROJECT_ROOT}"

echo "Starting training..."
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${entry_file}" "${args[@]}"

echo "Training completed successfully!"
echo "Model saved to: ${output_dir}"
