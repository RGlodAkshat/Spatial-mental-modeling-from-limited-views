#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_DIR="${RUNS_DIR:-runs/phase_a}"
BOOTSTRAP_SKIP_INSTALL="${BOOTSTRAP_SKIP_INSTALL:-false}"
PHASE_A_WRITE_FULL_CANONICAL="${PHASE_A_WRITE_FULL_CANONICAL:-false}"
PHASE_A_RUN_QWEN_SETUP="${PHASE_A_RUN_QWEN_SETUP:-false}"
PHASE_A_RUN_TRAINING="${PHASE_A_RUN_TRAINING:-false}"
RUN_COMMAND="${RUN_COMMAND:-BOOTSTRAP_SKIP_INSTALL=${BOOTSTRAP_SKIP_INSTALL} PHASE_A_WRITE_FULL_CANONICAL=${PHASE_A_WRITE_FULL_CANONICAL} PHASE_A_RUN_QWEN_SETUP=${PHASE_A_RUN_QWEN_SETUP} PHASE_A_RUN_TRAINING=${PHASE_A_RUN_TRAINING} bash scripts/phase_a/run_phase_a_smoke.sh}"
if [[ "${PHASE_A_RUN_TRAINING}" == "true" && "${PHASE_A_RUN_QWEN_SETUP}" != "true" ]]; then
  echo "[phase-a] PHASE_A_RUN_TRAINING=true requires Qwen setup; enabling PHASE_A_RUN_QWEN_SETUP=true"
  PHASE_A_RUN_QWEN_SETUP="true"
fi


# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate" 2>/dev/null || true

cd "${PROJECT_ROOT}"

CONFIG_JSON='{"subset_train":50,"subset_tinybench":50,"task":"plain_cgmap_ffr_out"}'
RUN_ID="$(${PYTHON_BIN} scripts/phase_a/track_run.py start --project-root "${PROJECT_ROOT}" --runs-dir "${RUNS_DIR}" --config-json "${CONFIG_JSON}" --command "${RUN_COMMAND}")"

echo "[phase-a] Run ID: ${RUN_ID}"

stage() {
  local stage_name="$1"
  local stage_status="$2"
  local stage_message="${3:-}"
  local stage_artifact="${4:-}"
  "${PYTHON_BIN}" scripts/phase_a/track_run.py stage \
    --runs-dir "${RUNS_DIR}" \
    --run-id "${RUN_ID}" \
    --stage "${stage_name}" \
    --status "${stage_status}" \
    --message "${stage_message}" \
    --artifact "${stage_artifact}"
}

finalize() {
  local status="$1"
  local note="${2:-}"
  "${PYTHON_BIN}" scripts/phase_a/track_run.py finish \
    --runs-dir "${RUNS_DIR}" \
    --run-id "${RUN_ID}" \
    --status "${status}" \
    --notes "${note}"
}

trap 'stage "pipeline" "failed" "Unexpected failure"; finalize "failed" "Unexpected failure"' ERR

stage "bootstrap" "started"
if [[ "${BOOTSTRAP_SKIP_INSTALL}" == "true" ]]; then
  bash scripts/phase_a/bootstrap.sh --skip-install
else
  bash scripts/phase_a/bootstrap.sh
fi
stage "bootstrap" "success"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  export PYTHON_BIN
fi

if [[ ! -f "data/raw/MindCube_train.jsonl" || ! -f "data/raw/MindCube_tinybench.jsonl" ]]; then
  stage "download_data" "started"
  bash scripts/bash_scripts/download_data.bash
  stage "download_data" "success"
else
  stage "download_data" "skipped" "Raw files already exist"
fi

stage "subset" "started"
"${PYTHON_BIN}" scripts/phase_a/make_deterministic_subset.py
stage "subset" "success" "Deterministic subsets ready" "data/manifests/phase_a_deterministic_50_50_manifest.json"

stage "data_pipeline" "started"
PHASE_A_WRITE_FULL_CANONICAL="${PHASE_A_WRITE_FULL_CANONICAL}" PYTHON_BIN="${PYTHON_BIN}" bash scripts/phase_a/run_data_pipeline.sh
stage "data_pipeline" "success" "Prompt + SFT conversion complete"

if [[ "${PHASE_A_RUN_QWEN_SETUP}" == "true" ]]; then
  stage "qwen_setup" "started"
  PYTHON_BIN="${PYTHON_BIN}" QWEN_SETUP_REQUIRED=true bash scripts/phase_a/setup_qwen_sft.sh
  stage "qwen_setup" "success"
else
  stage "qwen_setup" "skipped" "Optional on CPU smoke"
fi

if [[ "${PHASE_A_RUN_TRAINING}" == "true" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  stage "train_smoke" "started"
  bash experiments/sft/train_qwen_sft.sh experiments/sft/config_plain_cgmap_ffr_out_smoke.sh
  stage "train_smoke" "success" "Smoke training done" "checkpoints/sft/smoke/plain_cgmap_ffr_out_smoke"

  stage "eval_checkpoints" "started"
  "${PYTHON_BIN}" scripts/phase_a/eval_checkpoints.py \
    --task plain_cgmap_ffr_out_smoke \
    --checkpoints-root checkpoints/sft/smoke/plain_cgmap_ffr_out_smoke \
    --input-file data/prompts/general/MindCube_tinybench_plain_cgmap_ffr_out_smoke.jsonl \
    --results-dir data/results/sft/smoke/plain_cgmap_ffr_out_smoke \
    --eval-dir data/evaluate/sft/smoke/plain_cgmap_ffr_out_smoke
  stage "eval_checkpoints" "success" "Checkpoint evaluation complete" "data/evaluate/sft/plain_cgmap_ffr_out_smoke/best_checkpoint_summary.csv"
  finalize "success" "Full smoke pipeline complete"
else
  stage "train_smoke" "skipped" "Training intentionally disabled in Phase A smoke"
  stage "eval_checkpoints" "skipped" "No checkpoints yet"
  finalize "partial" "Data pipeline complete; training/eval skipped by default"
fi

echo "[phase-a] Done. Run metadata: ${RUNS_DIR}/${RUN_ID}.json"
