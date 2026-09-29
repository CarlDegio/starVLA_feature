#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")" && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-/mnt/benchmarks/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/mnt/miniconda3/envs/libero/bin/python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwen3fast_libero_all/checkpoints/steps_30000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
DATA_SPLIT="${DATA_SPLIT:-train}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"

case "${DATA_SPLIT}" in
  train)
    EPISODE_START_INDEX="${EPISODE_START_INDEX:-0}"
    SEED="${SEED:-7}"
    ;;
  test)
    EPISODE_START_INDEX="${EPISODE_START_INDEX:-10}"
    SEED="${SEED:-17}"
    ;;
  *)
    echo "Unknown DATA_SPLIT=${DATA_SPLIT}; expected train or test."
    exit 1
    ;;
esac

COLLECTION_ID="${COLLECTION_ID:-safe_metric_${DATA_SPLIT}_seed${SEED}}"
SEED_NAMESPACE="${SEED_NAMESPACE:-safe_metric_${DATA_SPLIT}_seed${SEED}}"
DATASET_DIR="${DATASET_DIR:-${STARVLA_DIR}/examples/LIBERO/safe_pred/datasets}"
LOG_DIR="${LOG_DIR:-${STARVLA_DIR}/examples/LIBERO/safe_pred/logs}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

mkdir -p "${DATASET_DIR}" "${LOG_DIR}"
cd "${STARVLA_DIR}"

for suite in "${SUITES[@]}"; do
  log_path="${LOG_DIR}/collect_${COLLECTION_ID}_${suite}.log"
  echo "Collecting split=${DATA_SPLIT} suite=${suite}; log=${log_path}"
  STARVLA_DIR="${STARVLA_DIR}" \
  LIBERO_HOME="${LIBERO_HOME}" \
  LIBERO_PYTHON="${LIBERO_PYTHON}" \
  CKPT="${CKPT}" \
  HOST="${HOST}" \
  PORT="${PORT}" \
  TASK_SUITE_NAME="${suite}" \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
  EPISODE_START_INDEX="${EPISODE_START_INDEX}" \
  SEED="${SEED}" \
  COLLECTION_ID="${COLLECTION_ID}" \
  SEED_NAMESPACE="${SEED_NAMESPACE}" \
  DATASET_DIR="${DATASET_DIR}" \
  OVERWRITE="${OVERWRITE}" \
  RESUME="${RESUME}" \
  bash examples/LIBERO/safe_pred/collect_libero_safe.sh 2>&1 | tee "${log_path}"
done

