#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
COLLECTOR="${STARVLA_DIR}/examples/LIBERO/eval_files/collect_libero_dataset.sh"

LIBERO_HOME="${LIBERO_HOME:-/mnt/benchmarks/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/mnt/miniconda3/envs/libero/bin/python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwen3fast_libero_all_edl_1e-2/checkpoints/steps_30000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"

COLLECTION_ID="${COLLECTION_ID:-rejection_test_v1}"
SEED_NAMESPACE="${SEED_NAMESPACE:-heldout_rejection_test_v1}"
SEED="${SEED:-17}"
EPISODES_PER_SUITE="${EPISODES_PER_SUITE:-100}"
EPISODE_START_INDEX="${EPISODE_START_INDEX:-10}"
TASKS_PER_SUITE=10
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/datasets/${COLLECTION_ID}}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"

SUITES=(libero_spatial libero_object libero_goal libero_10)

if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "LIBERO_PYTHON is not executable: ${LIBERO_PYTHON}"
  exit 1
fi
if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME does not exist: ${LIBERO_HOME}"
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint does not exist: ${CKPT}"
  exit 1
fi
if [[ ! -f "${COLLECTOR}" ]]; then
  echo "Collector does not exist: ${COLLECTOR}"
  exit 1
fi
if [[ ! "${EPISODES_PER_SUITE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPISODES_PER_SUITE must be a positive integer."
  exit 1
fi
if [[ ! "${EPISODE_START_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "EPISODE_START_INDEX must be a non-negative integer."
  exit 1
fi
if [[ ! "${COLLECTION_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "COLLECTION_ID must contain only letters, digits, '.', '_', or '-'."
  exit 1
fi
if [[ ! "${SEED_NAMESPACE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "SEED_NAMESPACE must contain only letters, digits, '.', '_', or '-'."
  exit 1
fi
if (( EPISODES_PER_SUITE % TASKS_PER_SUITE != 0 )); then
  echo "EPISODES_PER_SUITE must be divisible by ${TASKS_PER_SUITE}."
  exit 1
fi
if [[ "${OVERWRITE}" == "1" && "${RESUME}" == "1" ]]; then
  echo "OVERWRITE=1 and RESUME=1 cannot be used together."
  exit 1
fi

NUM_TRIALS_PER_TASK=$((EPISODES_PER_SUITE / TASKS_PER_SUITE))

for suite in "${SUITES[@]}"; do
  output_path="${DATASET_ROOT}/${COLLECTION_ID}_${suite}.hdf5"
  if [[ -e "${output_path}" && "${OVERWRITE}" != "1" && "${RESUME}" != "1" ]]; then
    echo "Dataset already exists: ${output_path}"
    echo "Use RESUME=1 to continue or choose a new COLLECTION_ID."
    exit 1
  fi
done

echo "Collecting ${EPISODES_PER_SUITE} episodes per suite into ${DATASET_ROOT}"
echo "Collection ID: ${COLLECTION_ID}; seed namespace: ${SEED_NAMESPACE}; seed: ${SEED}"
echo "Task init-state indices: [${EPISODE_START_INDEX}, $((EPISODE_START_INDEX + NUM_TRIALS_PER_TASK)))"

for suite in "${SUITES[@]}"; do
  output_path="${DATASET_ROOT}/${COLLECTION_ID}_${suite}.hdf5"
  echo "[${suite}] ${NUM_TRIALS_PER_TASK} trials/task x ${TASKS_PER_SUITE} tasks"
  STARVLA_DIR="${STARVLA_DIR}" \
  LIBERO_HOME="${LIBERO_HOME}" \
  LIBERO_PYTHON="${LIBERO_PYTHON}" \
  CKPT="${CKPT}" \
  HOST="${HOST}" \
  PORT="${PORT}" \
  TASK_SUITE_NAME="${suite}" \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
  EPISODE_START_INDEX="${EPISODE_START_INDEX}" \
  MAX_TASKS="${TASKS_PER_SUITE}" \
  SEED="${SEED}" \
  COLLECTION_ID="${COLLECTION_ID}" \
  SEED_NAMESPACE="${SEED_NAMESPACE}" \
  DATASET_OUTPUT_PATH="${output_path}" \
  OVERWRITE="${OVERWRITE}" \
  RESUME="${RESUME}" \
  bash "${COLLECTOR}"
done

echo "Collection complete: ${DATASET_ROOT}"
