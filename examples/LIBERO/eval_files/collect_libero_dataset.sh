#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/libero_example/checkpoints/steps_50000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
SEED="${SEED:-17}"
COLLECTION_ID="${COLLECTION_ID:-rejection_test_seed${SEED}}"
SEED_NAMESPACE="${SEED_NAMESPACE:-heldout_rejection_test_seed${SEED}}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  echo "Example: LIBERO_HOME=/path/to/LIBERO LIBERO_PYTHON=/path/to/python bash $0"
  exit 1
fi
if [[ "${OVERWRITE}" == "1" && "${RESUME}" == "1" ]]; then
  echo "OVERWRITE=1 and RESUME=1 cannot be used together."
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

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
RUN_ID="$(basename "${MODEL_ROOT}")"
CKPT_FILENAME="$(basename "${CKPT}")"
CKPT_STEM="${CKPT_FILENAME%.pt}"
DATASET_DIR="${DATASET_DIR:-${STARVLA_DIR}/examples/LIBERO/eval_files/datasets}"
DATASET_OUTPUT_PATH="${DATASET_OUTPUT_PATH:-${DATASET_DIR}/${RUN_ID}_${CKPT_STEM}_${COLLECTION_ID}_${TASK_SUITE_NAME}.hdf5}"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--args.dataset-overwrite)
fi
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--args.dataset-resume)
fi

echo "Collecting ${TASK_SUITE_NAME} uncertainty data into ${DATASET_OUTPUT_PATH}"

"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.max-tasks "${MAX_TASKS}" \
  --args.seed "${SEED}" \
  --args.collection-id "${COLLECTION_ID}" \
  --args.seed-namespace "${SEED_NAMESPACE}" \
  --args.dataset-output-path "${DATASET_OUTPUT_PATH}" \
  --args.no-save-artifacts \
  "${EXTRA_ARGS[@]}"

echo "Dataset written to ${DATASET_OUTPUT_PATH}"
