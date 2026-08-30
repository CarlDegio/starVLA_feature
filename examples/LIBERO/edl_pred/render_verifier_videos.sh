#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-/mnt/benchmarks/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/mnt/miniconda3/envs/libero/bin/python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwen3fast_libero_all_edl_1e-2/checkpoints/steps_30000_pytorch_model.pt}"
VERIFIER_CKPT="${VERIFIER_CKPT:-${SCRIPT_DIR}/outputs/sweeps/all_suites_seed7_v1/runs/all_mlp_flat_edl_seed7/checkpoints/best.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
TASK_ID="${TASK_ID:-0}"
NUM_VIDEOS="${NUM_VIDEOS:-5}"
EPISODE_START_INDEX="${EPISODE_START_INDEX:-0}"
SEED="${SEED:-7}"
VERIFIER_DEVICE="${VERIFIER_DEVICE:-cpu}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "LIBERO_PYTHON is not executable: ${LIBERO_PYTHON}"
  exit 1
fi
if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME does not exist: ${LIBERO_HOME}"
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Policy checkpoint does not exist: ${CKPT}"
  exit 1
fi
if [[ ! -f "${VERIFIER_CKPT}" ]]; then
  echo "Verifier checkpoint does not exist: ${VERIFIER_CKPT}"
  exit 1
fi
if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "TASK_ID must be a non-negative integer."
  exit 1
fi
if [[ ! "${NUM_VIDEOS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_VIDEOS must be a positive integer."
  exit 1
fi
if [[ "${NUM_VIDEOS}" != "5" ]]; then
  echo "This presentation renderer requires NUM_VIDEOS=5."
  exit 1
fi

TASK_ID_DECIMAL=$((10#${TASK_ID}))
MODEL_ROOT="${CKPT%%/checkpoints/*}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_ROOT}/results/edl_pred_videos/${TASK_SUITE_NAME}/task_$(printf '%02d' "${TASK_ID_DECIMAL}")}"

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

extra_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  extra_args+=(--overwrite)
fi

"${LIBERO_PYTHON}" -m examples.LIBERO.edl_pred.render_verifier_videos \
  --host "${HOST}" \
  --port "${PORT}" \
  --task-suite-name "${TASK_SUITE_NAME}" \
  --task-id "${TASK_ID}" \
  --num-videos "${NUM_VIDEOS}" \
  --episode-start-index "${EPISODE_START_INDEX}" \
  --seed "${SEED}" \
  --policy-checkpoint "${CKPT}" \
  --verifier-checkpoint "${VERIFIER_CKPT}" \
  --verifier-device "${VERIFIER_DEVICE}" \
  --output-dir "${OUTPUT_DIR}" \
  "${extra_args[@]}"
