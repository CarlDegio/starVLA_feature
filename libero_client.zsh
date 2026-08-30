#!/usr/bin/env zsh
set -euo pipefail

source /mnt/miniconda3/etc/profile.d/conda.sh
conda activate libero
cd /mnt/starVLA

export LIBERO_HOME=/mnt/benchmarks/LIBERO
# export LIBERO_PYTHON="$(which python)"
export CKPT="${CKPT:-/mnt/starVLA/playground/Checkpoints/qwen3fast_libero_all_edl_1e-2/checkpoints/steps_30000_pytorch_model.pt}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-6694}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
export RUN_MODE="${RUN_MODE:-collect}" #collect, eval
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

case "${RUN_MODE}" in
  eval)
    RUN_SCRIPT="examples/LIBERO/eval_files/eval_libero.sh"
    ;;
  collect)
    RUN_SCRIPT="examples/LIBERO/eval_files/collect_libero_dataset.sh"
    ;;
  *)
    echo "Unknown RUN_MODE=${RUN_MODE}; expected eval or collect."
    exit 1
    ;;
esac

for suite in libero_spatial libero_object libero_goal libero_10; do
  TASK_SUITE_NAME="$suite" bash "${RUN_SCRIPT}"
done

# for suite in libero_spatial libero_object; do
#   TASK_SUITE_NAME="$suite" bash "${RUN_SCRIPT}"
# done
