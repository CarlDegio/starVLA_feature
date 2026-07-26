#!/usr/bin/env zsh
set -euo pipefail

source /mnt/miniconda3/etc/profile.d/conda.sh
conda activate libero
cd /mnt/starVLA

export LIBERO_HOME=/mnt/benchmarks/LIBERO
# export LIBERO_PYTHON="$(which python)"
export CKPT=/mnt/starVLA/playground/Checkpoints/qwen3fast_libero_all_edl_1e-2/checkpoints/steps_30000_pytorch_model.pt
export HOST=127.0.0.1
export PORT=6694
export NUM_TRIALS_PER_TASK=10
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# for suite in libero_spatial libero_object libero_goal libero_10; do
#   TASK_SUITE_NAME="$suite" bash examples/LIBERO/eval_files/eval_libero.sh
# done

for suite in libero_spatial libero_object; do
  TASK_SUITE_NAME="$suite" bash examples/LIBERO/eval_files/eval_libero.sh
done
