#!/usr/bin/env zsh
set -euo pipefail

source /mnt/miniconda3/etc/profile.d/conda.sh
conda activate starvla
cd /mnt/starVLA

export CKPT=/mnt/starVLA/playground/Checkpoints/qwen3fast_libero_all_edl_1e-2/checkpoints/steps_30000_pytorch_model.pt
export GPU_ID=0
export PORT=6694
export USE_BF16=1

bash examples/LIBERO/eval_files/run_policy_server.sh
