#!/usr/bin/env bash
set -euo pipefail
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")" && pwd)}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${STARVLA_ENV:-starvla}"
cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
exec "${STARVLA_PYTHON:-python}" -u -m deployment.real.infer "$@"
