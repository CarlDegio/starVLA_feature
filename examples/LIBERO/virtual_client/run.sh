#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${STARVLA_ENV:-starvla}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT_DIR"
exec python -u -m examples.LIBERO.virtual_client.client "$@"
