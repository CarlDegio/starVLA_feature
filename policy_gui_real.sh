#!/usr/bin/env bash
set -euo pipefail
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")" && pwd)}"
YAM_DIR="${YAM_DIR:-/home/tiancai/yam-abc-reproduce}"
export PYTHONPATH="${STARVLA_DIR}:${YAM_DIR}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "$YAM_DIR"
exec "${YAM_PYTHON:-${YAM_DIR}/.venv/bin/python}" -u -m deployment.real.gui "$@"
