#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT"
exec "${PYTHON:-python}" "$SCRIPT_DIR/convert_edl_real.py" "$@"
