#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "$PROJECT_DIR/config/collector.env"
if [[ -f "$PROJECT_DIR/config/local.env" ]]; then
  source "$PROJECT_DIR/config/local.env"
fi
set +a

exec "$WUJI_PYTHON" "$PROJECT_DIR/runtime/wuji_joint_live_view.py" \
  --glove-sn "$WUJI_GLOVE_SN" "$@"
