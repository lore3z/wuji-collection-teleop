#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
wuji_source_ros

teleop_python="${EMG_GEORT_PYTHON:-$WUJI_PROJECT_DIR/.venv-teleop/bin/python}"
wuji_require_file "$teleop_python"
exec "$teleop_python" "$WUJI_PROJECT_DIR/wuji_l20_g20_teleop.py" \
  --control-hz 30 \
  "$@"
