#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
wuji_source_ros
mkdir -p "$WUJI_L20_DATA_DIR"

exec "$WUJI_PYTHON" "$WUJI_PROJECT_DIR/src/wuji_l20_sidecar_collect.py" \
  --output "$WUJI_L20_DATA_DIR/l20_teleop_pressure_120hz.zarr" \
  --fps 120 \
  "$@"
