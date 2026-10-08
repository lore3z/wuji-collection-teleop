#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"

exec "$WUJI_PYTHON" "$runtime_dir/wuji_tactile_official_calibrate.py" \
  --glove-sn "$WUJI_GLOVE_SN" "$@"
