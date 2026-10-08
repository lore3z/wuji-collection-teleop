#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
unset LD_LIBRARY_PATH || true
wuji_require_file "$WUJI_PYTHON"
exec "$WUJI_PYTHON" "$WUJI_PROJECT_DIR/wuji_tactile_health_check.py" --glove-sn "$WUJI_GLOVE_SN" "$@"
