#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
wuji_require_file "$WUJI_PYTHON"

exec "$WUJI_PYTHON" "$runtime_dir/myo_emg_live_plot.py" \
  --python "$WUJI_PYTHON" \
  --tty "$WUJI_MYO_TTY_RESOLVED" \
  --mac "$WUJI_MYO_MAC" \
  "$@"
