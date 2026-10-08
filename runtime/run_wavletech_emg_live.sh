#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"

exec "$WUJI_PYTHON" -u "$runtime_dir/wavletech_emg_live_plot.py" \
  --python "$WUJI_PYTHON" \
  --tty "$WUJI_WAVLETECH_TTY_RESOLVED" \
  --baud "$WUJI_WAVLETECH_BAUD" \
  "$@"
