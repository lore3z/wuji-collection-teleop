#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
unset LD_LIBRARY_PATH || true
export WUJI_SDK_LOG_LEVEL="${WUJI_SDK_LOG_LEVEL:-error}"
wuji_require_file "$WUJI_PYTHON"

output_dir="${WUJI_WAVLETECH_SKELETON_DATA_DIR:-$WUJI_PROJECT_DIR/data/wavletech_skeleton}"
mkdir -p "$output_dir"

exec "$WUJI_PYTHON" "$WUJI_PROJECT_DIR/src/wuji_wavletech_skeleton_collect.py" \
  --output-dir "$output_dir" \
  --glove-sn "$WUJI_GLOVE_SN" \
  --glove-hz 120 \
  --wavletech-python "$WUJI_PYTHON" \
  --wavletech-tty "$WUJI_WAVLETECH_TTY_RESOLVED" \
  --wavletech-baud "$WUJI_WAVLETECH_BAUD" \
  --wavletech-silence-timeout-s "$WUJI_WAVLETECH_SILENCE_TIMEOUT_S" \
  --min-emg-hz "$WUJI_WAVLETECH_MIN_HZ" \
  --max-emg-missing-ratio "$WUJI_MAX_WAVLETECH_EMG_MISSING_RATIO" \
  --max-emg-gap-ms "$WUJI_MAX_WAVLETECH_EMG_GAP_MS" \
  --max-skeleton-gap-ms "$WUJI_MAX_SOURCE_GAP_MS" \
  "$@"
