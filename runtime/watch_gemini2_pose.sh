#!/usr/bin/env bash
# Live Gemini 2 IMU pose monitor.  The collector must already have started
# the official Orbbec IMU driver (WUJI_GEMINI_IMU_ENABLED=1).
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
wuji_source_ros
wuji_require_file "$WUJI_GEMINI_ORBBEC_INSTALL/setup.bash"
set +u
# shellcheck disable=SC1090
source "$WUJI_GEMINI_ORBBEC_INSTALL/setup.bash"
set -u

exec "$WUJI_PYTHON" "$runtime_dir/watch_gemini2_pose.py" \
  --topic "$WUJI_GEMINI_IMU_TOPIC" "$@"
