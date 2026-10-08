#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
wuji_source_ros
wuji_require_file "$WUJI_L20_DRIVER_WS/install/setup.bash"
set +u
# shellcheck disable=SC1090
source "$WUJI_L20_DRIVER_WS/install/setup.bash"
set -u

exec ros2 launch linker_hand_ros2_sdk linker_hand.launch.py \
  control_hz:=120.0 \
  state_publish_hz:=120.0 \
  feedback_hz:=40.0 \
  realtime_g20:=true \
  tactile_scan_mode:=full_scan \
  tactile_scan_hz:=40.0 \
  tactile_reply_wait_ms:=3.0 \
  g20_fast_feedback:=true \
  feedback_reply_wait_ms:=4.0 \
  "$@"
