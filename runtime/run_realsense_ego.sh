#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
wuji_source_ros

# At 60 Hz the frame period is 16.7 ms. Use a short manual
# exposure. D415 RGB: exposure=120 (~12 ms), gain=80, verified at ~59.5 Hz.
# Override only when the scene has enough added light:
#   REALSENSE_COLOR_EXPOSURE_US=140 REALSENSE_COLOR_GAIN=128 REALSENSE_COLOR_BRIGHTNESS=56 ./runtime/run_realsense_ego.sh
exposure_us="${REALSENSE_COLOR_EXPOSURE_US:-140}"
gain="${REALSENSE_COLOR_GAIN:-128}"
brightness="${REALSENSE_COLOR_BRIGHTNESS:-56}"
node_name="/camera/camera"

ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=camera \
  camera_name:=camera \
  enable_color:=true \
  enable_depth:=false \
  rgb_camera.color_profile:="$WUJI_EGO_PROFILE" \
  rgb_camera.enable_auto_exposure:=false \
  "$@" &
launch_pid=$!

cleanup() {
  kill "$launch_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 50); do
  if ros2 param get "$node_name" rgb_camera.enable_auto_exposure >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

if ! ros2 param set "$node_name" rgb_camera.enable_auto_exposure false; then
  echo "[FAILED] RealSense 不支持 rgb_camera.enable_auto_exposure；未确认短曝光，已停止。" >&2
  exit 1
fi
if ! ros2 param set "$node_name" rgb_camera.exposure "$exposure_us"; then
  echo "[FAILED] 无法设置 rgb_camera.exposure=${exposure_us}us；未确认短曝光，已停止。" >&2
  exit 1
fi
if ! ros2 param set "$node_name" rgb_camera.gain "$gain"; then
  echo "[FAILED] 无法设置 rgb_camera.gain=${gain}；未确认短曝光，已停止。" >&2
  exit 1
fi

if ! ros2 param set "$node_name" rgb_camera.brightness "$brightness"; then
  echo "[FAILED] 无法设置 rgb_camera.brightness=${brightness}；已停止。" >&2
  exit 1
fi
echo "[OK] RealSense RGB manual exposure=${exposure_us}, gain=${gain}, brightness=${brightness}, auto_exposure=false"
wait "$launch_pid"
