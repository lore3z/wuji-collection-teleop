#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
wuji_source_ros
if [[ "${WUJI_GEMINI_IMU_ENABLED:-0}" == "1" ]]; then
  wuji_require_file "$WUJI_GEMINI_ORBBEC_INSTALL/setup.bash"
  set +u
  # shellcheck disable=SC1090
  source "$WUJI_GEMINI_ORBBEC_INSTALL/setup.bash"
  set -u
  # The Orbbec SDK requires an image profile to start its IMU pipeline. It
  # also owns the whole Gemini USB device, so it must be the only Gemini
  # driver; a V4L2 MJPEG process alongside it makes the camera reset/retry.
  # MJPG + a compressed subscriber forwards camera JPEG without raw RGB decode.
  exec ros2 launch orbbec_camera gemini2.launch.py \
    camera_name:=gemini2 \
    enable_color:=true color_format:=MJPG \
    color_width:="$WUJI_GEMINI_RGB_WIDTH" color_height:="$WUJI_GEMINI_RGB_HEIGHT" color_fps:="$WUJI_GEMINI_RGB_FPS" \
    enable_depth:=false enable_ir:=false \
    enable_point_cloud:=false enable_accel:=true accel_rate:=200hz \
    enable_gyro:=true gyro_rate:=200hz enable_sync_output_accel_gyro:=true \
    publish_tf:=false
fi

wuji_require_file "$WUJI_GEMINI_INSTALL/setup.bash"
set +u
# shellcheck disable=SC1090
source "$WUJI_GEMINI_INSTALL/setup.bash"
set -u
exec ros2 run gemini2_mjpeg_publisher gemini2_mjpeg_node \
  --ros-args \
  -p device:="$WUJI_GEMINI_DEVICE" \
  -p width:="$WUJI_GEMINI_RGB_WIDTH" \
  -p height:="$WUJI_GEMINI_RGB_HEIGHT" \
  -p fps:="$WUJI_GEMINI_RGB_FPS" \
  "$@"
