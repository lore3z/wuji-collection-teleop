#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
unset LD_LIBRARY_PATH || true
wuji_source_ros

if [[ "${WUJI_VR_EGO_MODE:-screenrecord}" == "raw" ]]; then
  wuji_require_file "$WUJI_PYTHON"
  : "${WUJI_VR_EGO_PICO_IP:?WUJI_VR_EGO_PICO_IP is required when WUJI_VR_EGO_MODE=raw}"
  exec "$WUJI_PYTHON" "$runtime_dir/pico_raw_camera_publisher.py" \
    --pico-ip "$WUJI_VR_EGO_PICO_IP" \
    --control-port "${WUJI_VR_EGO_RAW_CONTROL_PORT:-13579}" \
    --stream-port "${WUJI_VR_EGO_RAW_STREAM_PORT:-50081}" \
    --size "${WUJI_VR_EGO_RAW_SIZE:-2160x810}" \
    --fps "${WUJI_VR_EGO_RAW_FPS:-60}" \
    --bitrate "${WUJI_VR_EGO_RAW_BITRATE:-20000000}" \
    --callback-ip "${WUJI_VR_EGO_RAW_CALLBACK_IP:-auto}" \
    --topic "$WUJI_VR_EGO_TOPIC" \
    --frame-id "${WUJI_VR_EGO_RAW_FRAME_ID:-pico_vst_optical_frame}" \
    --jpeg-quality "${WUJI_VR_EGO_RAW_JPEG_QUALITY:-85}" \
    --queue-depth "${WUJI_VR_EGO_RAW_QUEUE_DEPTH:-6}"
fi

exec "$WUJI_PYTHON" "$runtime_dir/pico_vr_ego_publisher.py" \
  --serial "${WUJI_VR_EGO_ADB_SERIAL:-}" \
  --topic "$WUJI_VR_EGO_TOPIC" \
  --screen-size "$WUJI_VR_EGO_SCREEN_SIZE" \
  --bit-rate "$WUJI_VR_EGO_BIT_RATE" \
  --eye "$WUJI_VR_EGO_EYE" \
  --output-size "$WUJI_VR_EGO_OUTPUT_SIZE" \
  --policy-roi "${WUJI_VR_EGO_POLICY_ROI:-520,180,300,600}" \
  --jpeg-quality "$WUJI_VR_EGO_JPEG_QUALITY" \
  --queue-depth "${WUJI_VR_EGO_QUEUE_DEPTH:-12}" \
  --fps "$WUJI_VR_EGO_FPS"
