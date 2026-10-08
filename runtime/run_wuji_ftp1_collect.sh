#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
unset LD_LIBRARY_PATH || true
wuji_source_ros
wuji_require_file "$WUJI_PYTHON"
mkdir -p "$WUJI_DATA_DIR"
gemini_imu_topic="$WUJI_GEMINI_IMU_TOPIC"
if [[ "${WUJI_GEMINI_IMU_ENABLED:-0}" != "1" ]]; then
  gemini_imu_topic=""
fi
health_args=()
if [[ -n "$WUJI_TACTILE_HEALTH_MASK" ]]; then
  health_args+=(--tactile-health-mask "$WUJI_TACTILE_HEALTH_MASK")
fi
tracker_args=()
if [[ "${WUJI_TRACKER_ENABLED:-0}" == "1" ]]; then
  tracker_args+=(
    --tracker-camera-topic "$WUJI_TRACKER_CAMERA_TOPIC"
    --tracker-wrist-topic "$WUJI_TRACKER_WRIST_TOPIC"
    --tracker-camera-z-offset-m "$WUJI_TRACKER_CAMERA_Z_OFFSET_M"
    --max-tracker-age-ms "$WUJI_MAX_TRACKER_AGE_MS"
  )
fi
myo_args=()
if [[ "${WUJI_MYO_ENABLED:-1}" != "1" ]]; then
  myo_args+=(--no-myo)
fi
wavletech_args=()
if [[ "${WUJI_WAVLETECH_ENABLED:-1}" != "1" ]]; then
  wavletech_args+=(--no-wavletech-emg)
fi
ego_args=()
if [[ "${WUJI_SKIP_REALSENSE:-0}" == "1" && "${WUJI_VR_EGO_ENABLED:-0}" == "1" ]]; then
  ego_args+=(
    --ego-topic "$WUJI_VR_EGO_TOPIC"
    --ego-compressed
    --ego-source "${WUJI_VR_EGO_SOURCE:-PICO A9210 camera}"
  )
fi

exec "$WUJI_PYTHON" "$WUJI_PROJECT_DIR/src/wuji_glove_d435_ftp1_collect.py" \
  --output-dir "$WUJI_DATA_DIR" \
  --glove-sn "$WUJI_GLOVE_SN" \
  --glove-hz 120 \
  "${health_args[@]}" \
  --gemini-topic "$WUJI_GEMINI_TOPIC" \
  --gemini-imu-topic "$gemini_imu_topic" \
  "${myo_args[@]}" \
  "${wavletech_args[@]}" \
  "${ego_args[@]}" \
  "${tracker_args[@]}" \
  --max-glove-age-ms "$WUJI_MAX_GLOVE_AGE_MS" \
  --max-wrist-pose-age-ms "$WUJI_MAX_WRIST_POSE_AGE_MS" \
  --max-gemini-age-ms "$WUJI_MAX_GEMINI_AGE_MS" \
  --max-camera-pose-age-ms "$WUJI_MAX_CAMERA_POSE_AGE_MS" \
  --max-rgb-gap-ms "$WUJI_MAX_RGB_GAP_MS" \
  --max-missing-ratio "$WUJI_MAX_RGB_MISSING_RATIO" \
  --max-wuji-gap-ms "$WUJI_MAX_SOURCE_GAP_MS" \
  --max-joint-step-rad "$WUJI_MAX_JOINT_STEP_RAD" \
  --joint-preflight-seconds "${WUJI_JOINT_PREFLIGHT_SECONDS:-2.0}" \
  --joint-preflight-min-range-deg "${WUJI_JOINT_PREFLIGHT_MIN_RANGE_DEG:-3.0}" \
  --min-ego-sharpness "$WUJI_MIN_EGO_SHARPNESS" \
  --myo-python "$WUJI_PYTHON" \
  --myo-tty "$WUJI_MYO_TTY_RESOLVED" \
  --myo-mac "$WUJI_MYO_MAC" \
  --min-emg-hz "$WUJI_MYO_MIN_HZ" \
  --max-emg-age-ms "$WUJI_MAX_EMG_AGE_MS" \
  --max-emg-missing-ratio "$WUJI_MAX_EMG_MISSING_RATIO" \
  --max-emg-gap-ms "$WUJI_MAX_EMG_GAP_MS" \
  --myo-start-wait-s "${WUJI_MYO_START_WAIT_S:-20}" \
  --myo-silence-timeout-s "${WUJI_MYO_SILENCE_TIMEOUT_S:-0.35}" \
  --wavletech-python "$WUJI_PYTHON" \
  --wavletech-tty "$WUJI_WAVLETECH_TTY_RESOLVED" \
  --wavletech-baud "$WUJI_WAVLETECH_BAUD" \
  --min-wavletech-emg-hz "$WUJI_WAVLETECH_MIN_HZ" \
  --max-wavletech-emg-age-ms "$WUJI_MAX_WAVLETECH_EMG_AGE_MS" \
  --max-wavletech-emg-missing-ratio "$WUJI_MAX_WAVLETECH_EMG_MISSING_RATIO" \
  --max-wavletech-emg-gap-ms "$WUJI_MAX_WAVLETECH_EMG_GAP_MS" \
  --wavletech-start-wait-s "${WUJI_WAVLETECH_START_WAIT_S:-20}" \
  --wavletech-silence-timeout-s "${WUJI_WAVLETECH_SILENCE_TIMEOUT_S:-0.35}" \
  "$@"
