#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"

software_only=false
skip_realsense="${WUJI_SKIP_REALSENSE:-0}"
vr_ego_enabled=false
if [[ "$skip_realsense" == "1" && "${WUJI_VR_EGO_ENABLED:-0}" == "1" ]]; then
  vr_ego_enabled=true
fi
if [[ "${1:-}" == "--software" ]]; then
  software_only=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--software]" >&2
  exit 2
fi

failures=0
check() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "[OK] $label"
  else
    echo "[FAIL] $label" >&2
    failures=$((failures + 1))
  fi
}

check "project Python: $WUJI_PYTHON" test -x "$WUJI_PYTHON"
if [[ -x "$WUJI_PYTHON" ]]; then
  check "Python acquisition modules" "$WUJI_PYTHON" -c \
    "import cv2,numpy,numcodecs,serial,wuji_sdk,zarr"
  if [[ "${WUJI_MYO_ENABLED:-1}" == "1" ]]; then
    check "Myo pyomyo module" "$WUJI_PYTHON" -c "import pyomyo"
  fi
fi

if wuji_source_ros >/dev/null 2>&1; then
  echo "[OK] ROS 2 ${WUJI_ROS_DISTRO}"
  if [[ "$skip_realsense" == "1" ]]; then
    echo "[SKIP] RealSense ROS package (human_collect mode)"
  else
    check "RealSense ROS package" ros2 pkg prefix realsense2_camera
  fi
  check "Gemini project build" test -r "$WUJI_GEMINI_INSTALL/setup.bash"
  if [[ "${WUJI_GEMINI_IMU_ENABLED:-0}" == "1" ]]; then
    check "Gemini IMU Orbbec overlay" test -r "$WUJI_GEMINI_ORBBEC_INSTALL/setup.bash"
  fi
else
  echo "[FAIL] ROS 2 ${WUJI_ROS_DISTRO}" >&2
  failures=$((failures + 1))
fi
check "v4l2 tools" command -v v4l2-ctl
if $vr_ego_enabled; then
  if [[ "${WUJI_VR_EGO_MODE:-screenrecord}" == "raw" ]]; then
    if [[ -n "${WUJI_VR_EGO_PICO_IP:-}" ]]; then
      echo "[OK] PICO raw camera target: ${WUJI_VR_EGO_PICO_IP}:${WUJI_VR_EGO_RAW_CONTROL_PORT:-13579}"
    else
      echo "[FAIL] WUJI_VR_EGO_PICO_IP is required for PICO raw camera mode" >&2
      failures=$((failures + 1))
    fi
    check "PICO raw camera Python modules" "$WUJI_PYTHON" -c "import av,cv2,rclpy"
  else
    check "PICO ADB tool" command -v adb
    check "PICO H.264 decoder" command -v ffmpeg
  fi
fi

if ! $software_only; then
if [[ "${WUJI_MYO_ENABLED:-1}" == "1" && "${WUJI_WAVLETECH_ENABLED:-1}" == "1" \
    && -e "$WUJI_MYO_TTY_RESOLVED" && -e "$WUJI_WAVLETECH_TTY_RESOLVED" \
    && "$(readlink -f "$WUJI_MYO_TTY_RESOLVED")" == "$(readlink -f "$WUJI_WAVLETECH_TTY_RESOLVED")" ]]; then
  echo "[FAIL] Myo 与 Wavletech 被配置到同一个物理串口：$(readlink -f "$WUJI_MYO_TTY_RESOLVED")" >&2
  echo "       Myo 必须使用 Bluegiga BLED112；Wavletech 当前是 QinHeng 1a86:55d3。" >&2
  failures=$((failures + 1))
fi
if [[ "${WUJI_MYO_ENABLED:-1}" == "1" ]]; then
  check "Myo BLED112 readable: $WUJI_MYO_TTY_RESOLVED" test -r "$WUJI_MYO_TTY_RESOLVED"
  check "Myo BLED112 writable: $WUJI_MYO_TTY_RESOLVED" test -w "$WUJI_MYO_TTY_RESOLVED"
else
  echo "[SKIP] Myo checks (WUJI_MYO_ENABLED=0)"
fi
if [[ "${WUJI_WAVLETECH_ENABLED:-1}" == "1" ]]; then
  check "Wavletech EMG receiver readable: $WUJI_WAVLETECH_TTY_RESOLVED" test -r "$WUJI_WAVLETECH_TTY_RESOLVED"
  check "Wavletech EMG receiver writable: $WUJI_WAVLETECH_TTY_RESOLVED" test -w "$WUJI_WAVLETECH_TTY_RESOLVED"
else
  echo "[SKIP] Wavletech checks (WUJI_WAVLETECH_ENABLED=0)"
fi
  if $vr_ego_enabled; then
    if [[ "${WUJI_VR_EGO_MODE:-screenrecord}" == "raw" ]]; then
      echo "[SKIP] ADB device check (PICO native raw camera mode)"
    elif adb devices 2>/dev/null | awk 'NR > 1 && $2 == "device" {count++} END {exit count == 1 ? 0 : 1}'; then
      echo "[OK] exactly one authorized PICO ADB device"
    else
      echo "[FAIL] expected exactly one authorized PICO in 'adb devices'" >&2
      failures=$((failures + 1))
    fi
  elif [[ "$skip_realsense" == "1" ]]; then
    echo "[SKIP] V4L2 video device check (human_collect mode, ego disabled)"
  elif compgen -G '/dev/video*' >/dev/null; then
    echo "[OK] V4L2 video devices exist"
  else
    echo "[FAIL] no /dev/video* camera device" >&2
    failures=$((failures + 1))
  fi
fi

if (( failures > 0 )); then
  echo "[FAILED] preflight found $failures problem(s)." >&2
  exit 1
fi
echo "[PASS] preflight"
