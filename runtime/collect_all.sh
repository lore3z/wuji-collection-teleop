#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"

mkdir -p "$WUJI_RUNTIME_DIR/logs" "$WUJI_DATA_DIR"
# Hold a non-blocking lock for the entire collection session. Starting a
# second launcher used to terminate the first session's camera publishers and
# then fail on its still-running EMG child, leaving a deceptive half-alive
# collector. Refuse the second launch before touching any process or device.
exec 9>"$WUJI_RUNTIME_DIR/collect_all.lock"
if ! flock -n 9; then
  echo "[FAILED] 已有 human_collect/collect_all 采集会话在运行；请先在原终端退出。" >&2
  exit 1
fi
ego_pid=""
ego_rviz_pid=""
main_pid=""
tracker_pid=""
tracker_mode=""
ego_alive=true
main_alive=true
tracker_alive=true
skip_realsense="${WUJI_SKIP_REALSENSE:-0}"
vr_ego_enabled=false
if [[ "$skip_realsense" == "1" && "${WUJI_VR_EGO_ENABLED:-0}" == "1" ]]; then
  vr_ego_enabled=true
fi
cleanup_stale_runtime_processes() {
  # A launcher terminated outside its EXIT trap can leave setsid children
  # behind. Remove only nodes owned by this collector before binding devices.
  local pattern pid pgid
  for pattern in \
    "$WUJI_PROJECT_DIR/wuji_glove_d435_ftp1_collect.py" \
    "$runtime_dir/myo_200hz_stream.py" \
    "$runtime_dir/wavletech_serial_stream.py" \
    "$runtime_dir/pico_raw_camera_publisher.py" \
    "$runtime_dir/pico_vr_ego_publisher.py" \
    "gemini2_mjpeg_publisher/lib/gemini2_mjpeg_node" \
    "pico_install/lib/pico_input/pico_dual_tracker_pose_publisher"; do
    while read -r pid pgid; do
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      [[ "$pid" == "$$" ]] && continue
      kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      # Some ROS Python entry points catch SIGTERM and remain stuck while
      # shutting down rclpy. They must not survive into the new session and
      # publish a second copy of the same Tracker stream.
      for _ in {1..10}; do
        kill -0 -- "-$pgid" 2>/dev/null || break
        sleep 0.1
      done
      if kill -0 -- "-$pgid" 2>/dev/null; then
        kill -KILL -- "-$pgid" 2>/dev/null || true
      fi
    done < <(ps -eo pid=,pgid=,comm=,args= | awk -v pat="$pattern" \
      '$3 ~ /^(python|python3|gemini2_mjpeg)/ && index($0, pat) {print $1, $2}')
  done
  sleep 0.5
}
terminate_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  # Each local launcher is started with setsid, so its ROS wrapper and the
  # actual node share this process group. Killing only the wrapper leaves the
  # camera device occupied by an orphaned child.
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  terminate_group "$main_pid"
  terminate_group "$ego_rviz_pid"
  terminate_group "$ego_pid"
  if [[ "$tracker_mode" == "docker" && -n "$tracker_pid" ]] \
      && docker inspect --format '{{.State.Running}}' "${WUJI_PICO_CONTAINER}" 2>/dev/null | grep -qx true; then
    docker exec "${WUJI_PICO_CONTAINER}" bash -lc \
      "pkill -TERM -f '[p]ico_dual_tracker_pose_publisher' 2>/dev/null || true" \
      >/dev/null 2>&1 || true
  fi
  terminate_group "$tracker_pid"
  if [[ -n "$main_pid" ]]; then wait "$main_pid" 2>/dev/null || true; fi
  if [[ -n "$ego_pid" ]]; then wait "$ego_pid" 2>/dev/null || true; fi
  if [[ -n "$ego_rviz_pid" ]]; then wait "$ego_rviz_pid" 2>/dev/null || true; fi
  if [[ -n "$tracker_pid" ]]; then wait "$tracker_pid" 2>/dev/null || true; fi
  exit "$code"
}
trap cleanup EXIT INT TERM

echo "[1/6] 软件与 USB 预检"
cleanup_stale_runtime_processes
"$runtime_dir/preflight.sh"

echo "[2/6] Wuji 官方 526-taxel 合同检查"
contract_ok=false
for attempt in 1 2 3; do
  if "$WUJI_PYTHON" "$runtime_dir/wuji_tactile_contract_check.py" \
      --glove-sn "$WUJI_GLOVE_SN" --seconds 2; then
    contract_ok=true
    break
  fi
  if (( attempt < 3 )); then
    echo "[WAIT] Wuji 链路未稳定，3 秒后重试（$attempt/3）..."
    sleep 3
  fi
done
if [[ "$contract_ok" != true ]]; then
  echo "[FAILED] Wuji 手套连续 3 次无法稳定通信。" >&2
  echo "         当前已确认 192.168.1.101 存在丢包；请重插手套网线/供电，等指示灯稳定后重跑 ./human_collect.sh。" >&2
  exit 1
fi

echo "[3/6] 验证双 EMG 接收链路"
if [[ "${WUJI_MYO_ENABLED:-1}" == "1" ]]; then
  if fuser -s "$WUJI_MYO_TTY_RESOLVED" 2>/dev/null; then
    echo "[FAILED] Myo BLED112 $WUJI_MYO_TTY_RESOLVED 正被其他进程占用：" >&2
    fuser -v "$WUJI_MYO_TTY_RESOLVED" >&2 || true
    exit 1
  fi
  echo "[PASS] Myo BLED112 端口空闲：$WUJI_MYO_TTY_RESOLVED"
else
  echo "[SKIP] Myo EMG（WUJI_MYO_ENABLED=0）"
fi
if [[ "${WUJI_WAVLETECH_ENABLED:-1}" == "1" ]]; then
  if fuser -s "$WUJI_WAVLETECH_TTY_RESOLVED" 2>/dev/null; then
    echo "[FAILED] Wavletech 串口 $WUJI_WAVLETECH_TTY_RESOLVED 正被其他进程占用：" >&2
    fuser -v "$WUJI_WAVLETECH_TTY_RESOLVED" >&2 || true
    exit 1
  fi
  echo "[PASS] Wavletech 端口空闲：$WUJI_WAVLETECH_TTY_RESOLVED @ $WUJI_WAVLETECH_BAUD baud"
else
  echo "[SKIP] Wavletech EMG（WUJI_WAVLETECH_ENABLED=0）"
fi

if [[ "$skip_realsense" == "1" ]]; then
  if $vr_ego_enabled; then
    if [[ "${WUJI_VR_EGO_MODE:-screenrecord}" == "raw" ]]; then
      echo "[4/6] 启动 PICO 原生双目相机（${WUJI_VR_EGO_RAW_SIZE:-2160x810}@${WUJI_VR_EGO_RAW_FPS:-60}，Tracker UI 独立）"
    else
      echo "[4/6] 启动 PICO VR 第一人称${WUJI_VR_EGO_EYE:-left}眼视频（screenrecord 回退模式）"
    fi
    setsid "$runtime_dir/run_pico_vr_ego.sh" \
      >"$WUJI_RUNTIME_DIR/logs/pico_vr_ego.log" 2>&1 &
    ego_pid=$!
    if [[ "${WUJI_VR_EGO_RVIZ_ENABLED:-1}" == "1" ]]; then
      echo "      RViz 原始图像话题：${WUJI_VR_EGO_RVIZ_TOPIC:-/pico/ego/image_raw}"
      setsid "$runtime_dir/run_pico_rviz_republish.sh" \
        >"$WUJI_RUNTIME_DIR/logs/pico_rviz_republish.log" 2>&1 &
      ego_rviz_pid=$!
    fi
  else
    echo "[4/6] 跳过第一人称相机（human_collect 模式）"
  fi
else
  echo "[4/6] 启动第一人称 RealSense"
  setsid "$runtime_dir/run_realsense_ego.sh" \
    >"$WUJI_RUNTIME_DIR/logs/realsense.log" 2>&1 &
  ego_pid=$!
fi

echo "[5/6] 启动第三人称 Gemini"
setsid "$runtime_dir/run_gemini_main.sh" \
  >"$WUJI_RUNTIME_DIR/logs/gemini.log" 2>&1 &
main_pid=$!

wuji_source_ros
set +u
# shellcheck disable=SC1090
source "$WUJI_GEMINI_INSTALL/setup.bash"
set -u
if [[ "${WUJI_TRACKER_ENABLED:-0}" == "1" ]]; then
  if [[ -n "${WUJI_PICO_SETUP_BASH:-}" ]]; then
    wuji_require_file "$WUJI_PICO_SETUP_BASH"
    set +u
    # shellcheck disable=SC1090
    source "$WUJI_PICO_SETUP_BASH"
    set -u
  fi
  if [[ "${WUJI_PICO_DOCKER_ENABLED:-0}" == "1" ]] && [[ "${WUJI_PICO_DOCKER_AUTOSTART:-0}" == "1" ]] \
      && command -v docker >/dev/null 2>&1 && docker inspect "${WUJI_PICO_CONTAINER}" >/dev/null 2>&1 \
      && [[ "$(docker inspect --format '{{.State.Running}}' "${WUJI_PICO_CONTAINER}")" != "true" ]]; then
    echo "[5/6] 启动 PICO Tracker 容器 ${WUJI_PICO_CONTAINER}"
    docker start "${WUJI_PICO_CONTAINER}" >/dev/null
    sleep 2
  fi
  if ros2 pkg prefix pico_input >/dev/null 2>&1; then
    tracker_mode="host"
  elif [[ "${WUJI_PICO_DOCKER_ENABLED:-0}" == "1" ]] && command -v docker >/dev/null 2>&1 \
      && docker inspect "${WUJI_PICO_CONTAINER}" >/dev/null 2>&1 \
      && [[ "$(docker inspect --format '{{.State.Running}}' "${WUJI_PICO_CONTAINER}")" == "true" ]] \
      && docker exec "${WUJI_PICO_CONTAINER}" test -r "${WUJI_PICO_CONTAINER_SETUP}"; then
    tracker_mode="docker"
  else
    echo "[FAILED] 找不到 ROS 包 pico_input；请配置主机 overlay WUJI_PICO_SETUP_BASH，或启动容器 ${WUJI_PICO_CONTAINER}" >&2
    exit 1
  fi
  echo "[5/6] 启动 PICO 双 Tracker（${WUJI_TRACKER_PUBLISH_RATE_HZ} Hz）"
  if [[ "$tracker_mode" == "host" ]]; then
    # The vendored xrobotoolkit_sdk extension is installed under ~/.local and
    # links against its companion libraries. wuji_source_ros intentionally
    # strips LD_LIBRARY_PATH to avoid ROS/Conda collisions, so restore only
    # this SDK path for the Tracker process.
    setsid env "LD_LIBRARY_PATH=$HOME/.local/lib:${LD_LIBRARY_PATH:-}" \
      ros2 run pico_input pico_dual_tracker_pose_publisher \
      --ros-args \
      -p "wrist_serial:=${WUJI_TRACKER_WRIST_SERIAL}" \
      -p "upper_arm_serial:=${WUJI_TRACKER_CAMERA_SERIAL}" \
      -p "publish_rate_hz:=${WUJI_TRACKER_PUBLISH_RATE_HZ}" \
      >"$WUJI_RUNTIME_DIR/logs/pico_trackers.log" 2>&1 &
  else
    setsid docker exec "${WUJI_PICO_CONTAINER}" bash -lc \
      "set -e; source /opt/ros/${WUJI_ROS_DISTRO}/setup.bash; source '${WUJI_PICO_CONTAINER_SETUP}'; exec ros2 run pico_input pico_dual_tracker_pose_publisher --ros-args -p wrist_serial:=${WUJI_TRACKER_WRIST_SERIAL} -p upper_arm_serial:=${WUJI_TRACKER_CAMERA_SERIAL} -p publish_rate_hz:=${WUJI_TRACKER_PUBLISH_RATE_HZ}" \
      >"$WUJI_RUNTIME_DIR/logs/pico_trackers.log" 2>&1 &
  fi
  tracker_pid=$!
  # Keep the publisher under observation long enough for the Python entry
  # point to load and reject invalid ROS parameters. Merely seeing the newly
  # spawned docker/ros2 wrapper alive once is not a successful startup.
  tracker_started=true
  for _ in {1..12}; do
    if ! kill -0 "$tracker_pid" 2>/dev/null; then
      tracker_started=false
      break
    fi
    sleep 0.25
  done
  if [[ "$tracker_started" != true ]]; then
    echo "[FAILED] PICO Tracker publisher 启动失败；日志：$WUJI_RUNTIME_DIR/logs/pico_trackers.log" >&2
    exit 1
  fi
fi
wait_timeout=25
if $vr_ego_enabled && [[ "${WUJI_VR_EGO_MODE:-screenrecord}" == "raw" ]]; then
  # PICO CameraHandle may need one native-camera session reset after the app
  # has been left in a half-open state. Give the bridge enough time to retry
  # without weakening the requirement that a fresh RGB frame arrives.
  wait_timeout=60
fi
wait_args=(--main "$WUJI_MAIN_TOPIC" --timeout "$wait_timeout")
if $vr_ego_enabled; then
  wait_args+=(--ego "$WUJI_VR_EGO_TOPIC" --ego-compressed)
elif [[ "$skip_realsense" != "1" ]]; then
  wait_args+=(--ego "$WUJI_EGO_TOPIC")
fi
if [[ "${WUJI_TRACKER_ENABLED:-0}" == "1" ]]; then
  wait_args+=(
    --tracker-camera "$WUJI_TRACKER_CAMERA_TOPIC"
    --tracker-wrist "$WUJI_TRACKER_WRIST_TOPIC"
  )
fi
if ! "$WUJI_PYTHON" "$runtime_dir/wait_for_rgb_topics.py" "${wait_args[@]}"; then
  echo "[FAILED] 相机没有按要求出图。日志：" >&2
  if $vr_ego_enabled; then
    echo "  $WUJI_RUNTIME_DIR/logs/pico_vr_ego.log" >&2
  elif [[ "$skip_realsense" != "1" ]]; then
    echo "  $WUJI_RUNTIME_DIR/logs/realsense.log" >&2
  fi
  echo "  $WUJI_RUNTIME_DIR/logs/gemini.log" >&2
  if [[ "${WUJI_TRACKER_ENABLED:-0}" == "1" ]]; then
    echo "  $WUJI_RUNTIME_DIR/logs/pico_trackers.log" >&2
  fi
  exit 1
fi

if [[ "$skip_realsense" == "1" && "$vr_ego_enabled" != true ]]; then
  main_alive=true
  kill -0 "$main_pid" 2>/dev/null || main_alive=false
  tracker_alive=true
  if [[ "${WUJI_TRACKER_ENABLED:-0}" == "1" ]]; then
    kill -0 "$tracker_pid" 2>/dev/null || tracker_alive=false
  fi
  camera_processes_ok=$([[ "$main_alive" == true && "$tracker_alive" == true ]] && echo true || echo false)
else
  ego_alive=true
  main_alive=true
  kill -0 "$ego_pid" 2>/dev/null || ego_alive=false
  kill -0 "$main_pid" 2>/dev/null || main_alive=false
  tracker_alive=true
  if [[ "${WUJI_TRACKER_ENABLED:-0}" == "1" ]]; then
    kill -0 "$tracker_pid" 2>/dev/null || tracker_alive=false
  fi
  camera_processes_ok=$([[ "$ego_alive" == true && "$main_alive" == true && "$tracker_alive" == true ]] && echo true || echo false)
fi
if [[ "$camera_processes_ok" != true ]]; then
  echo "[FAILED] 至少一个采集进程提前退出；查看以下对应日志：" >&2
  if [[ "$ego_alive" != true ]]; then
    if $vr_ego_enabled; then
      echo "  PICO VR ego: $WUJI_RUNTIME_DIR/logs/pico_vr_ego.log" >&2
    else
      echo "  RealSense: $WUJI_RUNTIME_DIR/logs/realsense.log" >&2
    fi
  fi
  if [[ "$main_alive" != true ]]; then
    echo "  Gemini: $WUJI_RUNTIME_DIR/logs/gemini.log" >&2
  fi
  if [[ "${WUJI_TRACKER_ENABLED:-0}" == "1" && "$tracker_alive" != true ]]; then
    echo "  $WUJI_RUNTIME_DIR/logs/pico_trackers.log" >&2
  fi
  exit 1
fi

echo "[6/6] 启动 Wuji + FTP-1 采集器（Myo=$([[ "${WUJI_MYO_ENABLED:-1}" == "1" ]] && echo on || echo off)，Wavletech=$([[ "${WUJI_WAVLETECH_ENABLED:-1}" == "1" ]] && echo on || echo off)）"
echo "数据目录：$WUJI_DATA_DIR"
collector_args=("$@")
if [[ "$skip_realsense" == "1" && "$vr_ego_enabled" != true ]]; then
  collector_args+=(--no-ego-camera)
fi
"$runtime_dir/run_wuji_ftp1_collect.sh" "${collector_args[@]}"
