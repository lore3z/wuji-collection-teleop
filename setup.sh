#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$project_dir/runtime"

usage() {
  cat <<'EOF'
usage:
  ./setup.sh                  create Python env, install Python deps, build Gemini driver
  ./setup.sh --install-system install Ubuntu/ROS binary dependencies, then do the above
  ./setup.sh --check          check an existing installation without changing it

Target OS: Ubuntu 22.04 with the ROS 2 Humble apt repository configured.
EOF
}

mode=setup
case "${1:-}" in
  "") ;;
  --install-system) mode=install-system ;;
  --check) mode=check ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "$mode" == "check" ]]; then
  exec "$runtime_dir/preflight.sh" --software
fi

if [[ "$mode" == "install-system" ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[FAILED] --install-system only supports Ubuntu/Debian apt hosts." >&2
    exit 1
  fi
  sudo apt-get update
  sudo apt-get install -y \
    python3-venv python3-pip python3-colcon-common-extensions \
    python3-can python3-serial python3-yaml \
    ros-humble-realsense2-camera ros-humble-cv-bridge ros-humble-image-transport \
    libopencv-dev v4l-utils bluez can-utils
fi

if [[ ! -r /opt/ros/humble/setup.bash ]]; then
  echo "[FAILED] ROS 2 Humble not found. Install it first or run ./setup.sh --install-system" >&2
  echo "         after configuring the official ROS 2 apt repository." >&2
  exit 1
fi

python3 -m venv --system-site-packages "$project_dir/.venv"
# colcon-core currently requires setuptools < 80.  Keep this project-local
# interpreter compatible with ROS command-line tools on the same host.
"$project_dir/.venv/bin/python" -m pip install --upgrade "pip<26" "setuptools>=61,<80" wheel
"$project_dir/.venv/bin/python" -m pip install -r "$project_dir/requirements-collector.txt"
# Only RAW Myo streaming is used. pyomyo declares optional ML examples
# (xgboost/NCCL) as runtime dependencies; installing it without dependencies
# avoids downloading several hundred MB that the collector never imports.
"$project_dir/.venv/bin/python" -m pip install --no-deps "pyomyo==0.0.5"

runtime_root="$project_dir/.runtime/ros_ws"
mkdir -p "$runtime_root"
unset LD_LIBRARY_PATH || true
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u
colcon --log-base "$runtime_root/log" build \
  --base-paths "$project_dir/drivers" \
  --build-base "$runtime_root/build" \
  --install-base "$runtime_root/install" \
  --symlink-install \
  --packages-select gemini2_mjpeg_publisher

l20_ws="$project_dir/drivers/linker_hand_ros2_sdk"
if [[ -d "$l20_ws/src" ]]; then
  echo "[SETUP] Building bundled LinkerHand L20/G20 ROS driver (regular install)…"
  # The vendor package is an old ament_python package.  Its setup.py does not
  # support setuptools' --editable option, which colcon enables through
  # --symlink-install.  Use a normal install for this package only.
  PYTHONNOUSERSITE=1 colcon --log-base "$l20_ws/log" build \
  --base-paths "$l20_ws/src" \
  --build-base "$l20_ws/build" \
  --install-base "$l20_ws/install" \
  --packages-select linker_hand_ros2_sdk
fi

chmod +x \
  "$project_dir/collect.sh" "$project_dir/human_collect.sh" "$project_dir/l20.sh" "$project_dir/check.sh" "$project_dir/setup.sh" \
  "$runtime_dir"/*.sh "$runtime_dir"/*.py

if ! id -nG | tr ' ' '\n' | grep -qx dialout; then
  echo "[ACTION] 当前用户不在 dialout 组。运行以下命令后注销并重新登录："
  echo "         sudo usermod -aG dialout $USER"
fi
if ! id -nG | tr ' ' '\n' | grep -qx video; then
  echo "[ACTION] 当前用户不在 video 组。运行以下命令后注销并重新登录："
  echo "         sudo usermod -aG video $USER"
fi

echo
echo "[DONE] project-local runtime is ready."
echo "  check:   ./check.sh --software"
echo "  collect: ./collect.sh"
echo "  config:  config/collector.env (defaults), config/local.env (this computer)"
