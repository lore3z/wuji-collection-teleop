#!/usr/bin/env bash
# Unified entry point for the optional physical L20/G20 stack.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
usage:
  ./scripts/l20.sh can                 reset and show can0 (all L20 programs must be stopped)
  ./scripts/l20.sh driver [ROS args]   launch the bundled LinkerHand driver
  ./scripts/l20.sh teleop [args]       launch Wuji -> L20 teleoperation
  ./scripts/l20.sh record [args]       record the L20 sidecar Zarr (driver + teleop already running)
  ./scripts/l20.sh all [args]          start driver + teleop + interactive sidecar recorder
  ./scripts/l20.sh check               validate the bundled driver installation

All code and the vendor driver source live under this project.  Run
./scripts/setup.sh once after copying this directory to build the driver.
EOF
}

command="${1:-help}"
case "$command" in
  can)
    shift
    exec "$project_dir/runtime/prepare_can0.sh" "$@"
    ;;
  driver)
    shift
    exec "$project_dir/runtime/run_l20_driver.sh" "$@"
    ;;
  teleop)
    shift
    exec "$project_dir/runtime/run_l20_teleop.sh" "$@"
    ;;
  record)
    shift
    exec "$project_dir/runtime/run_l20_sidecar_collect.sh" "$@"
    ;;
  all)
    shift
    exec "$project_dir/runtime/collect_l20_all.sh" "$@"
    ;;
  check)
    source "$project_dir/runtime/common.sh"
    wuji_source_ros
    wuji_require_file "$WUJI_L20_DRIVER_WS/install/setup.bash"
    set +u
    # shellcheck disable=SC1090
    source "$WUJI_L20_DRIVER_WS/install/setup.bash"
    set -u
    ros2 pkg prefix linker_hand_ros2_sdk
    echo "[PASS] Bundled LinkerHand L20/G20 driver is built and discoverable."
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[FAILED] unknown L20 command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
