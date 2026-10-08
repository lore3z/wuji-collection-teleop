#!/usr/bin/env bash
# Start a complete physical L20 demonstration stack from one terminal.
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
wuji_source_ros
wuji_require_file "$WUJI_L20_DRIVER_WS/install/setup.bash"

if ! ip link show can0 2>/dev/null | grep -q 'UP'; then
  echo "[FAILED] can0 is not UP. Stop every L20/GUI process, then run: ./l20.sh can" >&2
  exit 1
fi

log_dir="$WUJI_RUNTIME_DIR/l20"
mkdir -p "$log_dir"
stamp="$(date +%Y%m%d_%H%M%S)"
driver_log="$log_dir/driver_${stamp}.log"
teleop_log="$log_dir/teleop_${stamp}.log"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  [[ -n "${teleop_pid:-}" ]] && kill "$teleop_pid" 2>/dev/null || true
  [[ -n "${driver_pid:-}" ]] && kill "$driver_pid" 2>/dev/null || true
  wait "${teleop_pid:-}" 2>/dev/null || true
  wait "${driver_pid:-}" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "[L20] Starting bundled LinkerHand driver; log: $driver_log"
"$runtime_dir/run_l20_driver.sh" >"$driver_log" 2>&1 &
driver_pid=$!

for _ in $(seq 1 50); do
  if ! kill -0 "$driver_pid" 2>/dev/null; then
    echo "[FAILED] L20 driver exited. Last log lines:" >&2
    tail -n 40 "$driver_log" >&2 || true
    exit 1
  fi
  if ros2 topic list 2>/dev/null | grep -qx '/cb_right_hand_control_cmd'; then
    break
  fi
  sleep 0.2
done

if ! ros2 topic list 2>/dev/null | grep -qx '/cb_right_hand_control_cmd'; then
  echo "[FAILED] L20 driver did not publish its control topic within 10 s." >&2
  tail -n 40 "$driver_log" >&2 || true
  exit 1
fi

echo "[L20] Starting Wuji -> L20 teleoperation; log: $teleop_log"
"$runtime_dir/run_l20_teleop.sh" >"$teleop_log" 2>&1 &
teleop_pid=$!
sleep 1
if ! kill -0 "$teleop_pid" 2>/dev/null; then
  echo "[FAILED] Teleoperation exited. Last log lines:" >&2
  tail -n 60 "$teleop_log" >&2 || true
  exit 1
fi

echo "[READY] Driver and teleop are running. Starting interactive L20 sidecar recorder."
echo "        Ctrl+C stops all three processes."
"$runtime_dir/run_l20_sidecar_collect.sh" "$@"
