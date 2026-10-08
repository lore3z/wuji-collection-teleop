#!/usr/bin/env bash

# Shared path/configuration helpers. This file is sourced by runtime scripts.
WUJI_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -a
# shellcheck disable=SC1091
source "$WUJI_PROJECT_DIR/config/collector.env"
if [[ -f "$WUJI_PROJECT_DIR/config/local.env" ]]; then
  # shellcheck disable=SC1091
  source "$WUJI_PROJECT_DIR/config/local.env"
fi
set +a

wuji_abs_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$WUJI_PROJECT_DIR" "$value"
  fi
}

WUJI_DATA_DIR="$(wuji_abs_path "$WUJI_DATA_DIR")"
if [[ -n "${WUJI_DATA_DIR_OVERRIDE:-}" ]]; then
  WUJI_DATA_DIR="$WUJI_DATA_DIR_OVERRIDE"
fi
WUJI_RUNTIME_DIR="$(wuji_abs_path "$WUJI_RUNTIME_DIR")"
WUJI_PYTHON="$(wuji_abs_path "$WUJI_PYTHON")"
WUJI_GEMINI_INSTALL="$WUJI_RUNTIME_DIR/ros_ws/install"
WUJI_GEMINI_ORBBEC_INSTALL="$(wuji_abs_path "$WUJI_GEMINI_ORBBEC_INSTALL")"
WUJI_L20_DRIVER_WS="$(wuji_abs_path "$WUJI_L20_DRIVER_WS")"
WUJI_L20_DATA_DIR="$(wuji_abs_path "$WUJI_L20_DATA_DIR")"
if [[ -n "$WUJI_TACTILE_HEALTH_MASK" ]]; then
  WUJI_TACTILE_HEALTH_MASK="$(wuji_abs_path "$WUJI_TACTILE_HEALTH_MASK")"
fi
if [[ -n "${WUJI_PICO_SETUP_BASH:-}" ]]; then
  WUJI_PICO_SETUP_BASH="$(wuji_abs_path "$WUJI_PICO_SETUP_BASH")"
fi

# Prefer the stable udev by-id alias when a numeric serial path changes. Ignore
# the old Bluegiga dongle and auto-select only when one other USB serial device
# exists; otherwise keep the configured path and fail explicitly at preflight.
wuji_resolve_wavletech_tty() {
  local configured="$1"
  local basename="${configured##*/}"
  local numeric_tty=false
  [[ "$basename" == ttyACM* || "$basename" == ttyUSB* ]] && numeric_tty=true
  local candidates=()
  local candidate
  shopt -s nullglob
  for candidate in /dev/serial/by-id/*; do
    [[ -e "$candidate" ]] || continue
    [[ "${candidate##*/}" == *Bluegiga* ]] && continue
    candidates+=("$candidate")
  done
  shopt -u nullglob
  if $numeric_tty && [[ "${#candidates[@]}" -eq 1 ]]; then
    printf '%s\n' "${candidates[0]}"
  elif [[ -e "$configured" ]]; then
    printf '%s\n' "$configured"
  elif [[ "${#candidates[@]}" -eq 1 ]]; then
    printf '%s\n' "${candidates[0]}"
  else
    printf '%s\n' "$configured"
  fi
}

wuji_resolve_myo_tty() {
  local configured="$1"
  local candidate
  shopt -s nullglob
  for candidate in /dev/serial/by-id/*Bluegiga*; do
    if [[ -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      shopt -u nullglob
      return
    fi
  done
  shopt -u nullglob
  printf '%s\n' "$configured"
}

WUJI_WAVLETECH_TTY_RESOLVED="$(wuji_resolve_wavletech_tty "$WUJI_WAVLETECH_TTY")"
WUJI_MYO_TTY_RESOLVED="$(wuji_resolve_myo_tty "$WUJI_MYO_TTY")"

wuji_source_ros() {
  local ros_setup="/opt/ros/${WUJI_ROS_DISTRO}/setup.bash"
  if [[ ! -r "$ros_setup" ]]; then
    echo "[FAILED] ROS setup 不存在：$ros_setup。先运行 ./scripts/setup.sh --check。" >&2
    return 1
  fi
  unset LD_LIBRARY_PATH || true
  set +u
  # shellcheck disable=SC1090
  source "$ros_setup"
  set -u
}

wuji_require_file() {
  if [[ ! -e "$1" ]]; then
    echo "[FAILED] 缺少：$1" >&2
    return 1
  fi
}

# Source modules and project-local namespace packages for child processes.
export PYTHONPATH="$WUJI_PROJECT_DIR/src:$WUJI_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
