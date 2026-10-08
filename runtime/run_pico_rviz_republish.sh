#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$runtime_dir/common.sh"
unset LD_LIBRARY_PATH || true
wuji_source_ros

compressed_topic="${WUJI_VR_EGO_TOPIC}"
raw_topic="${WUJI_VR_EGO_RVIZ_TOPIC:-/pico/ego/image_raw}"

exec "$WUJI_PYTHON" "$runtime_dir/pico_rviz_republisher.py" \
  --input "$compressed_topic" \
  --output "$raw_topic"
