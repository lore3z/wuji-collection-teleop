#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${EMG2POSE_ONLINE_PACKAGE:?Set EMG2POSE_ONLINE_PACKAGE to the installed online model project}"
export EMG2POSE_ONLINE_PACKAGE
export EMG2POSE_ONLINE_URL="${EMG2POSE_ONLINE_URL:-http://127.0.0.1:8765}"
exec "$root/human_collect.sh" "$@"
