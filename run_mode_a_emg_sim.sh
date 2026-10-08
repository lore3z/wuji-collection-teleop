#!/usr/bin/env bash
set -euo pipefail
EMG_TELEOP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EMG_GEORT_PYTHON="${EMG_GEORT_PYTHON:-$EMG_TELEOP_ROOT/.venv-teleop/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
exec "$EMG_GEORT_PYTHON" -B -u "$EMG_TELEOP_ROOT/skeleton_teleop_MODE_A_emg.py" --sim --viewer "$@"
