#!/usr/bin/env bash
set -euo pipefail
# Merely executing this file does not grant hardware authorization.
EMG_ARM_PRESENT=0
for EMG_ARG in "$@"; do
    if [[ "$EMG_ARG" == "--arm" ]]; then EMG_ARM_PRESENT=1; fi
done
if [[ "$EMG_ARM_PRESENT" != 1 ]]; then
    echo "Hardware disabled: explicitly add --arm to authorize this invocation." >&2
    exit 2
fi
EMG_TELEOP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EMG_GEORT_PYTHON="${EMG_GEORT_PYTHON:-$EMG_TELEOP_ROOT/.venv-teleop/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
set +u
source /opt/ros/humble/setup.bash
source "$EMG_TELEOP_ROOT/runtime/common.sh"
source "$WUJI_L20_DRIVER_WS/install/setup.bash"
set -u
exec "$EMG_GEORT_PYTHON" -B -u "$EMG_TELEOP_ROOT/src/skeleton_teleop_MODE_A_emg.py" --hardware "$@"
