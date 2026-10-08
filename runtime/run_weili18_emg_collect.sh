#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$runtime_dir/.." && pwd)"
python="${WEILI18_PYTHON:-$project_dir/.venv/bin/python}"
tty="${WEILI18_TTY:-/dev/ttyUSB0}"

[[ -x "$python" ]] || { echo "[FAILED] Python is unavailable: $python" >&2; exit 2; }
exec "$python" -u "$runtime_dir/collect_weili18_emg.py" --tty "$tty" "$@"
