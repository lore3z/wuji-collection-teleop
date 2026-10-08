#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$root/run_emg_model_hand.sh" --emg-source wavletech "$@"
