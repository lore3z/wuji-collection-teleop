#!/usr/bin/env -S -u LD_LIBRARY_PATH bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$project_dir/runtime/collect_all.sh" "$@"
