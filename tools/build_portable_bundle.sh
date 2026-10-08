#!/usr/bin/env bash
# Export the exact committed source tree, excluding all ignored local data.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="$(realpath -m "${1:-$project_dir/release/wuji-collection-teleop}")"
if [[ -e "$output" ]]; then
  echo "[FAILED] bundle target already exists: $output" >&2
  exit 2
fi
if [[ -n "$(git -C "$project_dir" status --porcelain)" ]]; then
  echo "[FAILED] commit source changes before exporting a reproducible bundle" >&2
  exit 2
fi
mkdir -p "$output"
git -C "$project_dir" archive HEAD | tar -x -C "$output"
echo "[DONE] source bundle: $output"
