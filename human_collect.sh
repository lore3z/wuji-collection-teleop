#!/usr/bin/env -S -u LD_LIBRARY_PATH bash
# Explicit alias for the pure-human FTP-1 collection stack.
# Human collection replaces the RealSense with the PICO headset's live
# passthrough left-eye stream. Gemini remains the third-person stream.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Human-demo metadata is selected for every episode after the operator presses
# s.  The collector then switches its output directory before recording starts.
export WUJI_EPISODE_LAYOUT_SELECTION=1
export WUJI_DATA_DIR_OVERRIDE="$project_dir/data"
# Keep routine SDK connection/status messages out of the collection console.
# Override when diagnosing a glove issue, e.g. WUJI_SDK_LOG_LEVEL=info ./human_collect.sh.
export WUJI_SDK_LOG_LEVEL="${WUJI_SDK_LOG_LEVEL:-error}"
export WUJI_SKIP_REALSENSE=1
# Human demonstrations do not need video sidecars.  Fast-save also avoids
# running post-hoc quality analysis after `e`; runtime monitors catch faults
# while recording, before the operator ends the episode.
exec "$project_dir/collect.sh" --no-save-mp4 --fast-save "$@"
