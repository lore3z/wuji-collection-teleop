#!/usr/bin/env bash
# Backward-compatible name.  The primary launcher has the same Myo pipeline.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_wuji_ftp1_collect.sh" "$@"
