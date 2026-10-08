#!/usr/bin/env python3
"""EMG input for the immutable GOOD pad-to-pad pinch mode."""

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))

import sys
sys.dont_write_bytecode = True
from emg_teleop_launcher import main

if __name__ == "__main__":
    raise SystemExit(main("A"))
