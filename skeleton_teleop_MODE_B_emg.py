#!/usr/bin/env python3
"""EMG input for the immutable GOOD natural/independent-thumb mode."""
import sys
sys.dont_write_bytecode = True
from emg_teleop_launcher import main

if __name__ == "__main__":
    raise SystemExit(main("B"))
