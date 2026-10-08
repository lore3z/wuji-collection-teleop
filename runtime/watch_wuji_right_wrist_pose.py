#!/usr/bin/env python3
"""Print the live Wuji dynamic ``waist -> r_wrist`` pose.

Run this in a second terminal while the collector is running:
  ./.venv/bin/python runtime/watch_wuji_right_wrist_pose.py

The output order is ``[x, y, z, roll, pitch, yaw]`` in metres/radians.
Press Ctrl-C to stop.  ``--seconds`` is useful for a short connection check.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


# Make this standalone script work even when launched outside the project root.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from wuji_glove_d435_collect import DEFAULT_GLOVE_SN, _right_wrist_pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-sn", default=DEFAULT_GLOVE_SN)
    parser.add_argument("--rate", type=float, default=10.0, help="terminal refresh rate in Hz (default: 10)")
    parser.add_argument("--seconds", type=float, default=0.0, help="0=until Ctrl-C (default); otherwise stop after this duration")
    args = parser.parse_args()
    if args.rate <= 0 or args.seconds < 0:
        parser.error("--rate must be positive and --seconds must be non-negative")

    # Keep the monitor readable even when it has its own SDK session.
    os.environ.setdefault("WUJI_SDK_LOG_LEVEL", "error")
    try:
        from wuji_sdk import SdkManager
    except ImportError as exc:
        raise SystemExit("找不到 wuji_sdk；请使用项目 .venv 或可运行采集的 Python 环境。") from exc

    manager = SdkManager.instance()
    glove = None
    sub = None
    try:
        glove = manager.connect(sn=args.glove_sn, device_name="right_wrist_pose_watch")
        sub = manager.tf().subscribe()
        print(f"[LIVE] {args.glove_sn}: waist -> r_wrist  (Ctrl-C 停止)")
        print("       xyz: m; rpy: rad / deg")
        deadline = time.monotonic() + args.seconds if args.seconds else None
        next_print = 0.0
        while deadline is None or time.monotonic() < deadline:
            frame = sub.recv()
            if frame is None:
                time.sleep(0.001)
                continue
            result = _right_wrist_pose(frame)
            if result is None or time.monotonic() < next_print:
                continue
            timestamp_us, pose = result
            xyz, rpy = pose[:3], pose[3:]
            print(
                f"t={timestamp_us} us  "
                f"xyz=[{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}] m  "
                f"rpy=[{rpy[0]:+.4f}, {rpy[1]:+.4f}, {rpy[2]:+.4f}] rad  "
                f"[{np.degrees(rpy[0]):+.1f}, {np.degrees(rpy[1]):+.1f}, {np.degrees(rpy[2]):+.1f}] deg",
                flush=True,
            )
            next_print = time.monotonic() + 1.0 / args.rate
    except KeyboardInterrupt:
        print("\n[LIVE] stopped")
    finally:
        try:
            if sub is not None:
                sub.close()
        except Exception:
            pass
        try:
            if glove is not None:
                glove.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
