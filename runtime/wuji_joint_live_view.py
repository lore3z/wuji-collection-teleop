#!/usr/bin/env python3
"""Live Wuji skeleton and joint-angle diagnostic window."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from wuji_glove_d435_collect import DEFAULT_GLOVE_SN, WujiGloveSource
from wuji_glove_d435_ftp1_collect import WUJI_21_FROM_25, WUJI_21_NAMES
from wuji_ftp1_hand_geometry import FTP1_HAND_NAMES, ftp1_right_hand_joints_from_mediapipe


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
FLEXION_NAMES = (
    "thumb_mcp", "thumb_ip", "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip", "ring_mcp", "ring_pip",
    "ring_dip", "pinky_mcp", "pinky_pip", "pinky_dip",
)
FLEXION_IDX = np.asarray([FTP1_HAND_NAMES.index(name) for name in FLEXION_NAMES])


def _draw_projection(axis, points: np.ndarray, x: int, y: int, title: str) -> None:
    axis.clear()
    for start, end in HAND_EDGES:
        axis.plot(points[[start, end], x], points[[start, end], y], color="tab:blue", linewidth=2)
    axis.scatter(points[:, x], points[:, y], c=np.arange(21), cmap="viridis", s=25)
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-sn", default=os.getenv("WUJI_GLOVE_SN", DEFAULT_GLOVE_SN))
    parser.add_argument("--window-seconds", type=float, default=2.0,
                        help="history used for frozen-channel detection")
    parser.add_argument("--freeze-range-deg", type=float, default=0.25,
                        help="mark a channel red when its rolling range is below this")
    args = parser.parse_args()
    if args.window_seconds <= 0 or args.freeze_range_deg <= 0:
        parser.error("window and freeze threshold must be positive")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/wuji_matplotlib_cache")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    source = WujiGloveSource(args.glove_sn, 120.0, capture_skeleton=True)
    source.start()
    history: deque[tuple[float, np.ndarray]] = deque()
    latest_angles = None
    latest_skeleton = None

    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 1.55))
    skeleton_xy_axis = figure.add_subplot(grid[0, 0])
    skeleton_xz_axis = figure.add_subplot(grid[1, 0])
    raw_axis = figure.add_subplot(grid[0, 1])
    ftp_axis = figure.add_subplot(grid[1, 1])
    figure.canvas.manager.set_window_title("Wuji joint live diagnostic")

    def update(_frame):
        nonlocal latest_angles, latest_skeleton
        angles, _tactile = source.drain()
        skeletons = source.drain_skeleton()
        if angles:
            latest_angles = angles[-1][2].reshape(-1)[WUJI_21_FROM_25]
        if skeletons:
            latest_skeleton = skeletons[-1][2]
        now = time.monotonic()
        if latest_skeleton is not None:
            canonical = ftp1_right_hand_joints_from_mediapipe(latest_skeleton[None])[0]
            history.append((now, canonical.copy()))
        while history and history[0][0] < now - args.window_seconds:
            history.popleft()

        if latest_skeleton is not None:
            points = latest_skeleton - latest_skeleton[0]
            _draw_projection(skeleton_xy_axis, points, 0, 1, "SDK skeleton XY")
            _draw_projection(skeleton_xz_axis, points, 0, 2, "SDK skeleton XZ")
        else:
            for axis, title in ((skeleton_xy_axis, "SDK skeleton XY"),
                                (skeleton_xz_axis, "SDK skeleton XZ")):
                axis.clear()
                axis.set_title(title)
                axis.text(0.18, 0.5, "Waiting for glove...", transform=axis.transAxes)

        raw_axis.clear()
        raw_axis.set_title("SDK anatomical 21 DoF (deg)")
        raw_axis.set_ylabel("degrees")
        if latest_angles is not None:
            values = np.rad2deg(latest_angles)
            raw_axis.bar(np.arange(21), values, color="tab:orange")
            raw_axis.set_xticks(np.arange(21), WUJI_21_NAMES, rotation=60, ha="right", fontsize=7)
        raw_axis.grid(axis="y", alpha=0.25)

        ftp_axis.clear()
        ftp_axis.set_title("FTP-1 flexion from SDK skeleton (red = frozen over rolling window)")
        ftp_axis.set_ylabel("degrees")
        if history:
            values = np.rad2deg(history[-1][1][FLEXION_IDX])
            enough = history[-1][0] - history[0][0] >= args.window_seconds * 0.8
            ranges = np.ptp(np.stack([item[1][FLEXION_IDX] for item in history]), axis=0)
            frozen = enough & (np.rad2deg(ranges) < args.freeze_range_deg)
            colors = np.where(frozen, "tab:red", "tab:green")
            ftp_axis.bar(np.arange(len(values)), values, color=colors)
            ftp_axis.set_xticks(np.arange(len(values)), FLEXION_NAMES, rotation=45, ha="right", fontsize=8)
            frozen_names = [name for name, flag in zip(FLEXION_NAMES, frozen) if flag]
            status = "Frozen: " + (", ".join(frozen_names) if frozen_names else "none")
            ftp_axis.text(0.01, 0.95, status, transform=ftp_axis.transAxes, va="top",
                          color="crimson" if frozen_names else "green", fontsize=9)
        ftp_axis.grid(axis="y", alpha=0.25)
        figure.suptitle(f"Glove {args.glove_sn} | {source.status()}", fontsize=10)
        return ()

    animation = FuncAnimation(figure, update, interval=50, cache_frame_data=False)
    try:
        plt.show()
    finally:
        # Keep a reference until the GUI closes, then release SDK subscriptions.
        _ = animation
        source.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
