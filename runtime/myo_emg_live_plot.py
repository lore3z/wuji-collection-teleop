#!/usr/bin/env python3
"""Plot all eight Myo RAW EMG channels in a live scrolling window."""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from wuji_myo_emg_source import MyoEmgSource


def prefer_matching_mpl_toolkits() -> None:
    """Keep Debian's system mpl_toolkits from mixing with venv Matplotlib."""
    try:
        import mpl_toolkits
    except ImportError:
        return
    for entry in sys.path:
        if not entry:
            continue
        package_root = Path(entry)
        matplotlib_dir = package_root / "matplotlib"
        toolkits_dir = package_root / "mpl_toolkits"
        if matplotlib_dir.is_dir() and (toolkits_dir / "mplot3d").is_dir():
            # The first Matplotlib package on sys.path is the one Python will
            # import. Point its namespace package at the matching toolkit.
            mpl_toolkits.__path__[:] = [str(toolkits_dir)]
            return


def display_arrays(
    timestamps_ns: deque[int],
    samples: deque[np.ndarray],
    gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relative times and samples, breaking lines across real outages."""
    if not timestamps_ns:
        return np.empty(0, dtype=np.float64), np.empty((0, 8), dtype=np.float32)
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    values = np.asarray(samples, dtype=np.float32)
    times = (timestamps - timestamps[-1]).astype(np.float64) / 1e9
    if len(timestamps) > 1:
        gaps = np.diff(timestamps) > int(gap_seconds * 1e9)
        values[1:][gaps] = np.nan
    return times, values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tty", default=os.getenv("WUJI_MYO_TTY", "/dev/ttyACM0"))
    parser.add_argument("--mac", default=os.getenv("WUJI_MYO_MAC", "auto"))
    parser.add_argument("--python", default=sys.executable, help="Python used by the Myo bridge")
    parser.add_argument("--window-seconds", type=float, default=5.0,
                        help="visible history in seconds (default: 5)")
    parser.add_argument("--display-fps", type=float, default=30.0,
                        help="plot refresh rate (default: 30)")
    parser.add_argument("--gap-ms", type=float, default=30.0,
                        help="break plotted lines across source gaps (default: 30)")
    args = parser.parse_args()
    if min(args.window_seconds, args.display_fps, args.gap_ms) <= 0:
        parser.error("--window-seconds, --display-fps and --gap-ms must be positive")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/wuji_matplotlib_cache")
    try:
        prefer_matching_mpl_toolkits()
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as exc:
        raise SystemExit("缺少 matplotlib；请先运行 ./setup.sh 安装项目依赖。") from exc

    source = MyoEmgSource(
        args.python,
        args.tty,
        args.mac,
        max_queue=max(1024, int(args.window_seconds * 400)),
    )
    timestamps_ns: deque[int] = deque()
    samples: deque[np.ndarray] = deque()

    figure, axes = plt.subplots(8, 1, sharex=True, figsize=(13, 9))
    figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.07, hspace=0.10)
    figure.canvas.manager.set_window_title("Myo RAW EMG live")
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, 8))
    lines = []
    for channel, (axis, color) in enumerate(zip(axes, colors), start=1):
        line, = axis.plot([], [], color=color, linewidth=0.9)
        lines.append(line)
        axis.set_ylim(-128, 127)
        axis.set_xlim(-args.window_seconds, 0)
        axis.set_ylabel(f"CH{channel}", rotation=0, labelpad=22)
        axis.grid(alpha=0.25)
        axis.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    axes[-1].set_xlabel("seconds before latest sample")

    source.start(wait_for_stability=False)
    print(
        f"[LIVE] Myo RAW EMG: MAC={args.mac}, tty={args.tty} "
        "(关闭窗口或按 Ctrl-C 停止)",
        flush=True,
    )

    def update(_frame: int):
        for timestamp_ns, _seq, _movement, emg, _arrival_ns in source.drain():
            timestamps_ns.append(timestamp_ns)
            samples.append(emg.copy())

        if timestamps_ns:
            cutoff_ns = timestamps_ns[-1] - int(args.window_seconds * 1e9)
            while timestamps_ns and timestamps_ns[0] < cutoff_ns:
                timestamps_ns.popleft()
                samples.popleft()
            times, values = display_arrays(
                timestamps_ns,
                samples,
                args.gap_ms / 1000.0,
            )
            for channel, line in enumerate(lines):
                line.set_data(times, values[:, channel])

        healthy = source.healthy()
        state_color = "darkgreen" if healthy else "firebrick"
        figure.suptitle(
            "Myo 8-channel RAW EMG  |  " + source.status(),
            color=state_color,
            fontsize=10,
        )
        return lines

    animation = FuncAnimation(
        figure,
        update,
        interval=1000.0 / args.display_fps,
        blit=False,
        cache_frame_data=False,
    )
    # Keep a named reference until plt.show() returns.
    _ = animation
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n[LIVE] stopped", flush=True)
    finally:
        source.close()
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
