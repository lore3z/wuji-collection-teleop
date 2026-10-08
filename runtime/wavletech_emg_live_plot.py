#!/usr/bin/env python3
"""Plot all eight Wavletech EMG channels in a live scrolling window."""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
for _project_path in (_project_root, _project_root / "src"):
    if str(_project_path) not in _project_sys.path:
        _project_sys.path.insert(0, str(_project_path))


import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from wuji_serial_emg_source import SerialEmgSource


def prefer_matching_mpl_toolkits() -> None:
    """Keep system mpl_toolkits from mixing with the venv Matplotlib."""
    try:
        import mpl_toolkits
    except ImportError:
        return
    for entry in sys.path:
        if not entry:
            continue
        package_root = Path(entry)
        if (package_root / "matplotlib").is_dir() and (
            package_root / "mpl_toolkits" / "mplot3d"
        ).is_dir():
            mpl_toolkits.__path__[:] = [str(package_root / "mpl_toolkits")]
            return


def display_arrays(
    timestamps_ns: deque[int],
    samples: deque[np.ndarray],
    gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relative times and samples, breaking curves across outages."""
    if not timestamps_ns:
        return np.empty(0, dtype=np.float64), np.empty((0, 8), dtype=np.float32)
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    values = np.asarray(samples, dtype=np.float32)
    times = (timestamps - timestamps[-1]).astype(np.float64) / 1e9
    if len(timestamps) > 1:
        values[1:][np.diff(timestamps) > int(gap_seconds * 1e9)] = np.nan
    return times, values


def robust_limit(values: np.ndarray, minimum_uv: float) -> float:
    """Choose a stable symmetric display range without hiding large activity."""
    finite = np.abs(values[np.isfinite(values)])
    if not len(finite):
        return float(minimum_uv)
    return max(float(minimum_uv), float(np.percentile(finite, 99.5)) * 1.20)


def center_channels(values: np.ndarray) -> np.ndarray:
    """Remove each channel's remaining constant DC offset."""
    if not values.size:
        return values.copy()
    with np.errstate(invalid="ignore"):
        baselines = np.nanmedian(values, axis=0)
    baselines = np.where(np.isfinite(baselines), baselines, 0.0)
    return values - baselines


def remove_running_baseline(
    values: np.ndarray,
    timestamps_ns: deque[int],
    baseline_ms: float,
) -> np.ndarray:
    """Remove slow baseline wander for DISPLAY ONLY.

    A centered moving mean is estimated independently for each channel.
    NaNs inserted for communication gaps are preserved.  This function acts
    only on the plotting copy; raw serial/collector/model data are untouched.
    """
    if not values.size:
        return values.copy()

    timestamps = np.asarray(timestamps_ns, dtype=np.int64)

    # Estimate the actual EMG sample rate from timestamps.  Ignore large gaps
    # so reconnects/outages do not corrupt the estimate.
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        dt = dt[(dt > 0) & (dt < 10_000_000)]  # ignore gaps >= 10 ms
    else:
        dt = np.empty(0, dtype=np.int64)

    if len(dt):
        sample_rate_hz = 1e9 / float(np.median(dt))
    else:
        sample_rate_hz = 2000.0

    window = max(3, int(round(sample_rate_hz * baseline_ms / 1000.0)))
    if window % 2 == 0:
        window += 1

    n = values.shape[0]
    half = window // 2
    indices = np.arange(n)
    left = np.maximum(0, indices - half)
    right = np.minimum(n, indices + half + 1)

    result = np.full(values.shape, np.nan, dtype=np.float64)

    for channel in range(values.shape[1]):
        x = np.asarray(values[:, channel], dtype=np.float64)
        finite = np.isfinite(x)

        if not np.any(finite):
            continue

        # O(N) moving average that ignores NaNs.
        filled = np.where(finite, x, 0.0)
        counts = finite.astype(np.int64)

        cumulative = np.concatenate(([0.0], np.cumsum(filled)))
        cumulative_count = np.concatenate(([0], np.cumsum(counts)))

        sums = cumulative[right] - cumulative[left]
        nums = cumulative_count[right] - cumulative_count[left]

        baseline = np.divide(
            sums,
            nums,
            out=np.zeros_like(sums, dtype=np.float64),
            where=nums > 0,
        )

        result[finite, channel] = x[finite] - baseline[finite]

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tty", default=os.getenv("WUJI_WAVLETECH_TTY", "/dev/ttyUSB0"))
    parser.add_argument("--baud", type=int, default=int(os.getenv("WUJI_WAVLETECH_BAUD", "921600")))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--display-fps", type=float, default=30.0)
    parser.add_argument("--gap-ms", type=float, default=100.0)
    parser.add_argument("--min-range-uv", type=float, default=100.0,
                        help="minimum symmetric y-axis range for each channel")
    parser.add_argument(
        "--display-baseline-ms",
        type=float,
        default=250.0,
        help="DISPLAY-ONLY moving baseline window in ms (default: 250)",
    )
    args = parser.parse_args()
    if min(
        args.baud,
        args.window_seconds,
        args.display_fps,
        args.gap_ms,
        args.min_range_uv,
        args.display_baseline_ms,
    ) <= 0:
        parser.error("baud and display parameters must be positive")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/wuji_matplotlib_cache")
    try:
        prefer_matching_mpl_toolkits()
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as exc:
        raise SystemExit("缺少 matplotlib；请先运行 ./scripts/setup.sh 安装项目依赖。") from exc

    source = SerialEmgSource(
        args.python,
        args.tty,
        baud=args.baud,
        max_queue=max(4096, int(args.window_seconds * 4000)),
    )
    timestamps_ns: deque[int] = deque()
    samples: deque[np.ndarray] = deque()
    figure, axes = plt.subplots(8, 1, sharex=True, figsize=(13, 9))
    figure.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.07, hspace=0.10)
    figure.canvas.manager.set_window_title("Wavletech 8-channel EMG live")
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, 8))
    lines = []
    for channel, (axis, color) in enumerate(zip(axes, colors), start=1):
        line, = axis.plot([], [], color=color, linewidth=0.9)
        lines.append(line)
        axis.set_xlim(-args.window_seconds, 0)
        axis.set_ylim(-args.min_range_uv, args.min_range_uv)
        axis.set_ylabel(f"CH{channel}", rotation=0, labelpad=24)
        axis.grid(alpha=0.25)
        axis.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    axes[-1].set_xlabel("seconds before latest sample")

    source.start(wait_for_stability=False)
    print(
        f"[LIVE] Wavletech 8-channel EMG: {args.tty} @ {args.baud} baud. "
        "串口为独占资源，请勿与采集器同时运行。关闭窗口或按 Ctrl-C 停止。",
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

            # DISPLAY PATH ONLY:
            # raw samples remain completely untouched.
            display_values = remove_running_baseline(
                values,
                timestamps_ns,
                args.display_baseline_ms,
            )
            # Remove any tiny residual offset so every visible channel is
            # centered exactly around zero.
            display_values = center_channels(display_values)

            for channel, (axis, line) in enumerate(zip(axes, lines)):
                line.set_data(times, display_values[:, channel])
                limit = robust_limit(display_values[:, channel], args.min_range_uv)
                axis.set_ylim(-limit, limit)
        figure.suptitle(
            f"Wavletech 8-channel EMG (display baseline removed, {args.display_baseline_ms:.0f} ms)  |  "
            + source.status(),
            color="darkgreen" if source.healthy() else "firebrick",
            fontsize=10,
        )
        return lines

    animation = FuncAnimation(
        figure, update, interval=1000.0 / args.display_fps,
        blit=False, cache_frame_data=False,
    )
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
