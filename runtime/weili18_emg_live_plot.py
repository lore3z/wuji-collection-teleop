#!/usr/bin/env python3
"""Live 18-channel EMG and 6-axis IMU view for the 唯理 WAVELETECH-18 band."""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
from pathlib import Path
import sys

import numpy as np

RUNTIME = Path(__file__).resolve().parent
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from weili18_emg import EMG_CHANNELS, EMG_RATE_HZ, EmgSample, ImuSample, Weili18EmgDevice, protocol_for


def prefer_matching_mpl_toolkits() -> None:
    """Avoid mixing the system mpl_toolkits with a virtualenv Matplotlib."""
    try:
        import mpl_toolkits
    except ImportError:
        return
    for entry in sys.path:
        if not entry:
            continue
        package_root = Path(entry)
        if (package_root / "matplotlib").is_dir() and (package_root / "mpl_toolkits" / "mplot3d").is_dir():
            mpl_toolkits.__path__[:] = [str(package_root / "mpl_toolkits")]
            return


def downsample_minmax(times: np.ndarray, values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce dense EMG to a min/max envelope while preserving brief peaks."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= max_points or max_points < 4:
        return times, values
    factor = int(math.ceil(len(values) / max(1, max_points // 2)))
    complete = len(values) // factor
    end = complete * factor
    if not complete:
        return times, values
    blocks = values[:end].reshape(complete, factor)
    finite = np.isfinite(blocks)
    safe = np.where(finite, blocks, 0.0)
    min_indices = np.argmin(np.where(finite, safe, np.inf), axis=1)
    max_indices = np.argmax(np.where(finite, safe, -np.inf), axis=1)
    has_values = np.any(finite, axis=1)
    offsets = np.arange(complete, dtype=np.int64) * factor
    indices = np.stack((offsets + min_indices, offsets + max_indices), axis=1).reshape(-1)
    indices.sort()
    selected = values[indices].copy()
    selected_blocks = np.repeat(has_values, 2)
    selected[~selected_blocks] = np.nan
    if end < len(values):
        indices = np.concatenate((indices, np.arange(end, len(values), dtype=np.int64)))
        selected = np.concatenate((selected, values[end:]))
    return times[indices], selected


def robust_limit(values: np.ndarray, minimum_uv: float) -> float:
    finite = np.abs(np.asarray(values)[np.isfinite(values)])
    if not len(finite):
        return float(minimum_uv)
    return max(float(minimum_uv), float(np.percentile(finite, 99.5)) * 1.15)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tty", default=os.environ.get("WEILI18_TTY", "/dev/ttyUSB0"))
    parser.add_argument("--packet-format", choices=("filtered", "raw"), default="filtered",
                        help="filtered=41-byte / 16-bit default; raw=59-byte / 24-bit")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--display-fps", type=float, default=20.0)
    parser.add_argument("--min-range-uv", type=float, default=50.0)
    parser.add_argument("--queue-capacity", type=int, default=50_000)
    parser.add_argument("--silence-timeout-s", type=float, default=0.5)
    args = parser.parse_args(argv)
    if min(args.window_seconds, args.display_fps, args.min_range_uv, args.silence_timeout_s) <= 0:
        parser.error("window, display FPS, voltage range and silence timeout must be positive")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/wuji_matplotlib_cache")
    try:
        prefer_matching_mpl_toolkits()
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.gridspec import GridSpec
    except ImportError as exc:
        raise SystemExit("缺少 matplotlib；请先安装 requirements-collector.txt。") from exc

    spec = protocol_for(args.packet_format)
    device = Weili18EmgDevice(
        args.tty,
        packet_format=spec,
        queue_capacity=args.queue_capacity,
        silence_timeout_s=args.silence_timeout_s,
        read_chunk_bytes=512,
    )
    emg_capacity = max(EMG_RATE_HZ, int(args.window_seconds * EMG_RATE_HZ * 1.25))
    emg_times: deque[float] = deque(maxlen=emg_capacity)
    emg_values: deque[np.ndarray] = deque(maxlen=emg_capacity)
    imu_times: deque[float] = deque(maxlen=max(500, int(args.window_seconds * 400)))
    imu_values: deque[np.ndarray] = deque(maxlen=max(500, int(args.window_seconds * 400)))
    rate_events: deque[tuple[int, str]] = deque(maxlen=20_000)
    latest_imu: ImuSample | None = None
    last_emg_index: int | None = None

    figure = plt.figure(figsize=(17, 12))
    grid = GridSpec(6, 4, figure=figure, width_ratios=(1.0, 1.0, 1.0, 1.5),
                    left=0.055, right=0.985, top=0.91, bottom=0.08, hspace=0.42, wspace=0.32)
    figure.canvas.manager.set_window_title("唯理 WAVELETECH-18 EMG + IMU")
    emg_axes = []
    emg_lines = []
    colors = plt.cm.turbo(np.linspace(0.02, 0.98, EMG_CHANNELS))
    for channel in range(EMG_CHANNELS):
        row, col = divmod(channel, 3)
        axis = figure.add_subplot(grid[row, col])
        line, = axis.plot([], [], color=colors[channel], linewidth=0.8)
        axis.set_xlim(-args.window_seconds, 0.0)
        axis.set_ylim(-args.min_range_uv, args.min_range_uv)
        axis.set_ylabel(f"CH{channel + 1} (μV)", rotation=0, labelpad=28)
        axis.grid(alpha=0.22)
        axis.axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
        if row == 5:
            axis.set_xlabel("relative seconds (nominal 2000 Hz sample index)")
        emg_axes.append(axis)
        emg_lines.append(line)

    gyro_axis = figure.add_subplot(grid[0:3, 3])
    accel_axis = figure.add_subplot(grid[3:6, 3], sharex=gyro_axis)
    gyro_lines = [gyro_axis.plot([], [], linewidth=0.9, label=f"gyro {axis}")[0] for axis in "XYZ"]
    accel_lines = [accel_axis.plot([], [], linewidth=0.9, label=f"accel {axis}")[0] for axis in "XYZ"]
    for axis, ylabel in ((gyro_axis, "rad/s"), (accel_axis, "m/s²")):
        axis.set_xlim(-args.window_seconds, 0.0)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=1, fontsize=8)
    gyro_axis.set_title("WAVELETECH-18 IMU device axes", fontsize=10)
    accel_axis.set_xlabel("seconds before latest IMU packet")
    info_text = figure.text(0.06, 0.025, "IMU: waiting for BB packet", fontsize=9, family="monospace")

    try:
        device.start()
    except Exception as exc:
        raise SystemExit(f"[FAILED] 串口打开失败: {exc}") from exc
    print(
        f"[LIVE] 唯理 WAVELETECH-18: {args.tty} @ 2,000,000 baud, {spec.name}. "
        "串口已独占；关闭窗口或 Ctrl-C 退出。",
        flush=True,
    )

    def update(_frame):
        nonlocal latest_imu, last_emg_index
        events = device.drain_events(max_items=50_000)
        for event in events:
            rate_events.append((event.host_monotonic_ns, event.kind))
            if isinstance(event, EmgSample):
                sample_time = event.sample_index / EMG_RATE_HZ
                if event.sequence_gap_before and last_emg_index is not None:
                    break_time = (last_emg_index + 0.5) / EMG_RATE_HZ
                    emg_times.append(break_time)
                    emg_values.append(np.full(EMG_CHANNELS, np.nan, dtype=np.float32))
                emg_times.append(sample_time)
                emg_values.append(np.asarray(event.voltage_uv, dtype=np.float32))
                last_emg_index = event.sample_index
            else:
                latest_imu = event
                imu_times.append(event.host_monotonic_ns / 1e9)
                imu_values.append(np.asarray((*event.gyro_rad_s, *event.accel_m_s2), dtype=np.float32))

        if emg_times:
            newest = emg_times[-1]
            while emg_times and emg_times[0] < newest - args.window_seconds:
                emg_times.popleft()
                emg_values.popleft()
            times = np.asarray(emg_times, dtype=np.float64)
            raw = np.asarray(emg_values, dtype=np.float32)
            with np.errstate(invalid="ignore"):
                baseline = np.nanmedian(raw, axis=0)
            baseline = np.where(np.isfinite(baseline), baseline, 0.0)
            display = raw - baseline
            relative = times - newest
            for index, (axis, line) in enumerate(zip(emg_axes, emg_lines)):
                x, y = downsample_minmax(relative, display[:, index], max_points=1600)
                line.set_data(x, y)
                limit = robust_limit(y, args.min_range_uv)
                axis.set_ylim(-limit, limit)

        if imu_times:
            latest_imu_time = imu_times[-1]
            while imu_times and imu_times[0] < latest_imu_time - args.window_seconds:
                imu_times.popleft()
                imu_values.popleft()
            times = np.asarray(imu_times, dtype=np.float64) - latest_imu_time
            values = np.asarray(imu_values, dtype=np.float32)
            for index, line in enumerate(gyro_lines):
                line.set_data(times, values[:, index])
            for index, line in enumerate(accel_lines):
                line.set_data(times, values[:, index + 3])
            gyro_axis.relim()
            gyro_axis.autoscale_view(scalex=False, scaley=True)
            accel_axis.relim()
            accel_axis.autoscale_view(scalex=False, scaley=True)

        stats = device.stats
        now_ns = time.monotonic_ns()
        while rate_events and now_ns - rate_events[0][0] > 2_000_000_000:
            rate_events.popleft()
        rate_hz = {kind: 0.0 for kind in ("emg", "imu")}
        if len(rate_events) > 1:
            span = (rate_events[-1][0] - rate_events[0][0]) / 1e9
            if span > 0:
                for kind in rate_hz:
                    count = sum(1 for _, event_kind in rate_events if event_kind == kind)
                    rate_hz[kind] = count / span
        imu_summary = "IMU: waiting for BB packet"
        if latest_imu is not None:
            imu_summary = (
                f"IMU t={latest_imu.device_time_ms} ms | T={latest_imu.temperature_c:.1f}°C "
                f"battery={latest_imu.battery_voltage_v:.3f} V/{latest_imu.battery_percent}% | "
                f"gyro={np.asarray(latest_imu.gyro_rad_s).round(3).tolist()} rad/s | "
                f"accel={np.asarray(latest_imu.accel_m_s2).round(3).tolist()} m/s²"
            )
        info_text.set_text(
            f"{imu_summary}\n"
            f"shared SN missing={stats['missing_packets']} duplicate={stats['duplicate_packets']} "
            f"out_of_order={stats['out_of_order_packets']} | queue={stats['queue_depth']} "
            f"discarded_bytes={stats['discarded_bytes']}"
        )
        state = stats["state"]
        error = stats["error"]
        figure.suptitle(
            f"唯理 WAVELETECH-18 | {spec.name} ({spec.packet_size} bytes) | "
            f"EMG {rate_hz['emg']:.0f}/{EMG_RATE_HZ} Hz  IMU {rate_hz['imu']:.0f}/208 Hz | "
            f"{state}" + (f" | {error}" if error else ""),
            color="firebrick" if error else ("darkgreen" if state == "running" else "darkorange"),
            fontsize=12,
        )
        return (*emg_lines, *gyro_lines, *accel_lines)

    animation = FuncAnimation(
        figure, update, interval=1000.0 / args.display_fps, blit=False, cache_frame_data=False,
    )
    _ = animation
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n[LIVE] stopped", flush=True)
    finally:
        device.close()
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
