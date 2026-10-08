#!/usr/bin/env python3
"""Visualize the recorded PICO wrist Tracker XYZ trajectory from an FTP-1 Zarr."""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import zarr


def _is_zarr(path: Path) -> bool:
    return path.is_dir() and ((path / ".zgroup").is_file() or (path / "zarr.json").is_file())


def resolve_episode(path: Path, latest: bool) -> Path:
    path = path.expanduser().resolve()
    if _is_zarr(path):
        return path
    if not path.is_dir():
        raise ValueError(f"path does not exist: {path}")
    episodes = sorted(item for item in path.rglob("episode_*.zarr") if _is_zarr(item))
    if not episodes:
        raise ValueError(f"no episode_*.zarr found under {path}")
    if latest or len(episodes) == 1:
        return episodes[-1]
    raise ValueError(f"found {len(episodes)} episodes; pass --latest or specify one episode")


def load_wrist_trajectory(path: Path, source: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    root = zarr.open_group(str(path), mode="r")
    data = root["data"] if "data" in root else root
    streams = root["streams"] if "streams" in root else None
    raw_available = streams is not None and "right_wrist_tracker_pose_raw" in streams
    canonical_available = "right_wrist_tracker_pose" in data

    use_raw = source == "raw" or (source == "auto" and raw_available)
    if use_raw:
        if not raw_available:
            raise ValueError("episode has no streams/right_wrist_tracker_pose_raw")
        pose = np.asarray(streams["right_wrist_tracker_pose_raw"][:], dtype=np.float64)
        timestamps_ns = np.asarray(streams["right_wrist_tracker_pose_timestamp_ns"][:], dtype=np.int64)
        label = "native wrist Tracker"
    else:
        if not canonical_available:
            raise ValueError("episode has no data/right_wrist_tracker_pose")
        pose = np.asarray(data["right_wrist_tracker_pose"][:], dtype=np.float64)
        timestamps_ns = np.asarray(data["timestamps"][:], dtype=np.int64)
        label = "canonical aligned wrist Tracker"

    if pose.ndim != 2 or pose.shape[1] < 3 or len(pose) != len(timestamps_ns):
        raise ValueError(f"invalid wrist trajectory shape pose={pose.shape}, timestamps={timestamps_ns.shape}")
    if len(pose) < 2 or not np.all(np.isfinite(pose[:, :3])):
        raise ValueError("wrist trajectory needs at least two finite XYZ samples")
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("wrist trajectory timestamps are not strictly increasing")
    return pose[:, :3], timestamps_ns, label


def trajectory_quality(xyz: np.ndarray, timestamps_ns: np.ndarray, epsilon_m: float = 1e-6) -> dict[str, float | int]:
    xyz = np.asarray(xyz, dtype=np.float64)
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    dt_s = np.diff(timestamps_ns) / 1e9
    steps_m = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    duration_s = max((timestamps_ns[-1] - timestamps_ns[0]) / 1e9, 1e-9)
    changed = steps_m > epsilon_m
    return {
        "samples": len(xyz),
        "duration_s": duration_s,
        "source_hz": (len(xyz) - 1) / duration_s,
        "xyz_update_count": int(np.count_nonzero(changed)),
        "xyz_update_hz": float(np.count_nonzero(changed) / duration_s),
        "adjacent_same_ratio": float(np.mean(~changed)),
        "max_step_m": float(steps_m.max(initial=0.0)),
        "max_speed_m_s": float(np.max(steps_m / dt_s, initial=0.0)),
        "path_length_m": float(steps_m.sum()),
        "span_m": float(np.linalg.norm(np.ptp(xyz, axis=0))),
    }


def _equal_3d_axes(axis, xyz: np.ndarray) -> None:
    low = xyz.min(axis=0)
    high = xyz.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _plot_projection(axis, xyz: np.ndarray, first: int, second: int, labels: tuple[str, str], title: str) -> None:
    axis.plot(xyz[:, first], xyz[:, second], linewidth=1.2)
    axis.scatter(xyz[0, first], xyz[0, second], color="limegreen", label="start")
    axis.scatter(xyz[-1, first], xyz[-1, second], color="crimson", marker="X", label="end")
    axis.set(xlabel=f"{labels[0]} (m)", ylabel=f"{labels[1]} (m)", title=title)
    axis.axis("equal")
    axis.grid(alpha=0.25)


def plot_trajectory(
    xyz: np.ndarray,
    timestamps_ns: np.ndarray,
    *,
    title: str,
    output: Path,
    show: bool,
    jump_threshold_m: float,
) -> None:
    cache = Path("/tmp/wuji_matplotlib_cache")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
    import matplotlib.pyplot as plt

    seconds = (timestamps_ns - timestamps_ns[0]) / 1e9
    steps_m = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    speed = steps_m / (np.diff(timestamps_ns) / 1e9)
    jumps = np.flatnonzero(steps_m > jump_threshold_m) + 1

    figure = plt.figure(figsize=(13, 6.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    try:
        axis3d = figure.add_subplot(grid[:, 0], projection="3d")
    except ValueError:
        # Some ROS images expose a system mpl_toolkits beside the venv's newer
        # matplotlib. Keep the tool useful without mutating that environment.
        plt.close(figure)
        figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        _plot_projection(axes[0, 0], xyz, 0, 1, ("X", "Y"), "XY projection")
        _plot_projection(axes[0, 1], xyz, 0, 2, ("X", "Z"), "XZ projection")
        _plot_projection(axes[1, 0], xyz, 1, 2, ("Y", "Z"), "YZ projection")
        axis_speed = axes[1, 1]
        axis_speed.plot(seconds[1:], speed, linewidth=1.0)
        if len(jumps):
            axis_speed.scatter(seconds[jumps], speed[jumps - 1], color="red", s=30)
        axis_speed.set(xlabel="time (s)", ylabel="speed (m/s)", title=f"Step speed; red markers > {jump_threshold_m * 100:.1f} cm/sample")
        axis_speed.grid(alpha=0.25)
        figure.suptitle(title)
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
        print(f"[TRAJECTORY] wrote {output} (2D fallback: Matplotlib 3D unavailable)")
        if show:
            plt.show()
        plt.close(figure)
        return
    axis3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.65", linewidth=1.0)
    points = axis3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=seconds, cmap="viridis", s=8)
    axis3d.scatter(*xyz[0], color="limegreen", s=65, marker="o", label="start")
    axis3d.scatter(*xyz[-1], color="crimson", s=65, marker="X", label="end")
    if len(jumps):
        axis3d.scatter(xyz[jumps, 0], xyz[jumps, 1], xyz[jumps, 2], color="red", s=55, marker="^", label="large step")
    axis3d.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title=title)
    axis3d.legend(loc="best")
    _equal_3d_axes(axis3d, xyz)
    figure.colorbar(points, ax=axis3d, shrink=0.65, label="time (s)")

    axis_xy = figure.add_subplot(grid[0, 1])
    _plot_projection(axis_xy, xyz, 0, 1, ("X", "Y"), "XY projection")

    axis_speed = figure.add_subplot(grid[1, 1])
    axis_speed.plot(seconds[1:], speed, linewidth=1.0)
    if len(jumps):
        axis_speed.scatter(seconds[jumps], speed[jumps - 1], color="red", s=30)
    axis_speed.axhline(jump_threshold_m / np.median(np.diff(timestamps_ns) / 1e9), color="red", linestyle="--", alpha=0.45)
    axis_speed.set(xlabel="time (s)", ylabel="speed (m/s)", title=f"Step speed; red markers > {jump_threshold_m * 100:.1f} cm/sample")
    axis_speed.grid(alpha=0.25)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"[TRAJECTORY] wrote {output}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="episode Zarr or directory containing episodes")
    parser.add_argument("--latest", action="store_true", help="use the latest episode below PATH")
    parser.add_argument("--source", choices=("auto", "raw", "canonical"), default="auto")
    parser.add_argument("--output", type=Path, help="PNG path; defaults beside the episode")
    parser.add_argument("--show", action="store_true", help="also open an interactive matplotlib window")
    parser.add_argument("--jump-threshold-m", type=float, default=0.05)
    args = parser.parse_args()
    if args.jump_threshold_m <= 0:
        parser.error("--jump-threshold-m must be positive")
    try:
        episode = resolve_episode(args.path, args.latest)
        xyz, timestamps_ns, source_label = load_wrist_trajectory(episode, args.source)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    quality = trajectory_quality(xyz, timestamps_ns)
    print(
        "[TRAJECTORY] "
        f"source={source_label}, samples={quality['samples']}, duration={quality['duration_s']:.3f}s, "
        f"messages={quality['source_hz']:.2f}Hz, XYZ-updates={quality['xyz_update_hz']:.2f}Hz, "
        f"same={quality['adjacent_same_ratio']:.2%}, path={quality['path_length_m']:.3f}m, "
        f"max-step={quality['max_step_m'] * 100:.2f}cm"
    )
    output = args.output or episode.with_name(f"{episode.stem}_wrist_trajectory.png")
    plot_trajectory(
        xyz, timestamps_ns,
        title=f"{episode.name} — {source_label}",
        output=output,
        show=args.show,
        jump_threshold_m=args.jump_threshold_m,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
