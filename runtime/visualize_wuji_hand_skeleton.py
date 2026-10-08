#!/usr/bin/env python3
"""Render an FTP-1 Wuji hand skeleton episode to MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import zarr


EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
FINGER_COLORS = (
    (80, 80, 80), (255, 170, 40), (80, 210, 80),
    (60, 190, 255), (190, 100, 255), (255, 100, 130),
)
FLEX_NAMES = (
    "thumb_mcp", "thumb_ip", "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip", "ring_mcp", "ring_pip",
    "ring_dip", "pinky_mcp", "pinky_pip", "pinky_dip",
)


def _episode(path: Path) -> Path:
    if path.name.endswith(".zarr"):
        return path
    episodes = sorted(path.glob("episode_*.zarr"))
    if not episodes:
        raise ValueError(f"no episode_*.zarr below {path}")
    return episodes[-1]


def _project(points: np.ndarray, dims: tuple[int, int], box: tuple[int, int, int, int], scale: float) -> np.ndarray:
    x0, y0, width, height = box
    xy = points[:, dims].astype(np.float64)
    xy -= xy[0]
    xy[:, 1] *= -1
    xy *= scale
    xy += np.asarray([x0 + width / 2, y0 + height / 2])
    return np.rint(xy).astype(np.int32)


def _draw_hand(frame: np.ndarray, pixels: np.ndarray, title: str, origin: tuple[int, int]) -> None:
    cv2.putText(frame, title, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    for edge_index, (start, end) in enumerate(EDGES):
        finger = min(edge_index // 4 + 1, 5) if edge_index < 20 else 0
        cv2.line(frame, tuple(pixels[start]), tuple(pixels[end]), FINGER_COLORS[finger], 4, cv2.LINE_AA)
    for index, point in enumerate(pixels):
        cv2.circle(frame, tuple(point), 5 if index else 7, (245, 245, 245), -1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="episode Zarr or directory containing episodes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=0.0, help="0 uses the canonical timestamp rate")
    args = parser.parse_args()
    episode = _episode(args.path.expanduser().resolve())
    root = zarr.open_group(str(episode), mode="r")
    skeleton = np.asarray(root["audit/wuji_hand_skeleton_mediapipe"][:], dtype=np.float32)
    joints = np.asarray(root["data/right_hand_joints"][:], dtype=np.float32)
    timestamps = np.asarray(root["data/timestamps"][:], dtype=np.int64)
    if skeleton.shape != (len(timestamps), 21, 3) or joints.shape != (len(timestamps), 21):
        raise ValueError("inconsistent skeleton/joint/timestamp shapes")
    names = list(root.attrs.get("right_hand_joint_names", []))
    flex_indices = [names.index(name) for name in FLEX_NAMES]
    source_fps = (len(timestamps) - 1) / ((timestamps[-1] - timestamps[0]) / 1e9)
    fps = args.fps or source_fps
    output = (args.output or episode.with_name(episode.stem + "_hand_skeleton.mp4")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    centered = skeleton - skeleton[:, :1]
    extent = float(np.percentile(np.abs(centered), 99.5))
    scale = 245.0 / max(extent, 1e-6)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output}")
    duration = (timestamps[-1] - timestamps[0]) / 1e9
    try:
        for index, points in enumerate(centered):
            canvas = np.full((720, 1280, 3), 24, dtype=np.uint8)
            elapsed = (timestamps[index] - timestamps[0]) / 1e9
            cv2.putText(canvas, f"Wuji hand skeleton | frame {index + 1}/{len(timestamps)} | {elapsed:6.2f}/{duration:.2f}s",
                        (28, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
            xy = _project(points, (0, 1), (20, 60, 430, 620), scale)
            xz = _project(points, (0, 2), (450, 60, 430, 620), scale)
            _draw_hand(canvas, xy, "SDK skeleton XY", (35, 82))
            _draw_hand(canvas, xz, "SDK skeleton XZ", (465, 82))

            values = np.rad2deg(joints[index, flex_indices])
            cv2.putText(canvas, "FTP-1 flexion (deg)", (900, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2, cv2.LINE_AA)
            for row, (name, value) in enumerate(zip(FLEX_NAMES, values)):
                y = 112 + row * 39
                cv2.putText(canvas, name, (900, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1, cv2.LINE_AA)
                cv2.rectangle(canvas, (1060, y - 13), (1235, y + 3), (65, 65, 65), -1)
                length = int(np.clip(abs(value) / 120.0, 0.0, 1.0) * 175)
                color = (80, 210, 80) if abs(value) > 1.0 else (70, 70, 220)
                cv2.rectangle(canvas, (1060, y - 13), (1060 + length, y + 3), color, -1)
                cv2.putText(canvas, f"{value:+6.1f}", (1180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 245, 245), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    print(f"[SKELETON] episode={episode}")
    print(f"[SKELETON] frames={len(timestamps)}, duration={duration:.3f}s, fps={fps:.3f}")
    print(f"[SKELETON] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
