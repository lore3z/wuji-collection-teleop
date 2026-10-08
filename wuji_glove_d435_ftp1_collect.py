#!/usr/bin/env python3
"""Record Wuji glove + configurable first-person RGB + Gemini to FTP-1 Zarr.

Every saved episode is one ``episode_XXXXXX.zarr`` directory. The canonical
time axis is the configured first-person RGB stream (RealSense or PICO VR).
``--no-ego-camera`` disables that stream and uses Wuji skeleton
timestamps instead; no placeholder image is written.  Third-person Gemini RGB
and Wuji source frames are retained on their native/aligned timelines.
When configured, PICO Tracker ``PoseStamped`` streams are captured at their
native SDK update rate; tracker 190573G supplies the camera pose (z + 0.10 m) and
tracker 190056G supplies the wrist pose.

This is deliberately RGB-only. Camera launchers run separately.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from wuji_glove_d435_collect import (
    DEFAULT_GLOVE_SN,
    D435Node,
    TactileHealthMask,
    WUJI_OFFICIAL_ACTIVE_TAXELS,
    WujiGloveSource,
    _tactile_features,
    _cleanup_stale_mapped_frame_stores,
    _decode_messages_to_memmap,
    load_tactile_health_mask,
    tactile_active_taxel_count,
)
from wuji_serial_emg_source import SerialEmgSource
from wuji_myo_emg_source import MyoEmgSource
from wuji_ftp1_hand_geometry import (
    FTP1_HAND_FAAS_IDX,
    FTP1_HAND_NAMES,
    ftp1_right_hand_joints_from_mediapipe,
    ftp1_right_hand_joints_official_from_mediapipe,
)


# Wuji's SDK has a fixed (thumb, index, middle, ring, pinky) ``(5, 5)``
# interface.  Thumb uses all five values; the fifth value of each other finger
# is interface padding and is always zero.  These indices and names are the
# vendor's 21-DoF URDF order, not a guessed layout.
WUJI_21_FROM_25 = np.asarray(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23],
    dtype=np.intp,
)
WUJI_21_NAMES = (
    "thumb_cmc_rot", "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
    "index_finger_mcp_flex", "index_finger_mcp_abd", "index_finger_pip", "index_finger_dip",
    "middle_finger_mcp_flex", "middle_finger_mcp_abd", "middle_finger_pip", "middle_finger_dip",
    "ring_finger_mcp_flex", "ring_finger_mcp_abd", "ring_finger_pip", "ring_finger_dip",
    "pinky_mcp_flex", "pinky_mcp_abd", "pinky_pip", "pinky_dip",
)
# Local IDs identify the raw Wuji anatomical channels.  They are retained as
# audit metadata; they are deliberately different from FTP-1 FAAS indices.
WUJI_LOCAL_HAND_IDX = np.arange(len(WUJI_21_NAMES), dtype=np.int32)
WUJI_TACTILE_ZONE_AREAS = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32)
WUJI_TACTILE_ZONE_NAMES = ("thumb", "index", "middle", "ring", "pinky", "palm")
TACTILE_GROUP = "wuji"
TACTILE_SENSOR = "wuji_wg1k_pressure_array"
TACTILE_ZONE_SENSOR = "wuji_wg1k_pressure_zones"
# The current FTP-1 model has a dedicated MatrixCNNEncoder for tactile
# ``matrix`` inputs.  A 24x31 pressure field must use it; calling it ``state``
# would incorrectly route it through the vector-state encoder.
TACTILE_TYPE = "matrix"
ADDUCTION_CHANNELS = np.asarray([0, 1, 4, 8, 12, 16], dtype=np.intp)
JOINT_PREFLIGHT_NAMES = ("index_pip", "index_dip", "middle_pip")
JOINT_PREFLIGHT_INDICES = np.asarray(
    [FTP1_HAND_NAMES.index(name) for name in JOINT_PREFLIGHT_NAMES], dtype=np.intp
)


def _joint_preflight_quality(
    skeleton_frames: list[tuple[int, int, np.ndarray]], min_range_rad: float
) -> tuple[list[str], dict[str, float]]:
    """Find key flexion channels that did not move during an operator check."""
    if len(skeleton_frames) < 2:
        return list(JOINT_PREFLIGHT_NAMES), {name: 0.0 for name in JOINT_PREFLIGHT_NAMES}
    landmarks = np.stack([frame[2] for frame in skeleton_frames]).astype(np.float32)
    joints = ftp1_right_hand_joints_from_mediapipe(landmarks)
    ranges = np.ptp(joints[:, JOINT_PREFLIGHT_INDICES], axis=0)
    by_name = {name: float(value) for name, value in zip(JOINT_PREFLIGHT_NAMES, ranges)}
    frozen = [name for name, value in by_name.items() if value < min_range_rad]
    return frozen, by_name


def _wrist_pose_with_tracker_translation(
    wuji_wrist_pose: np.ndarray, tracker_wrist_pose: np.ndarray
) -> np.ndarray:
    """Use tracker1 XYZ with Wuji glove orientation for FTP-1 wrist pose."""
    wuji = np.asarray(wuji_wrist_pose, dtype=np.float32)
    tracker = np.asarray(tracker_wrist_pose, dtype=np.float32)
    if wuji.shape != tracker.shape or wuji.ndim != 2 or wuji.shape[1] != 6:
        raise ValueError(f"expected matching (T,6) wrist poses, got {wuji.shape} and {tracker.shape}")
    result = wuji.copy()
    result[:, :3] = tracker[:, :3]
    return result


def _canonical_with_stable_adduction(
    published: np.ndarray, stable: np.ndarray
) -> np.ndarray:
    """Keep FTP-1 layout while replacing degenerate projected-axis channels.

    The published MCP-to-tip adduction formula becomes ill-conditioned when a
    curled finger's long axis has almost no palm-plane projection.  The local
    implementation measures the same MCP side-swing from MCP-to-PIP, which is
    independent of distal flexion.  Non-adduction channels remain byte-for-
    byte the published FTP-1 result.
    """
    published = np.asarray(published, dtype=np.float32)
    stable = np.asarray(stable, dtype=np.float32)
    if published.shape != stable.shape or published.ndim != 2 or published.shape[1] != 21:
        raise ValueError(
            f"expected matching (T,21) joint arrays, got {published.shape} and {stable.shape}"
        )
    result = published.copy()
    result[:, ADDUCTION_CHANNELS] = stable[:, ADDUCTION_CHANNELS]
    return result


def _nearest_indices(source_us: np.ndarray, target_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return closest source index and signed ``source-target`` age in us."""
    if source_us.ndim != 1 or len(source_us) == 0:
        raise ValueError("empty source timestamps")
    right = np.searchsorted(source_us, target_us, side="left")
    right = np.clip(right, 0, len(source_us) - 1)
    left = np.clip(right - 1, 0, len(source_us) - 1)
    choose_left = np.abs(source_us[left] - target_us) <= np.abs(source_us[right] - target_us)
    indices = np.where(choose_left, left, right).astype(np.int64)
    return indices, (source_us[indices] - target_us).astype(np.int64)


def _live_canonical_latest_ns(
    camera_windows: dict[str, list[int]],
    *,
    capture_ego: bool,
    skeleton_frames: list[tuple[int, int, np.ndarray]],
) -> int:
    """Safely read the live watermark while save detaches camera buffers."""
    if capture_ego:
        ego_timestamps = camera_windows.get("ego", [])
        return int(ego_timestamps[-1]) if ego_timestamps else 0
    return int(skeleton_frames[-1][0]) * 1_000 if skeleton_frames else 0


def _rate(timestamps_ns: np.ndarray) -> float:
    if len(timestamps_ns) < 2:
        return 0.0
    return float((len(timestamps_ns) - 1) / max((timestamps_ns[-1] - timestamps_ns[0]) / 1e9, 1e-9))


def _emg_corrected_timestamps_ns(
    frames: list[tuple[int, int, int, np.ndarray, int]],
) -> tuple[np.ndarray, int]:
    """Map the EMG source clock onto the host wall-clock axis.

    The serial protocol has no device timestamp, so the bridge records packet
    arrival time. Low-decile arrival-minus-source anchors keep this compatible
    with sources that reconstruct their own sample clock without creating,
    deleting or reordering any EMG sample.
    """
    if not frames:
        return np.empty(0, dtype=np.int64), 0
    raw = np.asarray([frame[0] for frame in frames], dtype=np.int64)
    arrival = np.asarray([frame[4] for frame in frames], dtype=np.int64)
    residual = arrival - raw
    if len(raw) < 40 or raw[-1] <= raw[0]:
        rank = min(max(int(len(residual) * 0.10), 0), len(residual) - 1)
        offset_ns = int(np.partition(residual, rank)[rank])
        return raw + offset_ns, offset_ns

    bin_count = min(20, max(2, len(raw) // 100))
    anchors_x: list[float] = []
    anchors_y: list[float] = []
    for indices in np.array_split(np.arange(len(raw)), bin_count):
        if not len(indices):
            continue
        local = residual[indices]
        cutoff = float(np.quantile(local, 0.20))
        low_latency = indices[local <= cutoff]
        anchors_x.append(float(np.median(raw[low_latency] - raw[0])))
        anchors_y.append(float(np.median(residual[low_latency])))
    slope, intercept = np.polyfit(np.asarray(anchors_x), np.asarray(anchors_y), 1)
    # A larger correction indicates corrupt clocks. Bounding it also
    # guarantees positive sample steps.
    slope = float(np.clip(slope, -0.05, 0.05))
    if abs(slope) < 1e-4:
        slope = 0.0
    raw_relative = raw - raw[0]
    corrected_relative = np.rint(
        raw_relative.astype(np.float64) * (1.0 + slope) + intercept
    ).astype(np.int64)
    # Add the epoch only after rounding. Converting a ~1.7e18 ns Unix epoch to
    # float64 first loses sub-microsecond precision from every sample step.
    corrected = raw[0] + corrected_relative
    if np.any(np.diff(corrected) <= 0):
        raise ValueError("corrected EMG timestamps are not strictly increasing")
    return corrected, int(round(intercept))


def _emg_episode_clock_arrays(
    frames: list[tuple[int, int, int, np.ndarray, int]], keep: np.ndarray
) -> tuple[list[tuple[int, int, int, np.ndarray, int]], np.ndarray, np.ndarray, np.ndarray, int]:
    """Filter an episode without losing its affine-corrected EMG clock."""
    corrected_all, offset_ns = _emg_corrected_timestamps_ns(frames)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (len(frames),):
        raise ValueError("EMG keep mask length mismatch")
    selected = [frame for frame, selected in zip(frames, keep) if selected]
    corrected = corrected_all[keep]
    raw = np.asarray([frame[0] for frame in selected], dtype=np.int64)
    arrival = np.asarray([frame[4] for frame in selected], dtype=np.int64)
    return selected, corrected, raw, arrival, offset_ns


def _tracker_xyz_quality(
    frames: list[tuple[int, int, np.ndarray]], *, change_epsilon_m: float = 1e-6
) -> dict[str, float | int]:
    """Describe position-value continuity separately from message cadence."""
    if len(frames) < 2:
        return {
            "adjacent_same_count": 0, "adjacent_same_ratio": 0.0,
            "xyz_update_count": 0, "xyz_update_hz": 0.0,
            "xyz_span_m": 0.0, "xyz_max_step_m": 0.0,
        }
    timestamps_ns = np.asarray([frame[0] for frame in frames], dtype=np.int64)
    xyz = np.stack([frame[2][:3] for frame in frames]).astype(np.float64)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    changed = steps > float(change_epsilon_m)
    duration_s = max((timestamps_ns[-1] - timestamps_ns[0]) / 1e9, 1e-9)
    same_count = int(np.count_nonzero(~changed))
    return {
        "adjacent_same_count": same_count,
        "adjacent_same_ratio": same_count / len(steps),
        "xyz_update_count": int(np.count_nonzero(changed)),
        "xyz_update_hz": float(np.count_nonzero(changed) / duration_s),
        "xyz_span_m": float(np.linalg.norm(np.ptp(xyz, axis=0))),
        "xyz_max_step_m": float(steps.max(initial=0.0)),
    }


def _timeline_quality(timestamps_ns: np.ndarray) -> dict[str, float | int]:
    """Measure missing nominal camera slots without inventing replacement frames."""
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    if len(timestamps_ns) < 2:
        return {
            "nominal_period_ns": 0,
            "max_gap_ns": 0,
            "missing_slot_count": 0,
            "missing_slot_ratio": 0.0,
        }
    dt = np.diff(timestamps_ns)
    if np.any(dt <= 0):
        raise ValueError("camera timestamps must be strictly increasing")
    nominal = int(np.median(dt))
    # Tolerate up to 25% timing jitter before classifying a gap as an extra
    # nominal slot. This prevents decoder delivery jitter from becoming a
    # false missing-frame report.
    slot_steps = np.maximum(
        np.floor(dt / max(nominal, 1) + 0.25).astype(np.int64), 1
    )
    missing = int(np.maximum(slot_steps - 1, 0).sum())
    return {
        "nominal_period_ns": nominal,
        "max_gap_ns": int(dt.max()),
        "missing_slot_count": missing,
        "missing_slot_ratio": float(missing / max(len(timestamps_ns) + missing, 1)),
    }


def _unique_source_mask(indices: np.ndarray, ages_us: np.ndarray) -> np.ndarray:
    """Keep only the closest target when nearest-neighbour reused one source frame."""
    indices = np.asarray(indices, dtype=np.int64)
    ages_us = np.asarray(ages_us, dtype=np.int64)
    keep = np.zeros(len(indices), dtype=bool)
    for source_index in np.unique(indices):
        positions = np.flatnonzero(indices == source_index)
        keep[positions[np.argmin(np.abs(ages_us[positions]))]] = True
    return keep


def _is_full_identity_selection(indices: np.ndarray, source_length: int) -> bool:
    """Whether indices retain every source row, including the final row."""
    indices = np.asarray(indices, dtype=np.intp)
    return (
        len(indices) == int(source_length)
        and np.array_equal(indices, np.arange(source_length, dtype=np.intp))
    )


def _max_false_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run of false entries."""
    invalid = np.flatnonzero(~np.asarray(mask, dtype=bool))
    if not len(invalid):
        return 0
    boundaries = np.flatnonzero(np.diff(invalid) > 1) + 1
    return max(len(run) for run in np.split(invalid, boundaries))


def _can_drop_sparse_invalid_rows(
    valid: np.ndarray,
    max_drop_ratio: float,
    max_drop_run: int = 3,
) -> bool:
    """Whether sparse invalid canonical rows may be removed safely."""
    valid = np.asarray(valid, dtype=bool)
    invalid_count = int(np.count_nonzero(~valid))
    return (
        invalid_count > 0
        and np.count_nonzero(valid) >= 2
        and invalid_count / max(len(valid), 1) <= max_drop_ratio
        and _max_false_run(valid) <= max_drop_run
    )


def _only_tracker_rows_invalid(
    tracker_camera_ok: np.ndarray | None,
    tracker_wrist_ok: np.ndarray | None,
    non_tracker_masks: tuple[np.ndarray | None, ...],
) -> bool:
    """Whether every invalid canonical row is caused only by Tracker age."""
    if tracker_camera_ok is None or tracker_wrist_ok is None:
        return False
    tracker_invalid = ~np.asarray(tracker_camera_ok, dtype=bool) | ~np.asarray(
        tracker_wrist_ok, dtype=bool
    )
    return bool(
        np.any(tracker_invalid)
        and all(mask is None or bool(np.all(mask)) for mask in non_tracker_masks)
    )


def _only_wuji_rows_invalid(
    wuji_masks: tuple[np.ndarray, ...],
    non_wuji_masks: tuple[np.ndarray | None, ...],
) -> bool:
    """Whether invalid canonical rows are caused only by Wuji stream ages."""
    return bool(
        any(np.any(~np.asarray(mask, dtype=bool)) for mask in wuji_masks)
        and all(mask is None or bool(np.all(mask)) for mask in non_wuji_masks)
    )


def _emg_alignment_drop_run_limit(
    canonical_period_ns: int, max_emg_gap_ns: int, *, native_quality_ok: bool
) -> int:
    """Canonical rows removable for a bounded, otherwise healthy EMG hole."""
    if not native_quality_ok:
        return 3
    return max(3, int(max_emg_gap_ns // max(canonical_period_ns, 1)))


def _configure_wuji_sdk_logging() -> None:
    """Default Wuji SDK logs to warning level so INFO logs do not flood the terminal."""
    level = os.environ.get("WUJI_SDK_LOG_LEVEL", "warning").strip()
    if not level:
        return
    try:
        from wuji_sdk import set_log_level
    except ImportError:
        return
    try:
        set_log_level(level)
    except Exception as exc:
        print(f"[Warn] 无法设置 wuji_sdk 日志级别为 {level!r}: {exc}")


def _fast_rgb_jpeg_quality() -> int:
    """Return the bounded JPEG quality used by the human fast-save path."""
    try:
        return min(100, max(1, int(os.environ.get("WUJI_FAST_RGB_JPEG_QUALITY", "90"))))
    except ValueError:
        return 90


def _create_array(group: zarr.Group, key: str, value: np.ndarray) -> None:
    """Create a compact, time-major zarr v2 array."""
    value = np.asarray(value)
    if value.ndim == 0:
        raise ValueError(f"{key} must have a time dimension")
    # One RGB image per chunk keeps random video access cheap. State and
    # tactile matrices amortize Zarr metadata with 256 temporal samples.
    is_rgb = value.ndim == 4 and value.shape[-1] in (3, 4) and value.dtype == np.uint8
    if is_rgb:
        fast_save = os.environ.get("WUJI_FAST_SAVE") == "1"
        # PICO frames are decoded H.264 output and have high entropy. LZ4
        # barely compresses them, so the old fast path wrote nearly 1 GB for
        # a short episode and spent most of `e` waiting on disk I/O. JPEG
        # preserves the normal decoded ndarray interface while matching the
        # compact layout used by the reference episodes.
        if fast_save:
            try:
                from imagecodecs.numcodecs import Jpeg
                from numcodecs.registry import register_codec
            except ImportError:
                # Keep collection usable on minimal installs; the regular
                # path still reports the missing dependency explicitly.
                chunks = (8, *value.shape[1:])
                group.create_dataset(
                    key, data=value, shape=value.shape, chunks=chunks, dtype=value.dtype,
                    compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
                )
                return
            quality = _fast_rgb_jpeg_quality()
            register_codec(Jpeg)
            compressor = Jpeg(
                level=quality,
                colorspace_data="RGB",
                colorspace_jpeg="YCbCr",
                subsampling="420",
            )
            group.create_dataset(
                key, data=value, shape=value.shape, chunks=(1, *value.shape[1:]),
                dtype=value.dtype, compressor=compressor,
            )
            return
        # A one-frame JPEG numcodec keeps the standard Zarr uint8
        # ``(T,H,W,3)`` interface: FTP-1 sees an ordinary decoded ndarray,
        # while disk use is comparable to the MP4/JPEG camera transport rather
        # than multi-gigabyte raw RGB.  4:2:0 quality 90 measured >55 dB PSNR
        # on the native PICO stream, whose source has already passed H.264 and
        # JPEG encoding before reaching this writer.
        try:
            from imagecodecs.numcodecs import Jpeg
            from numcodecs.registry import register_codec
        except ImportError as exc:
            raise RuntimeError(
                "imagecodecs is required for compact RGB Zarr chunks; "
                "run ./setup.sh to install requirements-collector.txt"
            ) from exc
        register_codec(Jpeg)
        compressor = Jpeg(
            level=90,
            colorspace_data="RGB",
            colorspace_jpeg="YCbCr",
            subsampling="420",
        )
        chunks = (1, *value.shape[1:])
    else:
        compressor = Blosc(
            cname="zstd",
            clevel=3,
            shuffle=Blosc.NOSHUFFLE if value.dtype.kind in "SU" else Blosc.BITSHUFFLE,
        )
        chunks = (min(max(len(value), 1), 256), *value.shape[1:])
    group.create_dataset(key, data=value, shape=value.shape, chunks=chunks, dtype=value.dtype, compressor=compressor)


def _create_arrays_parallel(group: zarr.Group, arrays: dict[str, np.ndarray], expected_t: int) -> None:
    """Write independent arrays concurrently while preserving full contents."""
    def write(item: tuple[str, np.ndarray]) -> None:
        key, array = item
        if array.shape[0] != expected_t:
            raise RuntimeError(f"axis mismatch: {key} has {array.shape[0]}, expected {expected_t}")
        _create_array(group, key, array)

    workers = min(8, max(1, len(arrays)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(write, arrays.items()))


def _uniform_grid(source_ns: np.ndarray) -> tuple[np.ndarray, int]:
    """Create the canonical uniform grid without fabricating camera images."""
    if len(source_ns) < 2:
        raise ValueError("need at least two RGB timestamps")
    period = int(np.median(np.diff(source_ns)))
    if period <= 0:
        raise ValueError("RGB timestamps are not strictly increasing")
    count = int((source_ns[-1] - source_ns[0]) // period) + 1
    return source_ns[0] + np.arange(count, dtype=np.int64) * period, period


def _write_mp4(path: Path, rgb_frames: np.ndarray, fps: float, codec: str) -> tuple[bool, str]:
    """Write an RGB sequence as a portable MP4 sidecar without touching Zarr."""
    try:
        import cv2
    except ImportError:
        return False, "OpenCV unavailable (install opencv-python to export MP4)"
    frames = np.asarray(rgb_frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or len(frames) == 0:
        return False, f"unexpected RGB array {frames.shape} {frames.dtype}"
    if len(codec) != 4:
        return False, f"codec must be fourcc, got {codec!r}"
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), max(float(fps), 1.0), (width, height), True
    )
    if not writer.isOpened():
        return False, f"cannot open VideoWriter codec={codec!r} size={width}x{height}"
    try:
        for frame in frames:
            # Zarr stores RGB while OpenCV's writer accepts BGR.
            writer.write(np.ascontiguousarray(frame[..., ::-1]))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        return False, "VideoWriter produced no file"
    return True, f"{path.stat().st_size / (1024 * 1024):.1f} MiB"


def _sharpness_values(rgb_frames: np.ndarray) -> list[float]:
    """Return per-frame Laplacian variance for one contiguous frame batch."""
    import cv2

    return [
        float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        for frame in rgb_frames
    ]


def _sharpness_stats(rgb_frames: np.ndarray, max_workers: int = 8) -> dict[str, float]:
    """Laplacian-variance image sharpness, evaluated in parallel before blur."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the RGB sharpness quality gate") from exc
    frames = np.asarray(rgb_frames)
    worker_count = min(max(int(max_workers), 1), max(len(frames), 1))
    chunks = [chunk for chunk in np.array_split(frames, worker_count) if len(chunk)]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        chunk_values = list(executor.map(_sharpness_values, chunks))
    values = np.asarray([value for chunk in chunk_values for value in chunk], dtype=np.float64)
    return {
        "min": float(values.min()) if len(values) else 0.0,
        "p05": float(np.percentile(values, 5)) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
    }


class FTP1Collector:
    def __init__(
        self,
        output_dir: Path,
        glove: WujiGloveSource,
        camera: D435Node,
        emg: SerialEmgSource | None,
        myo: MyoEmgSource | None,
        *,
        max_glove_age_ms: float = 6.0,
        max_wrist_pose_age_ms: float = 15.0,
        max_gemini_age_ms: float = 10.0,
        max_camera_pose_age_ms: float = 20.0,
        max_rgb_gap_ms: float = 50.0,
        max_missing_ratio: float = 0.02,
        max_align_drop_ratio: float = 0.05,
        max_wuji_gap_ms: float = 500.0,
        max_joint_step_rad: float = 2.5,
        baseline_seconds: float = 2.0,
        save_mp4: bool = True,
        fast_save: bool = False,
        mp4_codec: str = "mp4v",
        min_ego_sharpness: float = 100.0,
        max_emg_age_ms: float = 10.0,
        min_emg_hz: float = 180.0,
        max_emg_missing_ratio: float = 0.02,
        max_emg_gap_ms: float = 350.0,
        emg_start_wait_s: float = 20.0,
        max_myo_age_ms: float = 100.0,
        min_myo_hz: float = 180.0,
        max_myo_missing_ratio: float = 0.02,
        max_myo_gap_ms: float = 350.0,
        myo_start_wait_s: float = 20.0,
        max_tracker_age_ms: float = 25.0,
        joint_preflight_seconds: float = 2.0,
        joint_preflight_min_range_deg: float = 3.0,
        tactile_health_mask: TactileHealthMask | None = None,
    ):
        self.output_dir = output_dir.expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.participant_name = os.environ.get("WUJI_PARTICIPANT_NAME", "")
        self.task_id = os.environ.get("WUJI_TASK_ID", "")
        self.glove_condition = os.environ.get("WUJI_GLOVE_CONDITION", "")
        stale_mmaps = _cleanup_stale_mapped_frame_stores()
        if stale_mmaps:
            print(f"[SAVE] removed {stale_mmaps} stale RGB mmap file(s) from a previous interrupted save")
        self.glove = glove
        self.camera = camera
        self.emg = emg
        self.emg_enabled = emg is not None
        self.myo = myo
        self.myo_enabled = myo is not None
        self.max_glove_age_us = int(max_glove_age_ms * 1000.0)
        self.max_wrist_pose_age_us = int(max_wrist_pose_age_ms * 1000.0)
        self.max_gemini_age_us = int(max_gemini_age_ms * 1000.0)
        self.max_camera_pose_age_us = int(max_camera_pose_age_ms * 1000.0)
        self.max_rgb_gap_ns = int(max_rgb_gap_ms * 1_000_000.0)
        self.max_missing_ratio = float(max_missing_ratio)
        self.camera.configure_live_rgb_quality(self.max_rgb_gap_ns, self.max_missing_ratio)
        self.max_align_drop_ratio = float(max_align_drop_ratio)
        self.max_wuji_gap_ns = int(max_wuji_gap_ms * 1_000_000.0)
        self.glove.reconnect_after_s = self.max_wuji_gap_ns / 1e9
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.baseline_seconds = float(baseline_seconds)
        self.save_mp4 = bool(save_mp4)
        self.fast_save = bool(fast_save)
        self._save_thread = None
        if self.fast_save:
            os.environ["WUJI_FAST_SAVE"] = "1"
        self.mp4_codec = str(mp4_codec)
        self.min_ego_sharpness = float(min_ego_sharpness)
        self.max_emg_age_us = int(max_emg_age_ms * 1000.0)
        self.min_emg_hz = float(min_emg_hz)
        self.max_emg_missing_ratio = float(max_emg_missing_ratio)
        self.max_emg_gap_ns = int(max_emg_gap_ms * 1_000_000.0)
        self.emg_start_wait_s = float(emg_start_wait_s)
        self.max_myo_age_us = int(max_myo_age_ms * 1000.0)
        self.min_myo_hz = float(min_myo_hz)
        self.max_myo_missing_ratio = float(max_myo_missing_ratio)
        self.max_myo_gap_ns = int(max_myo_gap_ms * 1_000_000.0)
        self.myo_start_wait_s = float(myo_start_wait_s)
        self.max_tracker_age_us = int(max_tracker_age_ms * 1000.0)
        self.joint_preflight_seconds = float(joint_preflight_seconds)
        self.joint_preflight_min_range_rad = float(np.deg2rad(joint_preflight_min_range_deg))
        self.tactile_health_mask = tactile_health_mask
        self.state_lock = threading.Lock()
        self.recording = False
        self.last_refusal_reason = ""
        self.angle_frames: list[tuple[int, int, np.ndarray]] = []
        self.tactile_frames: list[tuple[int, int, np.ndarray]] = []
        self.skeleton_frames: list[tuple[int, int, np.ndarray]] = []
        self.zone_frames: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        self.right_wrist_pose_frames: list[tuple[int, int, np.ndarray]] = []
        self.emg_frames: list[tuple[int, int, int, np.ndarray, int]] = []
        self.emg_imu_frames: list[tuple[int, int, np.ndarray, int]] = []
        self.myo_frames: list[tuple[int, int, int, np.ndarray, int]] = []
        self.emg_last_episode_ts_ns = 0
        self.emg_episode_connection_id = 0
        self.emg_episode_interruption_id = 0
        self.emg_invalid_reason = ""
        self.emg_warning_reason = ""
        self.emg_alignment_warning_count = 0
        self.emg_alignment_warning_max_gap_ns = 0
        self.emg_next_clock_warning_mono = 0.0
        self.emg_clock_violation_started_mono = 0.0
        self.myo_last_episode_ts_ns = 0
        self.myo_episode_connection_id = 0
        self.myo_episode_interruption_id = 0
        self.myo_invalid_reason = ""
        self.live_last_canonical_joints: np.ndarray | None = None
        self.live_alignment_violation_started: dict[str, float] = {}
        self.live_next_alignment_check_mono = 0.0
        self.joint_preflight_capturing = False
        self.joint_preflight_frames: list[tuple[int, int, np.ndarray]] = []
        self.joint_preflight_passed = False
        self.instruction = ""
        self.baseline_capturing = False
        self.baseline_frames: list[tuple[int, int, np.ndarray]] = []
        self.baseline_zone_frames: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        self.tactile_baseline: np.ndarray | None = None
        self.tactile_zone_baseline: np.ndarray | None = None
        self.tactile_baseline_count = 0
        self.tactile_baseline_captured_ns = 0
        self.protocol_confirmed_ns = 0
        self.online_bridge = None
        online_url = os.environ.get("EMG2POSE_ONLINE_URL")
        if online_url:
            online_package = os.environ.get("EMG2POSE_ONLINE_PACKAGE")
            if not online_package:
                raise ValueError("Set EMG2POSE_ONLINE_PACKAGE to the online model package")
            sys.path.append(str(Path(online_package).expanduser().resolve()))
            from collector_bridge import CollectorBridge
            self.online_bridge = CollectorBridge(
                online_url,
                lambda landmarks: _canonical_with_stable_adduction(
                    ftp1_right_hand_joints_official_from_mediapipe(landmarks),
                    ftp1_right_hand_joints_from_mediapipe(landmarks),
                ),
            )
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._drain_loop, daemon=True)
        self.thread.start()
        self._install_reader()

    def _install_reader(self) -> None:
        """Keep a portable reader in the same directory as collected Zarrs."""
        source = Path(__file__).with_name("read_wuji_glove_ftp1_zarr.py")
        destination = self.output_dir / source.name
        if not source.is_file():
            raise RuntimeError(f"缺少配套读取脚本：{source}")
        shutil.copy2(source, destination)
        renderer = Path(__file__).parent / "runtime" / "render_episode_sync.py"
        if renderer.is_file():
            shutil.copy2(renderer, self.output_dir / renderer.name)

    def _drain_loop(self) -> None:
        while not self.stop_event.is_set():
            monitor_recording = False
            live_fault = ""
            # The state lock makes draining and the recording decision atomic;
            # save() can no longer lose an in-flight batch at the episode edge.
            with self.state_lock:
                angles, tactile = self.glove.drain()
                skeleton = self.glove.drain_skeleton()
                zones = self.glove.drain_zones()
                wrist_poses = self.glove.drain_right_wrist_pose()
                emg = self.emg.drain() if self.emg is not None else []
                emg_imu = getattr(self.emg, "drain_imu", lambda: [])() if self.emg is not None else []
                myo_source = getattr(self, "myo", None)
                myo = myo_source.drain() if myo_source is not None else []
                if self.baseline_capturing:
                    self.baseline_frames.extend(tactile)
                    self.baseline_zone_frames.extend(zones)
                if self.joint_preflight_capturing:
                    self.joint_preflight_frames.extend(skeleton)
                if self.recording:
                    monitor_recording = True
                    self.angle_frames.extend(angles)
                    self.tactile_frames.extend(tactile)
                    self.skeleton_frames.extend(skeleton)
                    self.zone_frames.extend(zones)
                    self.right_wrist_pose_frames.extend(wrist_poses)
                    bad_tactile = next(
                        (
                            tactile_active_taxel_count(frame[2])
                            for frame in tactile
                            if tactile_active_taxel_count(frame[2]) != WUJI_OFFICIAL_ACTIVE_TAXELS
                        ),
                        None,
                    )
                    bad_zone = next(
                        (
                            int(np.asarray(frame[3], dtype=np.int32).sum())
                            for frame in zones
                            if int(np.asarray(frame[3], dtype=np.int32).sum()) != WUJI_OFFICIAL_ACTIVE_TAXELS
                        ),
                        None,
                    )
                    if bad_tactile is not None or bad_zone is not None:
                        live_fault = (
                            "Wuji tactile contract mismatch during recording: "
                            f"matrix={bad_tactile}, zones={bad_zone}, "
                            f"official={WUJI_OFFICIAL_ACTIVE_TAXELS}"
                        )
                    if not live_fault and skeleton:
                        try:
                            landmarks = np.stack([frame[2] for frame in skeleton]).astype(np.float32)
                            published = ftp1_right_hand_joints_official_from_mediapipe(landmarks)
                            stable = ftp1_right_hand_joints_from_mediapipe(landmarks)
                            canonical = _canonical_with_stable_adduction(published, stable)
                            if self.live_last_canonical_joints is not None:
                                canonical = np.vstack((self.live_last_canonical_joints[None, :], canonical))
                            if len(canonical) >= 2:
                                max_step = float(np.abs(np.diff(canonical, axis=0)).max())
                                if max_step > self.max_joint_step_rad:
                                    live_fault = (
                                        f"canonical hand joint 实时跳变={max_step:.3f} rad "
                                        f"> {self.max_joint_step_rad:.3f} rad"
                                    )
                            self.live_last_canonical_joints = canonical[-1].copy()
                        except Exception as exc:
                            live_fault = f"手套骨架实时质量检查失败：{type(exc).__name__}: {exc}"
                    if self.emg_enabled and self.emg is not None and self.emg.current_connection_id() != self.emg_episode_connection_id:
                        self._mark_emg_reconnect(self.emg.current_connection_id())
                    if self.emg_enabled and self.emg is not None:
                        for frame in emg:
                            gap_ns = frame[0] - self.emg_last_episode_ts_ns if self.emg_last_episode_ts_ns else 0
                            if gap_ns > self.max_emg_gap_ns:
                                self._mark_emg_gap(gap_ns)
                            elif gap_ns > self.max_emg_age_us * 1_000:
                                self.emg_alignment_warning_count += 1
                                self.emg_alignment_warning_max_gap_ns = max(
                                    self.emg_alignment_warning_max_gap_ns, gap_ns
                                )
                            self.emg_last_episode_ts_ns = frame[0]
                        self.emg_frames.extend(emg)
                        getattr(self, "emg_imu_frames", []).extend(emg_imu)
                        if self.emg.age_ms() * 1_000_000.0 > self.max_emg_gap_ns:
                            self._mark_emg_gap(int(self.emg.age_ms() * 1_000_000.0))
                    if getattr(self, "myo_enabled", False) and myo_source is not None:
                        if myo_source.current_connection_id() != self.myo_episode_connection_id:
                            self.myo_invalid_reason = (
                                f"Myo BLE reconnected during episode: connection "
                                f"{self.myo_episode_connection_id}->{myo_source.current_connection_id()}"
                            )
                        for frame in myo:
                            gap_ns = frame[0] - self.myo_last_episode_ts_ns if self.myo_last_episode_ts_ns else 0
                            if gap_ns > self.max_myo_gap_ns and not self.myo_invalid_reason:
                                self.myo_invalid_reason = f"Myo RAW EMG interrupted for {gap_ns / 1e6:.0f} ms"
                            self.myo_last_episode_ts_ns = frame[0]
                        self.myo_frames.extend(myo)
                        if myo_source.age_ms() * 1_000_000.0 > self.max_myo_gap_ns and not self.myo_invalid_reason:
                            self.myo_invalid_reason = f"Myo RAW EMG interrupted for {myo_source.age_ms():.0f} ms"
                    if not live_fault:
                        live_fault = self._emg_live_fault()
                    if not live_fault:
                        live_fault = self._myo_live_fault()
            online_bridge = getattr(self, "online_bridge", None)
            if online_bridge is not None:
                online_bridge.publish(myo, skeleton)
            if monitor_recording and self.camera.capture_ego and not live_fault:
                live_fault = self.camera.live_rgb_fault()
            if monitor_recording and not live_fault:
                live_fault = self.camera.live_source_fault()
            if monitor_recording and not live_fault and (time.time_ns() - self.started_ns) >= 500_000_000:
                if not self.glove.healthy(max_age_s=self.max_wuji_gap_ns / 1e9):
                    live_fault = f"Wuji source stopped or became unhealthy: {self.glove.status()}"
                elif not self.camera.ready(max_age_s=0.25):
                    live_fault = f"camera/IMU/Tracker source stopped updating: {self.camera.status()}"
            now_mono = time.monotonic()
            if (
                monitor_recording
                and not live_fault
                and (time.time_ns() - self.started_ns) >= 500_000_000
                and now_mono >= self.live_next_alignment_check_mono
            ):
                self.live_next_alignment_check_mono = now_mono + 0.02
                camera_windows = self.camera.live_alignment_timestamp_windows_ns()
                with self.state_lock:
                    # save()/refuse() may have ended the episode and detached
                    # camera buffers after monitor_recording was sampled at
                    # the top of this loop iteration. Recheck state and treat
                    # an empty boundary snapshot as normal transition state.
                    monitor_recording = monitor_recording and self.recording
                    canonical_latest_ns = _live_canonical_latest_ns(
                        camera_windows,
                        capture_ego=self.camera.capture_ego,
                        skeleton_frames=self.skeleton_frames,
                    )
                    # Inspect a completed historical point, not the newest
                    # callback from each asynchronous source. USB serial may
                    # deliver transport batches, so a 300 ms historical point
                    # ensures the corresponding samples have arrived before
                    # applying the same strict nearest-age gate used at save.
                    canonical_ns = canonical_latest_ns - 300_000_000 if canonical_latest_ns else 0
                    live_sources = {
                        "angle": ([frame[0] * 1_000 for frame in self.angle_frames[-64:]], self.max_glove_age_us * 1_000),
                        "tactile": ([frame[0] * 1_000 for frame in self.tactile_frames[-64:]], self.max_glove_age_us * 1_000),
                        "skeleton": ([frame[0] * 1_000 for frame in self.skeleton_frames[-64:]], self.max_glove_age_us * 1_000),
                        # Wuji dynamic TF can arrive near 800 Hz. Sixty-four
                        # samples cover less than the 300 ms watermark and
                        # falsely make the historical target look stale.
                        "wrist_tf": ([frame[0] * 1_000 for frame in self.right_wrist_pose_frames[-512:]], self.max_wrist_pose_age_us * 1_000),
                    }
                        # Do not put EMG into this generic alignment watchdog.
                        # USB serial may deliver a healthy stream in short
                        # bursts. EMG is checked against its complete native
                        # sequence at save; sparse stale canonical rows are
                        # removed only when rate, sequence gaps and max gap are
                        # still within the quality contract.
                if self.camera.gemini_imu_enabled:
                    live_sources["gemini_imu"] = (camera_windows["gemini_imu"], self.max_camera_pose_age_us * 1_000)
                if self.camera.trackers_enabled:
                    live_sources["tracker_camera"] = (camera_windows["tracker_camera"], self.max_tracker_age_us * 1_000)
                    live_sources["tracker_wrist"] = (camera_windows["tracker_wrist"], self.max_tracker_age_us * 1_000)
                if canonical_ns:
                    # EMG acquisition timestamps are mapped from the native
                    # source clock. Do not reject online from a rolling
                    # affine-clock nearest-age estimate: USB can deliver a
                    # healthy stream in batches, and save() has the complete
                    # source sequence for the authoritative real-neighbor gate.
                    violating_sources: set[str] = set()
                    for name, (source_stamps, limit_ns) in live_sources.items():
                        nearest_age_ns = (
                            min(abs(int(stamp) - canonical_ns) for stamp in source_stamps)
                            if source_stamps else 0
                        )
                        if nearest_age_ns > limit_ns:
                            violating_sources.add(name)
                            started = self.live_alignment_violation_started.setdefault(name, now_mono)
                            grace_s = (
                                self.max_wuji_gap_ns / 1e9
                                if name in {"angle", "tactile", "skeleton", "wrist_tf"}
                                else 0.04
                            )
                            if now_mono - started >= grace_s:
                                live_fault = (
                                    f"{name} delayed-watermark nearest age={nearest_age_ns / 1e6:.1f} ms "
                                    f"> {limit_ns / 1e6:.1f} ms continuously for {grace_s:g} s"
                                )
                                break
                    for recovered in set(self.live_alignment_violation_started) - violating_sources:
                        self.live_alignment_violation_started.pop(recovered, None)
            if monitor_recording and self.camera.trackers_enabled and not live_fault and (time.time_ns() - self.started_ns) >= 2_000_000_000:
                _camera_tracker, wrist_tracker = self.camera.live_tracker_frames()
                wrist_quality = _tracker_xyz_quality(wrist_tracker)
                if (
                    wrist_quality["xyz_span_m"] >= 0.02
                    and wrist_quality["xyz_update_hz"] < 10.0
                ):
                    live_fault = (
                        "PICO wrist Tracker XYZ 非连续更新："
                        f"span={wrist_quality['xyz_span_m'] * 100:.1f} cm, "
                        f"updates={wrist_quality['xyz_update_count']} "
                        f"({wrist_quality['xyz_update_hz']:.1f} Hz), "
                        f"same={wrist_quality['adjacent_same_ratio']:.1%}, "
                        f"max-step={wrist_quality['xyz_max_step_m'] * 100:.1f} cm"
                    )
            if live_fault:
                self.refuse(live_fault, automatic=True)
            time.sleep(0.001)

    def _mark_emg_gap(self, gap_ns: int) -> None:
        if not self.emg_invalid_reason:
            self.emg_invalid_reason = f"serial EMG interrupted for {gap_ns / 1e6:.0f} ms"

    def _mark_emg_reconnect(self, connection_id: int) -> None:
        if not self.emg_invalid_reason:
            self.emg_invalid_reason = (
                f"serial EMG reconnected during episode: connection "
                f"{self.emg_episode_connection_id}->{connection_id}"
            )

    def _emg_live_fault(self) -> str:
        if not self.emg_enabled or self.emg is None:
            return ""
        if self.emg_invalid_reason:
            return self.emg_invalid_reason
        interruption_id, reason = self.emg.interruption_snapshot()
        if interruption_id != self.emg_episode_interruption_id:
            return f"串口 EMG 断流：{reason}"
        if self.emg.current_connection_id() != self.emg_episode_connection_id:
            return "串口 EMG 连接已更换"
        if not self.emg.healthy(max_age_s=min(self.max_emg_gap_ns / 1e9, self.emg.silence_timeout_s)):
            return "串口 EMG 已停止更新或正在恢复"
        return ""

    def _myo_live_fault(self) -> str:
        myo = getattr(self, "myo", None)
        if not getattr(self, "myo_enabled", False) or myo is None:
            return ""
        if self.myo_invalid_reason:
            return self.myo_invalid_reason
        interruption_id, reason = myo.interruption_snapshot()
        if interruption_id != self.myo_episode_interruption_id:
            return f"Myo 断流：{reason}"
        if myo.current_connection_id() != self.myo_episode_connection_id:
            return "Myo BLE 连接已更换"
        if not myo.healthy(max_age_s=min(self.max_myo_gap_ns / 1e9, myo.silence_timeout_s)):
            return "Myo RAW EMG 已停止更新或正在恢复"
        return ""

    def capture_baseline(self) -> None:
        """Capture a fresh unloaded tactile baseline for the currently worn glove."""
        with self.state_lock:
            if self.recording or self.baseline_capturing:
                print("[REFUSED] 请在未采集时标定触觉基线")
                return
            self.glove.clear()
            self.baseline_frames = []
            self.baseline_zone_frames = []
            self.joint_preflight_frames = []
            self.joint_preflight_passed = False
            self.baseline_capturing = True
            self.joint_preflight_capturing = True
        check_seconds = max(self.baseline_seconds, self.joint_preflight_seconds)
        print(
            f"[BASELINE+JOINT CHECK] {check_seconds:.1f} 秒内保持手套不接触物体，"
            "同时做一次张手→握拳（重点弯曲食指和中指）…"
        )
        time.sleep(check_seconds)
        with self.state_lock:
            angles, tactile = self.glove.drain()
            self.baseline_frames.extend(tactile)
            self.joint_preflight_frames.extend(self.glove.drain_skeleton())
            self.baseline_zone_frames.extend(self.glove.drain_zones())
            self.baseline_capturing = False
            self.joint_preflight_capturing = False
            frames = list(self.baseline_frames)
            preflight_frames = list(self.joint_preflight_frames)
            self.joint_preflight_frames = []
        frozen, ranges = _joint_preflight_quality(
            preflight_frames, self.joint_preflight_min_range_rad
        )
        if frozen:
            self.tactile_baseline = None
            self.tactile_zone_baseline = None
            details = ", ".join(
                f"{name}={np.rad2deg(ranges[name]):.2f}°" for name in frozen
            )
            print(
                "[BASELINE REFUSED] 关节活动自检失败："
                f"{details}，要求每路至少 {np.rad2deg(self.joint_preflight_min_range_rad):.1f}°。"
            )
            print("                   请保持无接触并完整张手/握拳，然后重试 b。")
            return
        if len(frames) < max(8, int(self.baseline_seconds * 40)):
            print(f"[BASELINE REFUSED] 触觉帧不足：{len(frames)}；检查 Wuji 后重试 b")
            return
        if len(self.baseline_zone_frames) < 8:
            print(
                f"[BASELINE REFUSED] Wuji tactile_zones 帧不足：{len(self.baseline_zone_frames)}；"
                "六区触觉不可用，检查 SDK 后重试 b"
            )
            return
        raw_source = np.stack([frame[2] for frame in frames]).astype(np.float32)
        # Check the device contract before applying our health mask. A healthy
        # firmware stream has 526 taxels; health-masked output naturally has
        # fewer and must not be rejected as a firmware mismatch.
        active_counts = np.asarray([tactile_active_taxel_count(frame) for frame in raw_source], dtype=np.int32)
        zone_totals = np.asarray([np.asarray(frame[3], dtype=np.int32).sum() for frame in self.baseline_zone_frames], dtype=np.int32)
        if not np.all(active_counts == WUJI_OFFICIAL_ACTIVE_TAXELS) or not np.all(zone_totals == WUJI_OFFICIAL_ACTIVE_TAXELS):
            observed_matrix = sorted(set(active_counts.tolist()))
            observed_zones = sorted(set(zone_totals.tolist()))
            self.tactile_baseline = None
            self.tactile_zone_baseline = None
            print(
                "[BASELINE REFUSED] Wuji tactile contract mismatch: "
                f"full-matrix active={observed_matrix}, zone total={observed_zones}, "
                f"official={WUJI_OFFICIAL_ACTIVE_TAXELS}. 当前 WG1K 固件/SDK 映射不一致。"
            )
            print("                  不会将 row 10 或任意 8 点插值/猜测屏蔽；请按吴极官方流程升级到与当前 SDK 匹配的固件后重试。")
            return
        raw = self.tactile_health_mask.apply_batch(raw_source) if self.tactile_health_mask is not None else raw_source
        valid = raw >= 0.0
        sums = np.where(valid, raw, 0.0).sum(axis=0)
        counts = valid.sum(axis=0)
        baseline = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        self.tactile_baseline = baseline.astype(np.float32)
        if self.baseline_zone_frames:
            self.tactile_zone_baseline = np.stack(
                [frame[2] for frame in self.baseline_zone_frames]
            ).astype(np.float32).mean(axis=0)
        else:
            self.tactile_zone_baseline = None
        self.tactile_baseline_count = len(frames)
        self.tactile_baseline_captured_ns = time.time_ns()
        self.joint_preflight_passed = True
        print(
            f"[BASELINE] 完成：{len(frames)} tactile frames, {len(self.baseline_zone_frames)} zone frames；"
            "之后每条 episode 将保存 baseline-subtracted 压力。"
        )
        print(
            "[JOINT CHECK] 通过：" + ", ".join(
                f"{name}={np.rad2deg(ranges[name]):.1f}°"
                for name in JOINT_PREFLIGHT_NAMES
            )
        )

    def _next_episode_path(self) -> Path:
        i = 0
        while any(
            (self.output_dir / name).exists()
            for name in (f"episode_{i:06d}.zarr", f".episode_{i:06d}.zarr.tmp")
        ):
            i += 1
        return self.output_dir / f"episode_{i:06d}.zarr"

    def select_episode_layout(self, participant_name: str, task_id: str, glove_condition: str) -> None:
        """Select and create one human-demo episode destination."""
        name = participant_name.strip()
        if not name or name.isdigit() or name in {".", ".."} or any(ch in name for ch in "/\r\n"):
            raise ValueError("人员姓名必须是非纯数字的安全单级目录名，不能包含 /、换行、. 或 ..")
        task_number = task_id[1:] if task_id.startswith("t") else ""
        if (
            not task_number.isdecimal()
            or int(task_number) < 1
            or task_number != str(int(task_number))
        ):
            raise ValueError("task 必须是 t 加正整数，例如 t1、t15")
        if glove_condition not in {"有手套", "无手套"}:
            raise ValueError("手套条件必须是 有手套 或 无手套")
        data_root = self.output_dir
        # Once selected, output_dir points at a leaf. The next selection must
        # still resolve from the original data root.
        if hasattr(self, "_episode_data_root"):
            data_root = self._episode_data_root
        else:
            self._episode_data_root = data_root
        # Task count is intentionally open-ended.  Create only the selected
        # leaf instead of baking a dataset-specific maximum into the code.
        for condition in ("有手套", "无手套"):
            (data_root / name / task_id / condition).mkdir(parents=True, exist_ok=True)
        self.output_dir = data_root / name / task_id / glove_condition
        self.participant_name = name
        self.task_id = task_id
        self.glove_condition = glove_condition
        # Keep the reader next to the episodes it indexes, so a copied
        # participant/task directory remains self-contained.
        self._install_reader()

    def confirm_protocol(self) -> None:
        """Optionally record an operator acknowledgement for collection notes."""
        print(
            "采集规范确认：1) 第二人负责开始/结束录制；2) 示范者双手只参与任务；"
            "3) 主相机仅覆盖工作区和对象；4) 画面无无关人脸、屏幕私密信息。"
        )
        if input("确认以上均满足，输入 YES 或 1（直接回车跳过）：").strip().upper() not in {"YES", "1"}:
            print("[PROTOCOL] 未记录确认；不阻塞采集。")
            return
        self.protocol_confirmed_ns = time.time_ns()
        print("[PROTOCOL] 已确认；可开始当前采集。")

    def start(self, instruction: str) -> bool:
        if self._save_thread is not None and self._save_thread.is_alive():
            print("[REFUSED] 上一个 episode 仍在后台保存，请等待保存完成后再开始")
            return False
        with self.state_lock:
            if self.recording:
                print("已经在采集中")
                return False
        if not self.glove.healthy():
            print("[REFUSED] Wuji 手套数据流不稳定；本次未开始录制，请检查网线/供电")
            print("          Wuji:", self.glove.status())
            return False
        if self.emg_enabled and self.emg is not None:
            stable_args = dict(
                duration_s=3.0,
                min_hz=self.min_emg_hz,
                max_age_s=min(self.max_emg_gap_ns / 1e9, self.emg.silence_timeout_s),
            )
            if not self.emg.stable(**stable_args):
                print(
                    f"[WAIT] 串口 EMG 正在连接/恢复；最多等待 {self.emg_start_wait_s:g} 秒，"
                    "恢复并连续稳定 3 秒后自动开始...",
                    flush=True,
                )
                deadline = time.monotonic() + self.emg_start_wait_s
                next_status = time.monotonic() + 5.0
                while time.monotonic() < deadline and not self.emg.stable(**stable_args):
                    time.sleep(0.1)
                    if time.monotonic() >= next_status:
                        print("       EMG:", self.emg.status(), flush=True)
                        next_status += 5.0
                if not self.emg.stable(**stable_args):
                    print("[REFUSED] 串口 EMG 在等待窗口内仍未连接并连续稳定出流；后台会继续自动重连")
                    print("          EMG:", self.emg.status())
                    print("          最近事件:", self.emg.diagnostic_tail())
                    return False
                print("[PASS] 串口 EMG 已自动恢复并连续稳定出流 3 秒")
        if self.myo_enabled and self.myo is not None:
            myo_stable_args = dict(
                duration_s=3.0,
                min_hz=self.min_myo_hz,
                max_age_s=min(self.max_myo_gap_ns / 1e9, self.myo.silence_timeout_s),
            )
            if not self.myo.stable(**myo_stable_args):
                print(
                    f"[WAIT] Myo 正在连接/恢复；最多等待 {self.myo_start_wait_s:g} 秒，"
                    "恢复并连续稳定 3 秒后自动开始...",
                    flush=True,
                )
                deadline = time.monotonic() + self.myo_start_wait_s
                while time.monotonic() < deadline and not self.myo.stable(**myo_stable_args):
                    time.sleep(0.1)
                if not self.myo.stable(**myo_stable_args):
                    print("[REFUSED] Myo 在等待窗口内仍未连续稳定出流")
                    print("          Myo:", self.myo.status())
                    return False
                print("[PASS] Myo 已连续稳定出流 3 秒")
        if not self.camera.ready():
            needed = (
                f"{self.camera.ego_source} 第一人称 RGB"
                + ("/内参" if self.camera.ego_require_camera_info else "")
                if self.camera.capture_ego else "Gemini 第三人称 RGB"
            )
            if self.camera.gemini_enabled:
                needed += " 和 Gemini 第三人称 RGB"
            if self.camera.gemini_imu_enabled:
                needed += " 和 Gemini IMU"
            if self.camera.trackers_enabled:
                needed += " 和两个 PICO Tracker"
            print(f"[REFUSED] 尚未收到 {needed}；检查 ROS 相机驱动。")
            return False
        if self.tactile_baseline is None:
            print("[REFUSED] 先输入 b 完成无接触触觉基线和关节活动自检")
            return False
        if not self.joint_preflight_passed:
            print("[REFUSED] 最近一次 b 未通过关节活动自检；请重新输入 b")
            return False
        instruction = " ".join(instruction.split())
        if len(instruction) < 4 or instruction.lower() in {"a", "test", "demo", "none", "null"}:
            print("[REFUSED] 请输入真实、具体的任务描述（至少 4 个字符；不能用 a/test 占位）")
            return False
        # Start glove collection first.  The canonical RGB stream starts last,
        # so its first timestamp is already covered by every glove source.
        with self.state_lock:
            self.glove.clear()
            self.angle_frames = []
            self.tactile_frames = []
            self.skeleton_frames = []
            self.zone_frames = []
            self.right_wrist_pose_frames = []
            if self.emg is not None:
                self.emg.clear()
            myo = getattr(self, "myo", None)
            if myo is not None:
                myo.clear()
            self.emg_frames = []
            self.emg_imu_frames = []
            self.myo_frames = []
            self.emg_last_episode_ts_ns = 0
            self.emg_episode_connection_id = self.emg.current_connection_id() if self.emg is not None else 0
            self.emg_episode_interruption_id = self.emg.interruption_snapshot()[0] if self.emg is not None else 0
            self.myo_last_episode_ts_ns = 0
            self.myo_episode_connection_id = self.myo.current_connection_id() if self.myo is not None else 0
            self.myo_episode_interruption_id = self.myo.interruption_snapshot()[0] if self.myo is not None else 0
            self.myo_invalid_reason = ""
            self.emg_invalid_reason = ""
            self.emg_warning_reason = ""
            self.emg_alignment_warning_count = 0
            self.emg_alignment_warning_max_gap_ns = 0
            self.emg_next_clock_warning_mono = 0.0
            self.emg_clock_violation_started_mono = 0.0
            self.live_last_canonical_joints = None
            self.live_alignment_violation_started = {}
            self.live_next_alignment_check_mono = 0.0
            self.instruction = instruction
            self.last_refusal_reason = ""
            self.started_ns = time.time_ns()
            self.recording = True
            self.episode_started_mono = time.perf_counter()
        try:
            self.camera.start_episode()
        except Exception:
            # Do not leave the drain thread appending into an episode that
            # never received its canonical camera stream.
            with self.state_lock:
                self.recording = False
                self.glove.clear()
                self.angle_frames = []
                self.tactile_frames = []
                self.skeleton_frames = []
                self.zone_frames = []
                self.right_wrist_pose_frames = []
                if self.emg is not None:
                    self.emg.clear()
                if self.myo is not None:
                    self.myo.clear()
                self.emg_frames = []
                self.emg_imu_frames = []
                self.myo_frames = []
            raise
        print("[REC] FTP-1 episode 开始：", instruction)
        return True

    def save_async(self) -> None:
        if self._save_thread is not None and self._save_thread.is_alive():
            print("[SAVE] 上一个 episode 仍在后台保存，请稍候")
            return
        self._save_thread = threading.Thread(target=self.save, name="ftp1-save", daemon=True)
        self._save_thread.start()

    def save(self) -> None:
        with self.state_lock:
            if not self.recording:
                if self.last_refusal_reason:
                    print("当前 episode 已丢弃；数据流就绪后按 s 重新采集。")
                else:
                    print("当前没有 episode")
                return
        save_started = time.perf_counter()
        episode_started_mono = getattr(self, "episode_started_mono", save_started)
        capture_seconds = save_started - episode_started_mono
        # Freeze the canonical glove/EMG boundary before stopping the ROS
        # sources. The camera methods only detach captured messages here;
        # bounded JPEG decoding happens after the timestamp quality gates.
        # Leaving self.recording true during that work would extend the 120 Hz
        # human-mode axis beyond the already stopped Tracker buffers.
        with self.state_lock:
            if not self.recording:
                return
            emg_fault = self._emg_live_fault()
            if not emg_fault:
                emg_fault = self._myo_live_fault()
            if not emg_fault:
                self.recording = False
                angles, tactile = self.glove.drain()
                skeleton = self.glove.drain_skeleton()
                zones = self.glove.drain_zones()
                wrist_poses = self.glove.drain_right_wrist_pose()
                emg = self.emg.drain() if self.emg is not None else []
                emg_imu = getattr(self.emg, "drain_imu", lambda: [])() if self.emg is not None else []
                myo = self.myo.drain() if self.myo is not None else []
                self.angle_frames.extend(angles)
                self.tactile_frames.extend(tactile)
                self.skeleton_frames.extend(skeleton)
                self.zone_frames.extend(zones)
                self.right_wrist_pose_frames.extend(wrist_poses)
                self.emg_frames.extend(emg)
                if not hasattr(self, "emg_imu_frames"):
                    self.emg_imu_frames = []
                self.emg_imu_frames.extend(emg_imu)
                self.myo_frames.extend(myo)
        if emg_fault:
            self.refuse(emg_fault, automatic=True)
            return
        # The ROS boundary now lands just after the canonical boundary, so all
        # external streams cover the final canonical sample. Keep compressed
        # messages encoded until the quality gates have selected the episode;
        # the image payloads are decoded into bounded disk-backed stores below.
        color, _depth, color_info, _depth_info = self.camera.finish_episode(decode=False)
        # FTP-1 is RGB-only; do not retain optional depth message payloads
        # while decoding the two RGB streams.
        del _depth, _depth_info
        gemini = self.camera.finish_gemini_episode(decode=False)
        camera_poses = self.camera.finish_gemini_pose_episode()
        tracker_camera, tracker_wrist = self.camera.finish_tracker_episode()
        missing: list[str] = []
        if not self.angle_frames:
            missing.append("手套角度")
        if not self.skeleton_frames:
            missing.append("手套骨架")
        if not self.tactile_frames:
            missing.append("全手触觉")
        if not self.zone_frames:
            missing.append("六区触觉")
        if not self.right_wrist_pose_frames:
            missing.append("waist→r_wrist TF")
        if self.emg_enabled and not self.emg_frames:
            missing.append("串口 EMG")
        if self.myo_enabled and not self.myo_frames:
            missing.append("Myo RAW EMG")
        if self.camera.capture_ego and not color:
            missing.append(f"{self.camera.ego_source} RGB")
        if self.camera.gemini_enabled and not gemini:
            missing.append("Gemini RGB")
        if self.camera.gemini_imu_enabled and not camera_poses:
            missing.append("Gemini IMU")
        if self.camera.trackers_enabled:
            if not tracker_camera:
                missing.append("PICO camera Tracker")
            if not tracker_wrist:
                missing.append("PICO wrist Tracker")
        if missing:
            counts = (
                f"angles={len(self.angle_frames)}, skeleton={len(self.skeleton_frames)}, "
                f"tactile={len(self.tactile_frames)}, zones={len(self.zone_frames)}, "
                f"wrist_tf={len(self.right_wrist_pose_frames)}, emg={len(self.emg_frames)}, "
                f"ego_rgb={len(color)}, gemini_rgb={len(gemini)}, gemini_imu={len(camera_poses)}, "
                f"tracker=({len(tracker_camera)},{len(tracker_wrist)})"
            )
            print(f"[SAVE REFUSED] 缺少数据源：{', '.join(missing)}")
            print(f"              当前帧数：{counts}")
            return

        # Exclude camera frames that were already in a ROS callback queue when
        # ``s`` was pressed.  They are episode-boundary data, not a valid first
        # sample for this demonstration.
        color = [frame for frame in color if frame[0] >= self.started_ns]
        gemini = [frame for frame in gemini if frame[0] >= self.started_ns]
        camera_poses = [frame for frame in camera_poses if frame[0] >= self.started_ns]
        tracker_camera = [frame for frame in tracker_camera if frame[0] >= self.started_ns]
        tracker_wrist = [frame for frame in tracker_wrist if frame[0] >= self.started_ns]
        boundary_missing: list[str] = []
        if self.camera.capture_ego and len(color) < 2:
            boundary_missing.append(f"{self.camera.ego_source} RGB")
        if self.camera.gemini_enabled and len(gemini) < 2:
            boundary_missing.append("Gemini RGB")
        if self.camera.gemini_imu_enabled and len(camera_poses) < 2:
            boundary_missing.append("Gemini IMU")
        if self.camera.trackers_enabled:
            if len(tracker_camera) < 2:
                boundary_missing.append("PICO camera Tracker")
            if len(tracker_wrist) < 2:
                boundary_missing.append("PICO wrist Tracker")
        if boundary_missing:
            print(f"[SAVE REFUSED] episode 开始后来源帧不足：{', '.join(boundary_missing)}")
            print(f"              episode 边界后帧数：ego_rgb={len(color)}, gemini_rgb={len(gemini)}, gemini_imu={len(camera_poses)}, tracker=({len(tracker_camera)},{len(tracker_wrist)})")
            return
        # The configured first-person camera is the canonical axis.  If it is
        # disabled, use the glove skeleton clock;
        # no image is synthesized or written for the missing ego stream.
        if self.camera.capture_ego:
            raw_color_ts_ns = np.asarray([frame[0] for frame in color], dtype=np.int64)
        else:
            raw_color_ts_ns = np.asarray([frame[0] for frame in self.skeleton_frames], dtype=np.int64) * 1_000
            if len(raw_color_ts_ns) < 2:
                print("[SAVE REFUSED] episode 边界后手套时间戳不足；请重录")
                return
        raw_timeline = _timeline_quality(raw_color_ts_ns) if not self.fast_save else {"max_gap_ns": 0, "missing_slot_ratio": 0.0, "missing_slot_count": 0}
        if (not self.fast_save) and self.camera.capture_ego and (raw_timeline["max_gap_ns"] > self.max_rgb_gap_ns or raw_timeline["missing_slot_ratio"] > self.max_missing_ratio):
            print(
                "[SAVE REFUSED] 第一人称 RGB 源帧不连续："
                f"max-gap={raw_timeline['max_gap_ns'] / 1e6:.2f} ms, "
                f"missing={raw_timeline['missing_slot_ratio']:.2%}；请重录"
            )
            return
        # Use every actually captured first-person RGB frame as the canonical
        # axis.  Small source dropouts remain visible as timestamp gaps instead
        # of fabricating a 60 Hz slot that has no real camera image.
        #
        # This preserves real images + real timestamps and never duplicates or
        # interpolates RGB frames.
        canonical_period_ns = int(np.median(np.diff(raw_color_ts_ns)))
        color_ts_ns = raw_color_ts_ns.copy()
        target_us = color_ts_ns // 1_000
        ego_idx = np.arange(len(raw_color_ts_ns), dtype=np.intp)
        ego_age_us = np.zeros(len(raw_color_ts_ns), dtype=np.int64)
        angle_ts_us = np.asarray([frame[0] for frame in self.angle_frames], dtype=np.int64)
        tactile_ts_us = np.asarray([frame[0] for frame in self.tactile_frames], dtype=np.int64)
        skeleton_ts_us = np.asarray([frame[0] for frame in self.skeleton_frames], dtype=np.int64)
        zone_ts_us = np.asarray([frame[0] for frame in self.zone_frames], dtype=np.int64)
        wrist_pose_ts_us = np.asarray([frame[0] for frame in self.right_wrist_pose_frames], dtype=np.int64)
        angle_idx, angle_age_us = _nearest_indices(angle_ts_us, target_us)
        tactile_idx, tactile_age_us = _nearest_indices(tactile_ts_us, target_us)
        skeleton_idx, skeleton_age_us = _nearest_indices(skeleton_ts_us, target_us)
        zone_idx, zone_age_us = _nearest_indices(zone_ts_us, target_us)
        wrist_pose_idx, wrist_pose_age_us = _nearest_indices(wrist_pose_ts_us, target_us)
        emg_frames: list[tuple[int, int, int, np.ndarray, int]] = []
        emg_ts_ns = np.empty(0, dtype=np.int64)
        emg_clock_raw_ts_ns = np.empty(0, dtype=np.int64)
        emg_arrival_ts_ns = np.empty(0, dtype=np.int64)
        emg_clock_offset_ns = 0
        emg_idx = emg_age_us = None
        emg_hz = 0.0
        emg_missing = 0
        emg_missing_ratio = 0.0
        emg_max_gap_ns = 0
        if self.emg_enabled:
            emg_ts_ns_all, _ = _emg_corrected_timestamps_ns(self.emg_frames)
            # Keep native EMG within the actual RGB interval plus one alignment
            # tolerance. It remains lossless under ``streams/`` below.
            emg_keep = (
                (emg_ts_ns_all >= color_ts_ns[0] - self.max_emg_age_us * 1_000)
                & (emg_ts_ns_all <= color_ts_ns[-1] + self.max_emg_age_us * 1_000)
            )
            (
                emg_frames,
                emg_ts_ns,
                emg_clock_raw_ts_ns,
                emg_arrival_ts_ns,
                emg_clock_offset_ns,
            ) = _emg_episode_clock_arrays(self.emg_frames, emg_keep)
            if len(emg_frames) < 2:
                print("[SAVE REFUSED] RGB 时间区间内串口 EMG 样本不足；检查接收器/臂带后重录")
                return
            emg_max_gap_ns = int(np.diff(emg_ts_ns).max()) if len(emg_ts_ns) >= 2 else 0
            emg_idx, emg_age_ns = _nearest_indices(emg_ts_ns, color_ts_ns)
            emg_age_us = (emg_age_ns // 1_000).astype(np.int64)
        myo_frames: list[tuple[int, int, int, np.ndarray, int]] = []
        myo_ts_ns = np.empty(0, dtype=np.int64)
        myo_clock_raw_ts_ns = np.empty(0, dtype=np.int64)
        myo_arrival_ts_ns = np.empty(0, dtype=np.int64)
        myo_clock_offset_ns = 0
        myo_idx = myo_age_us = None
        myo_hz = 0.0
        myo_missing = 0
        myo_missing_ratio = 0.0
        myo_max_gap_ns = 0
        if self.myo_enabled:
            myo_ts_ns_all, _ = _emg_corrected_timestamps_ns(self.myo_frames)
            myo_keep = (
                (myo_ts_ns_all >= color_ts_ns[0] - self.max_myo_age_us * 1_000)
                & (myo_ts_ns_all <= color_ts_ns[-1] + self.max_myo_age_us * 1_000)
            )
            (
                myo_frames,
                myo_ts_ns,
                myo_clock_raw_ts_ns,
                myo_arrival_ts_ns,
                myo_clock_offset_ns,
            ) = _emg_episode_clock_arrays(self.myo_frames, myo_keep)
            if len(myo_frames) < 2:
                print("[SAVE REFUSED] RGB 时间区间内 Myo EMG 样本不足；检查 BLED112/Myo 后重录")
                return
            myo_max_gap_ns = int(np.diff(myo_ts_ns).max())
            myo_idx, myo_age_ns = _nearest_indices(myo_ts_ns, color_ts_ns)
            myo_age_us = (myo_age_ns // 1_000).astype(np.int64)
        # Gemini 2's selected RGB profile is independent from the canonical
        # first-person timeline. Do not repeat a
        # 30 Hz source image just to create a T-major second camera tensor;
        # preserve its native frames/timestamps under streams/ below instead.
        gemini_ts_ns = np.asarray([frame[0] for frame in gemini], dtype=np.int64) if gemini else np.empty(0, dtype=np.int64)
        gemini_timeline = _timeline_quality(gemini_ts_ns)
        if len(gemini_ts_ns) >= 2 and np.any(np.diff(gemini_ts_ns) <= 0):
            print("[SAVE REFUSED] Gemini RGB 源时间戳无效或重复；请重录")
            return
        camera_pose_ts_ns = np.asarray([frame[0] for frame in camera_poses], dtype=np.int64) if camera_poses else np.empty(0, dtype=np.int64)
        if len(camera_pose_ts_ns):
            camera_pose_idx, camera_pose_age_us = _nearest_indices(camera_pose_ts_ns // 1_000, target_us)
        else:
            camera_pose_idx, camera_pose_age_us = None, None
        tracker_camera_ts_ns = np.asarray([frame[0] for frame in tracker_camera], dtype=np.int64) if tracker_camera else np.empty(0, dtype=np.int64)
        tracker_wrist_ts_ns = np.asarray([frame[0] for frame in tracker_wrist], dtype=np.int64) if tracker_wrist else np.empty(0, dtype=np.int64)
        tracker_camera_xyz_quality = _tracker_xyz_quality(tracker_camera)
        tracker_wrist_xyz_quality = _tracker_xyz_quality(tracker_wrist)
        if self.camera.trackers_enabled:
            if np.any(np.diff(tracker_camera_ts_ns) <= 0) or np.any(np.diff(tracker_wrist_ts_ns) <= 0):
                print("[SAVE REFUSED] PICO Tracker 时间戳无效或重复；请检查 tracker publisher 后重录")
                return
            # A fixed camera-mounted Tracker is expected to remain still and
            # is never rejected for identical XYZ. The wrist Tracker is
            # different: when it traverses at least 2 cm, fewer than 10 real
            # value updates/s indicates cached poses with fresh message stamps.
            if (
                tracker_wrist_xyz_quality["xyz_span_m"] >= 0.02
                and tracker_wrist_xyz_quality["xyz_update_hz"] < 10.0
            ):
                print(
                    "[SAVE REFUSED] PICO wrist Tracker XYZ 非连续更新："
                    f"span={tracker_wrist_xyz_quality['xyz_span_m'] * 100:.1f} cm, "
                    f"updates={tracker_wrist_xyz_quality['xyz_update_count']} "
                    f"({tracker_wrist_xyz_quality['xyz_update_hz']:.1f} Hz), "
                    f"same={tracker_wrist_xyz_quality['adjacent_same_ratio']:.1%}, "
                    f"max-step={tracker_wrist_xyz_quality['xyz_max_step_m'] * 100:.1f} cm；"
                    "camera Tracker 静止不参与此判定，请检查 wrist Tracker/PC-Service 后重录"
                )
                return
            # Tracker poses are numeric state, not images. Align them with the
            # same nearest-real-sample rule used for glove state, wrist TF,
            # EMG and Gemini IMU. A roughly 57 Hz Tracker stream aligned to a
            # roughly 60 Hz RGB axis will occasionally reference one real pose
            # from two adjacent rows; the native lossless stream and source
            # seq/timestamp audit arrays below make that explicit. Requiring a
            # unique source for every row incorrectly invalidates the entire
            # episode whenever len(source) < len(target).
            tracker_camera_idx, tracker_camera_age_us = _nearest_indices(tracker_camera_ts_ns // 1_000, target_us)
            tracker_wrist_idx, tracker_wrist_age_us = _nearest_indices(tracker_wrist_ts_ns // 1_000, target_us)
        else:
            tracker_camera_idx = tracker_wrist_idx = tracker_camera_age_us = tracker_wrist_age_us = None

        # Never extrapolate an endpoint or duplicate a source image just to
        # make the Zarr look uniform.  Invalid internal slots reject the whole
        # episode; a stale suffix is handled below by shortening the episode.
        ego_ok = (
            (np.abs(ego_age_us) <= canonical_period_ns // 2_000)
            & _unique_source_mask(ego_idx, ego_age_us)
        ) if self.camera.capture_ego else np.ones(len(color_ts_ns), dtype=bool)
        angle_ok = np.abs(angle_age_us) <= self.max_glove_age_us
        tactile_ok = np.abs(tactile_age_us) <= self.max_glove_age_us
        skeleton_ok = np.abs(skeleton_age_us) <= self.max_glove_age_us
        zone_ok = np.abs(zone_age_us) <= self.max_glove_age_us
        wrist_pose_ok = np.abs(wrist_pose_age_us) <= self.max_wrist_pose_age_us
        emg_ok = np.ones(len(color_ts_ns), dtype=bool) if not self.emg_enabled else np.abs(emg_age_us) <= self.max_emg_age_us
        myo_ok = np.ones(len(color_ts_ns), dtype=bool) if not self.myo_enabled else np.abs(myo_age_us) <= self.max_myo_age_us
        tracker_camera_ok = (
            None if tracker_camera_idx is None
            else np.abs(tracker_camera_age_us) <= self.max_tracker_age_us
        )
        tracker_wrist_ok = (
            None if tracker_wrist_idx is None
            else np.abs(tracker_wrist_age_us) <= self.max_tracker_age_us
        )

        valid = (
            ego_ok
            & angle_ok
            & tactile_ok
            & skeleton_ok
            & zone_ok
            & wrist_pose_ok
            & emg_ok
            & myo_ok
        )
        non_emg_valid = (
            ego_ok
            & angle_ok
            & tactile_ok
            & skeleton_ok
            & zone_ok
            & wrist_pose_ok
        )

        camera_pose_ok = None
        if camera_pose_idx is not None:
            camera_pose_ok = np.abs(camera_pose_age_us) <= self.max_camera_pose_age_us
            valid &= camera_pose_ok
            non_emg_valid &= camera_pose_ok
        if tracker_camera_ok is not None:
            valid &= tracker_camera_ok
            non_emg_valid &= tracker_camera_ok
        if tracker_wrist_ok is not None:
            valid &= tracker_wrist_ok
            non_emg_valid &= tracker_wrist_ok

        raw_t = len(raw_color_ts_ns)
        alignment_dropped = int(np.count_nonzero(~valid))
        alignment_drop_ratio = alignment_dropped / max(len(color_ts_ns), 1)
        alignment_max_drop_run = _max_false_run(valid)
        emg_only_degradation = False

        if not np.all(valid):
            # Keep the age gate strict instead of accepting stale numeric
            # state. A few sparse invalid rows may be removed transparently:
            # retained RGB frames are all real, no state is interpolated, and
            # the exact drop count/run are stored in metadata. A larger ratio
            # or a run beyond the source-specific recovery budget remains
            # fatal because it indicates a meaningful source outage.
            invalid = np.flatnonzero(~valid)
            max_drop_run = 3
            if self.emg_enabled and emg_idx is not None:
                alignment_emg_seq = np.asarray([frame[1] for frame in emg_frames], dtype=np.int64)
                alignment_emg_missing = (
                    int(np.maximum(np.diff(alignment_emg_seq) - 1, 0).sum())
                    if len(alignment_emg_seq) >= 2 else 0
                )
                alignment_emg_missing_ratio = alignment_emg_missing / max(
                    len(alignment_emg_seq) + alignment_emg_missing, 1
                )
                emg_only_degradation = bool(
                    np.all(non_emg_valid)
                    and np.all(myo_ok)
                    and _rate(emg_ts_ns) >= self.min_emg_hz
                    and alignment_emg_missing_ratio <= self.max_emg_missing_ratio
                    and emg_max_gap_ns <= self.max_emg_gap_ns
                )
                if emg_only_degradation:
                    # A healthy high-rate native EMG stream may contain one
                    # short serial delivery hole. Keep the episode when
                    # the configured sparse-row ratio permits it, but remove
                    # every affected canonical row instead of repeating stale
                    # EMG. The lossless native stream remains in streams/.
                    max_drop_run = _emg_alignment_drop_run_limit(
                        canonical_period_ns,
                        self.max_emg_gap_ns,
                        native_quality_ok=True,
                    )
            allowed_drop_ratio = 1.0 if emg_only_degradation else self.max_align_drop_ratio
            wuji_only_degradation = _only_wuji_rows_invalid(
                (angle_ok, tactile_ok, skeleton_ok, zone_ok, wrist_pose_ok),
                (
                    ego_ok,
                    emg_ok if self.emg_enabled else None,
                    myo_ok if self.myo_enabled else None,
                    camera_pose_ok,
                    tracker_camera_ok,
                    tracker_wrist_ok,
                ),
            )
            if wuji_only_degradation:
                # All Wuji streams share one SDK transport and can pause
                # together during a short scheduling/session disturbance.
                # Remove uncovered canonical rows after recovery instead of
                # discarding an otherwise intact episode.
                max_drop_run = max(
                    max_drop_run,
                    int(np.ceil(self.max_wuji_gap_ns / max(canonical_period_ns, 1))),
                )
            # Tracker is an auxiliary pose source. Sparse delayed samples are
            # safer to remove than to reject an otherwise healthy episode;
            # the complete native tracker stream remains in ``streams`` and
            # the live watchdog still aborts a sustained tracker outage.
            tracker_only_invalid = _only_tracker_rows_invalid(
                tracker_camera_ok,
                tracker_wrist_ok,
                (
                    # Do not use ``non_emg_valid`` here: it already includes
                    # both Tracker masks, which made this branch impossible
                    # precisely when a Tracker row was stale.
                    ego_ok,
                    angle_ok,
                    tactile_ok,
                    skeleton_ok,
                    zone_ok,
                    wrist_pose_ok,
                    emg_ok if self.emg_enabled else None,
                    myo_ok if self.myo_enabled else None,
                    camera_pose_ok,
                ),
            )
            if tracker_only_invalid:
                max_drop_run = max(max_drop_run, 12)
            if _can_drop_sparse_invalid_rows(
                valid,
                allowed_drop_ratio,
                max_drop_run=max_drop_run,
            ):
                dropped_sources = [
                    name
                    for name, ok in (
                        ("ego", ego_ok),
                        ("angle", angle_ok),
                        ("tactile", tactile_ok),
                        ("skeleton", skeleton_ok),
                        ("zone", zone_ok),
                        ("wrist_tf", wrist_pose_ok),
                        ("EMG", emg_ok),
                        ("Myo", myo_ok),
                        ("GeminiIMU", camera_pose_ok),
                        ("tracker_cam", tracker_camera_ok),
                        ("tracker_wrist", tracker_wrist_ok),
                    )
                    if ok is not None and bool(np.any(~ok[invalid]))
                ]
                print(
                    f"[ALIGN] 丢弃 {alignment_dropped} 个 canonical RGB slot "
                    f"(最长连续 {alignment_max_drop_run})："
                    f"{', '.join(dropped_sources) or 'source'} 没有满足 age 门槛的真实近邻样本"
                    + ("；native EMG 频率/sequence/最大间断均合格，本轮继续保存" if emg_only_degradation else "")
                    + ("；Wuji 已从短暂停顿恢复，本轮继续保存" if wuji_only_degradation else "")
                )

                keep = valid.copy()

                def select(values):
                    return values[keep]

                color_ts_ns = select(color_ts_ns)
                target_us = select(target_us)
                ego_idx, ego_age_us = select(ego_idx), select(ego_age_us)
                angle_idx, angle_age_us = select(angle_idx), select(angle_age_us)
                tactile_idx, tactile_age_us = select(tactile_idx), select(tactile_age_us)
                skeleton_idx, skeleton_age_us = select(skeleton_idx), select(skeleton_age_us)
                zone_idx, zone_age_us = select(zone_idx), select(zone_age_us)
                wrist_pose_idx, wrist_pose_age_us = select(wrist_pose_idx), select(wrist_pose_age_us)
                if emg_idx is not None:
                    emg_idx, emg_age_us = select(emg_idx), select(emg_age_us)
                if myo_idx is not None:
                    myo_idx, myo_age_us = select(myo_idx), select(myo_age_us)
                if camera_pose_idx is not None:
                    camera_pose_idx, camera_pose_age_us = select(camera_pose_idx), select(camera_pose_age_us)
                if tracker_camera_idx is not None:
                    tracker_camera_idx, tracker_camera_age_us = select(tracker_camera_idx), select(tracker_camera_age_us)
                    tracker_camera_ok = select(tracker_camera_ok)
                if tracker_wrist_idx is not None:
                    tracker_wrist_idx, tracker_wrist_age_us = select(tracker_wrist_idx), select(tracker_wrist_age_us)
                    tracker_wrist_ok = select(tracker_wrist_ok)
                valid = select(valid)
            else:
                def report(name, ok, ages, hz, limit_us):
                    bad = int(np.count_nonzero(~ok))
                    max_age = float(np.abs(ages).max(initial=0)) / 1000.0
                    p95_age = float(np.percentile(np.abs(ages), 95)) / 1000.0 if len(ages) else 0.0
                    print(
                        f"    {name:10s}: bad={bad:3d}/{len(ok)}, "
                        f"source={hz:6.1f} Hz, "
                        f"age p95={p95_age:6.2f} ms, max={max_age:6.2f} ms, "
                        f"limit={limit_us / 1000.0:5.2f} ms"
                    )

                print(
                    "[SAVE REFUSED] 统一时间轴存在超龄/无效 source："
                    f"{alignment_dropped}/{len(color_ts_ns)} slots"
                )
                if self.camera.capture_ego:
                    report("ego", ego_ok, ego_age_us, _rate(raw_color_ts_ns), canonical_period_ns // 2_000)
                report("angle", angle_ok, angle_age_us, _rate(angle_ts_us * 1_000), self.max_glove_age_us)
                report("tactile", tactile_ok, tactile_age_us, _rate(tactile_ts_us * 1_000), self.max_glove_age_us)
                report("skeleton", skeleton_ok, skeleton_age_us, _rate(skeleton_ts_us * 1_000), self.max_glove_age_us)
                report("zone", zone_ok, zone_age_us, _rate(zone_ts_us * 1_000), self.max_glove_age_us)
                report("wrist_tf", wrist_pose_ok, wrist_pose_age_us, _rate(wrist_pose_ts_us * 1_000), self.max_wrist_pose_age_us)
                if self.emg_enabled:
                    assert emg_age_us is not None
                    report("Wavletech", emg_ok, emg_age_us, _rate(emg_ts_ns), self.max_emg_age_us)
                if self.myo_enabled:
                    assert myo_age_us is not None
                    report("Myo", myo_ok, myo_age_us, _rate(myo_ts_ns), self.max_myo_age_us)
                if camera_pose_ok is not None:
                    report("GeminiIMU", camera_pose_ok, camera_pose_age_us, _rate(camera_pose_ts_ns), self.max_camera_pose_age_us)
                if tracker_camera_ok is not None:
                    report("tracker_cam", tracker_camera_ok, tracker_camera_age_us, _rate(tracker_camera_ts_ns), self.max_tracker_age_us)
                if tracker_wrist_ok is not None:
                    report("tracker_wrist", tracker_wrist_ok, tracker_wrist_age_us, _rate(tracker_wrist_ts_ns), self.max_tracker_age_us)
                print("    非图像状态仅对齐到最近真实样本，不插值、不伪造图像；请根据上面具体 source 修复。")
                return
        timeline = _timeline_quality(color_ts_ns)
        # Raw camera continuity was already checked before alignment.  If the
        # sparse-row gate explicitly approved N consecutive source-age drops,
        # the retained RGB axis necessarily gains roughly N camera periods at
        # that location.  Account for those approved removals here instead of
        # contradicting the earlier gate at the 3 x 16.67 ms = 50.01 ms
        # floating/timestamp boundary.  Missing ratio remains independently
        # bounded, and a real raw-camera outage still uses max_rgb_gap_ns.
        final_max_gap_ns = max(
            self.max_rgb_gap_ns,
            int(raw_timeline["max_gap_ns"]) + alignment_max_drop_run * canonical_period_ns,
        )
        final_missing_limit = max(self.max_missing_ratio, self.max_align_drop_ratio)
        if emg_only_degradation:
            # These holes were already bounded by native EMG rate, source
            # sequence quality and max_emg_gap_ns. Preserve only rows with a
            # real <=max_emg_age sample without applying the unrelated generic
            # canonical drop-ratio budget a second time.
            final_missing_limit = max(final_missing_limit, alignment_drop_ratio)
        if (
            timeline["max_gap_ns"] > final_max_gap_ns
            or timeline["missing_slot_ratio"] > final_missing_limit + 1e-12
        ):
            print(
                "[SAVE REFUSED] 对齐行丢弃后的最终时间轴不连续："
                f"max-gap={timeline['max_gap_ns'] / 1e6:.2f} ms, "
                f"allowed-gap={final_max_gap_ns / 1e6:.2f} ms, "
                f"missing={timeline['missing_slot_ratio']:.2%}；请重录"
            )
            return

        # A large alignment age normally means the ego-camera and glove clocks do not
        # share an epoch or a source stalled.  Do not create a superficially
        # valid FTP-1 file in that case.
        max_age_us = max(
            int(np.abs(angle_age_us).max(initial=0)),
            int(np.abs(tactile_age_us).max(initial=0)),
            int(np.abs(skeleton_age_us).max(initial=0)),
            int(np.abs(zone_age_us).max(initial=0)),
            int(np.abs(wrist_pose_age_us).max(initial=0)),
            int(np.abs(emg_age_us).max(initial=0)) if self.emg_enabled and emg_age_us is not None else 0,
            int(np.abs(myo_age_us).max(initial=0)) if self.myo_enabled and myo_age_us is not None else 0,
            int(np.abs(camera_pose_age_us).max(initial=0)) if camera_pose_age_us is not None else 0,
            int(np.abs(tracker_camera_age_us).max(initial=0)) if tracker_camera_age_us is not None else 0,
            int(np.abs(tracker_wrist_age_us).max(initial=0)) if tracker_wrist_age_us is not None else 0,
        )
        if max_age_us > 100_000:
            print(
                "[SAVE REFUSED] RGB 与手套时间轴相差超过 100 ms "
                f"(max={max_age_us / 1000.0:.1f} ms)；请先检查系统时间与 source 状态。"
            )
            return

        ego_store = (
            _decode_messages_to_memmap(
                color,
                message_index=1,
                compressed=self.camera.ego_compressed,
            )
            if self.camera.capture_ego
            else None
        )
        if ego_store is not None:
            # The normal canonical axis uses every retained source frame, so
            # this is a zero-copy mmap view. Sparse alignment drops require a
            # compact copy only for the retained rows.
            aligned_color = (
                ego_store.array
                if _is_full_identity_selection(ego_idx, len(ego_store.array))
                else np.asarray(ego_store.array[ego_idx], dtype=np.uint8)
            )
        else:
            aligned_color = None
        # No later quality or audit path needs the ROS color messages; the
        # timestamp array and mmap now carry the retained source frames.
        del color
        sharpness = {"min": 0.0, "p05": 0.0, "median": 0.0} if self.fast_save else (_sharpness_stats(aligned_color) if aligned_color is not None else {"min": 0.0, "p05": 0.0, "median": 0.0})
        if (not self.fast_save) and self.camera.capture_ego and sharpness["p05"] < self.min_ego_sharpness:
            print(
                "[QUALITY WARN] 第一人称 RGB 低纹理/运动模糊："
                f"sharpness-p05={sharpness['p05']:.1f} < {self.min_ego_sharpness:.1f}；"
                "该指标无法区分白墙/桌面等低纹理画面，本 episode 继续保存并记录警告"
            )
        aligned_angles_raw = np.stack([self.angle_frames[i][2] for i in angle_idx]).astype(np.float32)
        if aligned_angles_raw.shape[1:] != (5, 5):
            raise RuntimeError(f"Unexpected Wuji angle shape {aligned_angles_raw.shape}; expected (T, 5, 5)")
        # Preserve Wuji's anatomical state.  The official FTP-1 human state is
        # instead calculated from the same glove's 21 MediaPipe landmarks.
        aligned_angles = aligned_angles_raw.reshape(len(color_ts_ns), 25)[:, WUJI_21_FROM_25]
        aligned_skeleton = np.stack([self.skeleton_frames[i][2] for i in skeleton_idx]).astype(np.float32)
        published_joints = ftp1_right_hand_joints_official_from_mediapipe(aligned_skeleton)
        stable_joints = ftp1_right_hand_joints_from_mediapipe(aligned_skeleton)
        canonical_joints = _canonical_with_stable_adduction(published_joints, stable_joints)
        # Known WG1K firmware limits are exact half-degree constants.  The
        # clipped value cannot be reconstructed, but its prevalence must be
        # visible so saturated episodes are not mistaken for calibrated ROM.
        if self.fast_save:
            joint_limit_fraction = np.zeros(len(WUJI_21_NAMES), dtype=np.float32)
            joint_limit_max_index = 0
            joint_limit_max_fraction = 0.0
        else:
            known_limits_rad = np.deg2rad(
                np.asarray([89.5, -10.0, 0.5, -35.0, -65.0, -15.0, 45.0], dtype=np.float32)
            )
            joint_limit_fraction = np.max(
                np.mean(
                    np.isclose(
                        aligned_angles[:, :, None],
                        known_limits_rad[None, None, :],
                        atol=np.deg2rad(0.02),
                    ),
                    axis=0,
                ),
                axis=1,
            )
            joint_limit_max_index = int(np.argmax(joint_limit_fraction))
            joint_limit_max_fraction = float(joint_limit_fraction[joint_limit_max_index])
        if (not self.fast_save) and joint_limit_max_fraction >= 0.05:
            print(
                "[QUALITY WARN] Wuji joint 命中已知固件限位："
                f"{WUJI_21_NAMES[joint_limit_max_index]}={joint_limit_max_fraction:.1%} frames；"
                "IK 校准保持不变；裁剪后的超限角度不可由软件恢复，已记录到 metadata"
            )
        if self.fast_save:
            joint_step_max, joint_step_joint = 0.0, 0
        elif len(canonical_joints) >= 2:
            joint_step = np.abs(np.diff(canonical_joints, axis=0))
            joint_step_max = float(joint_step.max())
            joint_step_joint = int(np.unravel_index(np.argmax(joint_step), joint_step.shape)[1])
        else:
            joint_step_max, joint_step_joint = 0.0, 0
        if (not self.fast_save) and joint_step_max > self.max_joint_step_rad:
            print(
                "[SAVE REFUSED] canonical hand joint 跳变过大："
                f"{FTP1_HAND_NAMES[joint_step_joint]}={joint_step_max:.3f} rad > "
                f"{self.max_joint_step_rad:.3f} rad；请检查手套佩戴/骨架跟踪后重录"
            )
            return
        gemini_store = (
            _decode_messages_to_memmap(
                gemini,
                message_index=2,
                compressed=self.camera.gemini_compressed,
            )
            if gemini
            else None
        )
        gemini_seq = np.asarray([frame[1] for frame in gemini], dtype=np.int64) if gemini else np.empty(0, dtype=np.int64)
        gemini_rgb_raw = gemini_store.array if gemini_store is not None else None
        # The mmap store now owns decoded pixels; release Gemini ROS messages
        # and their compressed byte payloads before tactile/Zarr work.
        del gemini
        aligned_wuji_wrist_pose = np.stack([self.right_wrist_pose_frames[i][2] for i in wrist_pose_idx]).astype(np.float32)
        aligned_camera_ego_pose = np.stack([camera_poses[i][2] for i in camera_pose_idx]).astype(np.float32) if camera_pose_idx is not None else None
        aligned_tracker_camera = np.stack([tracker_camera[i][2] for i in tracker_camera_idx]).astype(np.float32) if tracker_camera_idx is not None else None
        aligned_tracker_wrist = np.stack([tracker_wrist[i][2] for i in tracker_wrist_idx]).astype(np.float32) if tracker_wrist_idx is not None else None
        aligned_right_wrist_pose = (
            _wrist_pose_with_tracker_translation(aligned_wuji_wrist_pose, aligned_tracker_wrist)
            if aligned_tracker_wrist is not None
            else aligned_wuji_wrist_pose
        )
        aligned_tactile_source = np.stack([self.tactile_frames[i][2] for i in tactile_idx]).astype(np.float32)
        active_counts = np.asarray([tactile_active_taxel_count(frame) for frame in aligned_tactile_source], dtype=np.int32)
        if not np.all(active_counts == WUJI_OFFICIAL_ACTIVE_TAXELS):
            print(
                f"[SAVE REFUSED] Wuji full tactile map is {sorted(set(active_counts.tolist()))} active taxels, "
                f"not official {WUJI_OFFICIAL_ACTIVE_TAXELS}; do not record training data with this mapping."
            )
            return
        aligned_tactile_raw = (
            self.tactile_health_mask.apply_batch(aligned_tactile_source)
            if self.tactile_health_mask is not None
            else aligned_tactile_source
        )
        tactile_valid, tactile_raw_matrix, _summary = zip(*[_tactile_features(frame) for frame in aligned_tactile_raw])
        tactile_raw_matrix = np.stack(tactile_raw_matrix).astype(np.float32)
        tactile_valid = np.stack(tactile_valid).astype(np.uint8)
        # FTP-1 MatrixCNN must never consume the collector-only -1 padding.
        # The audit group retains the physical-taxel mask and raw -1 sentinel.
        tactile_pressure = np.maximum(tactile_raw_matrix - self.tactile_baseline[None, :, :], 0.0)
        tactile_pressure = np.where(tactile_valid.astype(bool), tactile_pressure, 0.0).astype(np.float32)
        zone_raw_stats = np.stack([self.zone_frames[i][2] for i in zone_idx]).astype(np.float32)
        zone_counts = np.stack([self.zone_frames[i][3] for i in zone_idx]).astype(np.int32)
        if not np.all(zone_counts.sum(axis=1) == WUJI_OFFICIAL_ACTIVE_TAXELS):
            print("[SAVE REFUSED] Wuji tactile_zones active-count sum does not match official 526-taxel contract")
            return
        if self.tactile_zone_baseline is None:
            print("[SAVE REFUSED] 无可用六区触觉 baseline；重新输入 b 后重录")
            return
        zone_pressure = np.maximum(zone_raw_stats - self.tactile_zone_baseline[None, :, :], 0.0).astype(np.float32)
        aligned_emg = None
        emg_seq = np.empty(0, dtype=np.int64)
        emg_movement = np.empty(0, dtype=np.int32)
        emg_clip_fraction = 0.0
        emg_clip_fraction_by_channel = np.zeros(8, dtype=np.float64)
        if self.emg_enabled:
            assert emg_idx is not None and emg_age_us is not None
            aligned_emg = np.stack([emg_frames[i][3] for i in emg_idx]).astype(np.int32)
            emg_seq = np.asarray([frame[1] for frame in emg_frames], dtype=np.int64)
            emg_movement = np.asarray([frame[2] for frame in emg_frames], dtype=np.int32)
            emg_hz = _rate(emg_ts_ns)
            emg_missing = int(np.maximum(np.diff(emg_seq) - 1, 0).sum()) if len(emg_seq) >= 2 else 0
            emg_missing_ratio = emg_missing / max(len(emg_seq) + emg_missing, 1)
            if self.fast_save:
                pass
            else:
                native_emg = np.stack([frame[3] for frame in emg_frames]).astype(np.int32)
                emg_clipped = (native_emg == self.emg.clip_max) | (native_emg == self.emg.clip_min)
                emg_clip_fraction = float(np.mean(emg_clipped))
                emg_clip_fraction_by_channel = np.mean(emg_clipped, axis=0)
                if emg_clip_fraction >= 0.01 or float(emg_clip_fraction_by_channel.max()) >= 0.05:
                    worst_channel = int(np.argmax(emg_clip_fraction_by_channel))
                    print(
                        "[QUALITY WARN] serial EMG clipping/saturation: "
                        f"overall={emg_clip_fraction:.2%}, channel-{worst_channel}="
                        f"{emg_clip_fraction_by_channel[worst_channel]:.2%}；"
                        "边界值不可由软件恢复，不建议用于精细肌力或连续 force estimation"
                    )
            if (not self.fast_save) and (emg_hz < self.min_emg_hz or emg_missing_ratio > self.max_emg_missing_ratio):
                print(
                    "[QUALITY WARN] serial EMG native stream quality reduced: "
                    f"source={emg_hz:.1f} Hz (min={self.min_emg_hz:.1f}), missing={emg_missing_ratio:.2%} "
                    f"(target={self.max_emg_missing_ratio:.2%})；有效对齐行继续保存，原生流保留供审计"
                )
        aligned_myo = None
        myo_seq = np.empty(0, dtype=np.int64)
        myo_movement = np.empty(0, dtype=np.int32)
        myo_clip_fraction = 0.0
        myo_clip_fraction_by_channel = np.zeros(8, dtype=np.float64)
        if self.myo_enabled:
            assert myo_idx is not None and myo_age_us is not None
            aligned_myo = np.stack([myo_frames[i][3] for i in myo_idx]).astype(np.int8)
            myo_seq = np.asarray([frame[1] for frame in myo_frames], dtype=np.int64)
            myo_movement = np.asarray([frame[2] for frame in myo_frames], dtype=np.int32)
            myo_hz = _rate(myo_ts_ns)
            myo_missing = int(np.maximum(np.diff(myo_seq) - 1, 0).sum()) if len(myo_seq) >= 2 else 0
            myo_missing_ratio = myo_missing / max(len(myo_seq) + myo_missing, 1)
            if not self.fast_save:
                native_myo = np.stack([frame[3] for frame in myo_frames]).astype(np.int8)
                myo_clipped = (native_myo == 127) | (native_myo == -128)
                myo_clip_fraction = float(np.mean(myo_clipped))
                myo_clip_fraction_by_channel = np.mean(myo_clipped, axis=0)
            if (not self.fast_save) and (
                myo_hz < self.min_myo_hz or myo_missing_ratio > self.max_myo_missing_ratio
            ):
                print(
                    "[QUALITY WARN] Myo RAW EMG native stream quality reduced: "
                    f"source={myo_hz:.1f} Hz (min={self.min_myo_hz:.1f}), "
                    f"missing={myo_missing_ratio:.2%} (target={self.max_myo_missing_ratio:.2%})"
                )
        t = len(color_ts_ns)
        instruction_width = max(256, len(self.instruction))

        final_episode_path = self._next_episode_path()
        episode_path = final_episode_path.with_name(f".{final_episode_path.stem}.zarr.tmp")
        self._active_temp_episode = episode_path
        video_jobs = []
        video_results = []
        video_executor = None
        if self.save_mp4:
            video_dir = self.output_dir / "videos"
            if aligned_color is not None:
                ego_path = video_dir / f"{episode_path.stem}_ego.mp4"
                video_jobs.append(("camera_ego_rgb", ego_path, aligned_color, _rate(color_ts_ns)))
            if gemini_rgb_raw is not None:
                main_path = video_dir / f"{episode_path.stem}_main.mp4"
                video_jobs.append(("camera_main_rgb", main_path, gemini_rgb_raw, _rate(gemini_ts_ns)))
            if video_jobs:
                print(f"[SAVE] writing Zarr + encoding {len(video_jobs)} MP4 sidecar(s) in parallel...")
                video_executor = ThreadPoolExecutor(max_workers=len(video_jobs))
                video_results = [
                    (name, path, video_executor.submit(_write_mp4, path, frames, fps, self.mp4_codec))
                    for name, path, frames, fps in video_jobs
                ]
        io_started = time.perf_counter()
        group = zarr.open_group(str(episode_path), mode="w")
        data = group.create_group("data")
        meta = group.create_group("meta")
        audit = group.create_group("audit")
        streams = group.create_group("streams")
        arrays = {
            # FTP-1 training contract. Every array here is exactly T-major.
            "timestamps": color_ts_ns,
            "right_hand_joints": canonical_joints,
            "right_wrist_pose": aligned_right_wrist_pose,
            "right_hand_joints_idx": np.broadcast_to(FTP1_HAND_FAAS_IDX, (t, len(FTP1_HAND_FAAS_IDX))).copy(),
            "right_tactile_data_wuji": tactile_pressure[:, None, :, :],
            # Whole-hand fallback area.  No vendor zone map has been supplied,
            # so this is truthfully the FTP-1 whole-hand area id, not thumb tip.
            "right_tactile_area_wuji": np.full((t, 1), 5, dtype=np.int32),
            "right_tactile_sensor_wuji": np.full(t, TACTILE_SENSOR, dtype=f"<U{len(TACTILE_SENSOR)}"),
            "right_tactile_type_wuji": np.full(t, TACTILE_TYPE, dtype=f"<U{len(TACTILE_TYPE)}"),
            "right_tactile_data_wuji_zones": zone_pressure,
            "right_tactile_area_wuji_zones": np.broadcast_to(WUJI_TACTILE_ZONE_AREAS, (t, len(WUJI_TACTILE_ZONE_AREAS))).copy(),
            "right_tactile_sensor_wuji_zones": np.full(t, TACTILE_ZONE_SENSOR, dtype=f"<U{len(TACTILE_ZONE_SENSOR)}"),
            "right_tactile_type_wuji_zones": np.full(t, "state", dtype="<U5"),
            "sub_task_instruction": np.full(t, self.instruction, dtype=f"<U{instruction_width}"),
        }
        if aligned_emg is not None:
            arrays["right_forearm_emg_wavletech"] = aligned_emg
        if aligned_myo is not None:
            arrays["right_forearm_emg"] = aligned_myo
        if aligned_color is not None:
            arrays["camera_ego_rgb"] = aligned_color
        if aligned_tracker_camera is not None:
            arrays["camera_tracker_pose"] = aligned_tracker_camera
        if aligned_tracker_wrist is not None:
            arrays["right_wrist_tracker_pose"] = aligned_tracker_wrist
        audit_arrays = {
            "right_wrist_pose_source_timestamp_us": wrist_pose_ts_us[wrist_pose_idx],
            "right_wrist_pose_source_seq": np.asarray([self.right_wrist_pose_frames[i][1] for i in wrist_pose_idx], dtype=np.int64),
            "right_wrist_pose_age_us": wrist_pose_age_us,
            "wuji_hand_joints_anatomical": aligned_angles,
            "wuji_local_joint_idx": np.broadcast_to(WUJI_LOCAL_HAND_IDX, (t, len(WUJI_LOCAL_HAND_IDX))).copy(),
            "wuji_hand_skeleton_mediapipe": aligned_skeleton,
            "right_hand_joints_stable_local": stable_joints,
            "right_hand_joints_published_ftp1": published_joints,
            "right_hand_joints_source_timestamp_us": skeleton_ts_us[skeleton_idx],
            "right_hand_joints_source_seq": np.asarray([self.skeleton_frames[i][1] for i in skeleton_idx], dtype=np.int64),
            "right_hand_joints_age_us": skeleton_age_us,
            "wuji_hand_joints_raw_wuji_5x5": aligned_angles_raw,
            "wuji_hand_joints_anatomical_source_timestamp_us": angle_ts_us[angle_idx],
            "wuji_hand_joints_anatomical_source_seq": np.asarray([self.angle_frames[i][1] for i in angle_idx], dtype=np.int64),
            "wuji_hand_joints_anatomical_age_us": angle_age_us,
            "wuji_hand_skeleton_source_timestamp_us": skeleton_ts_us[skeleton_idx],
            "wuji_hand_skeleton_source_seq": np.asarray([self.skeleton_frames[i][1] for i in skeleton_idx], dtype=np.int64),
            "wuji_hand_skeleton_age_us": skeleton_age_us,
            "right_tactile_source_timestamp_us": tactile_ts_us[tactile_idx],
            "right_tactile_source_seq": np.asarray([self.tactile_frames[i][1] for i in tactile_idx], dtype=np.int64),
            "right_tactile_age_us": tactile_age_us,
            "right_tactile_valid_mask_wuji": tactile_valid[:, None, :, :],
            "right_tactile_raw_wuji": tactile_raw_matrix[:, None, :, :],
            "right_tactile_zone_source_timestamp_us": zone_ts_us[zone_idx],
            "right_tactile_zone_source_seq": np.asarray([self.zone_frames[i][1] for i in zone_idx], dtype=np.int64),
            "right_tactile_zone_age_us": zone_age_us,
            "right_tactile_zone_valid_taxels": zone_counts,
            "right_tactile_zone_raw_stats_wuji": zone_raw_stats,
        }
        if self.emg_enabled:
            assert emg_idx is not None and emg_age_us is not None
            audit_arrays.update({
                "right_forearm_emg_wavletech_source_timestamp_ns": emg_ts_ns[emg_idx],
                "right_forearm_emg_wavletech_source_seq": emg_seq[emg_idx],
                "right_forearm_emg_wavletech_age_us": emg_age_us,
            })
        if self.myo_enabled:
            assert myo_idx is not None and myo_age_us is not None
            audit_arrays.update({
                "right_forearm_emg_source_timestamp_ns": myo_ts_ns[myo_idx],
                "right_forearm_emg_source_seq": myo_seq[myo_idx],
                "right_forearm_emg_age_us": myo_age_us,
            })
        if self.camera.capture_ego:
            audit_arrays.update({
                "camera_ego_rgb_source_timestamp_ns": raw_color_ts_ns[ego_idx],
                "camera_ego_rgb_source_index": ego_idx,
                "camera_ego_rgb_age_us": ego_age_us,
            })
        if aligned_camera_ego_pose is not None:
            arrays["camera_ego_pose"] = aligned_camera_ego_pose
            audit_arrays.update({
                "camera_ego_pose_source_timestamp_ns": camera_pose_ts_ns[camera_pose_idx],
                "camera_ego_pose_source_seq": np.asarray([camera_poses[i][1] for i in camera_pose_idx], dtype=np.int64),
                "camera_ego_pose_age_us": camera_pose_age_us,
            })
        if tracker_camera_idx is not None:
            audit_arrays.update({
                "camera_tracker_pose_source_timestamp_ns": tracker_camera_ts_ns[tracker_camera_idx],
                "camera_tracker_pose_source_seq": np.asarray([tracker_camera[i][1] for i in tracker_camera_idx], dtype=np.int64),
                "camera_tracker_pose_age_us": tracker_camera_age_us,
                "right_wrist_tracker_pose_source_timestamp_ns": tracker_wrist_ts_ns[tracker_wrist_idx],
                "right_wrist_tracker_pose_source_seq": np.asarray([tracker_wrist[i][1] for i in tracker_wrist_idx], dtype=np.int64),
                "right_wrist_tracker_pose_age_us": tracker_wrist_age_us,
                "right_wrist_pose_translation_source_timestamp_ns": tracker_wrist_ts_ns[tracker_wrist_idx],
                "right_wrist_pose_translation_source_seq": np.asarray([tracker_wrist[i][1] for i in tracker_wrist_idx], dtype=np.int64),
                "right_wrist_pose_translation_age_us": tracker_wrist_age_us,
            })
        _create_arrays_parallel(data, arrays, t)
        _create_arrays_parallel(audit, audit_arrays, t)
        if self.emg_enabled:
            # Native EMG is intentionally not resampled to the RGB time axis.
            # It is lossless source data for signal processing and provenance.
            _create_array(streams, "right_forearm_emg_wavletech_raw", np.stack([frame[3] for frame in emg_frames]).astype(np.int32))
            _create_array(streams, "right_forearm_emg_wavletech_timestamp_ns", emg_ts_ns)
            _create_array(streams, "right_forearm_emg_wavletech_clock_raw_timestamp_ns", emg_clock_raw_ts_ns)
            _create_array(streams, "right_forearm_emg_wavletech_arrival_timestamp_ns", emg_arrival_ts_ns)
            _create_array(streams, "right_forearm_emg_wavletech_seq", emg_seq)
            imu_frames = [
                frame for frame in self.emg_imu_frames
                if color_ts_ns[0] - self.max_emg_age_us * 1_000
                <= frame[0]
                <= color_ts_ns[-1] + self.max_emg_age_us * 1_000
            ]
            if imu_frames:
                _create_array(streams, "right_forearm_imu_wavletech_raw", np.stack([frame[2] for frame in imu_frames]).astype(np.float32))
                _create_array(streams, "right_forearm_imu_wavletech_timestamp_ns", np.asarray([frame[0] for frame in imu_frames], dtype=np.int64))
                _create_array(streams, "right_forearm_imu_wavletech_seq", np.asarray([frame[1] for frame in imu_frames], dtype=np.int64))
                _create_array(streams, "right_forearm_imu_wavletech_arrival_timestamp_ns", np.asarray([frame[3] for frame in imu_frames], dtype=np.int64))
        if self.myo_enabled:
            _create_array(streams, "right_forearm_emg_raw", np.stack([frame[3] for frame in myo_frames]).astype(np.int8))
            _create_array(streams, "right_forearm_emg_timestamp_ns", myo_ts_ns)
            _create_array(streams, "right_forearm_emg_clock_raw_timestamp_ns", myo_clock_raw_ts_ns)
            _create_array(streams, "right_forearm_emg_arrival_timestamp_ns", myo_arrival_ts_ns)
            _create_array(streams, "right_forearm_emg_seq", myo_seq)
            _create_array(streams, "right_forearm_emg_movement", myo_movement)
        _create_array(streams, "right_wrist_pose_raw", np.stack([frame[2] for frame in self.right_wrist_pose_frames]).astype(np.float32))
        _create_array(streams, "right_wrist_pose_timestamp_us", wrist_pose_ts_us)
        _create_array(streams, "right_wrist_pose_seq", np.asarray([frame[1] for frame in self.right_wrist_pose_frames], dtype=np.int64))
        if gemini_rgb_raw is not None:
            _create_array(streams, "camera_main_rgb_raw", gemini_rgb_raw)
            _create_array(streams, "camera_main_rgb_timestamp_ns", gemini_ts_ns)
            _create_array(streams, "camera_main_rgb_seq", gemini_seq)
        if camera_poses:
            _create_array(streams, "camera_ego_pose_raw", np.stack([frame[2] for frame in camera_poses]).astype(np.float32))
            _create_array(streams, "camera_ego_pose_timestamp_ns", camera_pose_ts_ns)
            _create_array(streams, "camera_ego_pose_seq", np.asarray([frame[1] for frame in camera_poses], dtype=np.int64))
        if tracker_camera:
            _create_array(streams, "camera_tracker_pose_raw", np.stack([frame[2] for frame in tracker_camera]).astype(np.float32))
            _create_array(streams, "camera_tracker_pose_timestamp_ns", tracker_camera_ts_ns)
            _create_array(streams, "camera_tracker_pose_seq", np.asarray([frame[1] for frame in tracker_camera], dtype=np.int64))
        if tracker_wrist:
            _create_array(streams, "right_wrist_tracker_pose_raw", np.stack([frame[2] for frame in tracker_wrist]).astype(np.float32))
            _create_array(streams, "right_wrist_tracker_pose_timestamp_ns", tracker_wrist_ts_ns)
            _create_array(streams, "right_wrist_tracker_pose_seq", np.asarray([frame[1] for frame in tracker_wrist], dtype=np.int64))
        _create_array(meta, "episode_ends", np.asarray([t], dtype=np.int64))
        _create_array(audit, "tactile_baseline_wuji", self.tactile_baseline[None, :, :])
        _create_array(audit, "tactile_zone_baseline_wuji", self.tactile_zone_baseline[None, :, :])
        group.attrs.update({
            "format": (
                "ftp1_standardized_zarr_wuji_d435_gemini_dual_emg_v10_no_ego" if not self.camera.capture_ego
                else "ftp1_standardized_zarr_wuji_ego_gemini_dual_emg_v10"
            ) if (self.emg_enabled or self.myo_enabled) else (
                "ftp1_standardized_zarr_wuji_d435_gemini_v8_no_ego" if not self.camera.capture_ego
                else "ftp1_standardized_zarr_wuji_ego_gemini_v8"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "success": True,
            "participant_name": self.participant_name,
            "task_id": self.task_id,
            "glove_condition": self.glove_condition,
            "canonical_time_axis": f"actual captured first-person {self.camera.ego_source} RGB ROS header stamps in ns; timestamp gaps preserve real source dropouts; sparse unsynchronized rows may be dropped only within the recorded quality gate; no duplicated/interpolated RGB" if self.camera.capture_ego else "Wuji hand skeleton ROS timestamps in us converted to ns; first-person RGB disabled",
            "canonical_hz": _rate(color_ts_ns),
            "canonical_nominal_period_ns": timeline["nominal_period_ns"],
            "canonical_max_gap_ns": timeline["max_gap_ns"],
            "canonical_missing_slot_count": timeline["missing_slot_count"],
            "canonical_missing_slot_ratio": timeline["missing_slot_ratio"],
            "alignment_rows_dropped": alignment_dropped,
            "alignment_rows_drop_ratio": alignment_drop_ratio,
            "alignment_max_consecutive_rows_dropped": alignment_max_drop_run,
            "quality_gate_max_rgb_gap_ms": self.max_rgb_gap_ns / 1e6,
            "quality_gate_final_max_gap_after_alignment_ms": final_max_gap_ns / 1e6,
            "quality_gate_max_missing_ratio": self.max_missing_ratio,
            "quality_gate_max_alignment_drop_ratio": self.max_align_drop_ratio,
            "quality_gate_max_wuji_gap_ms": self.max_wuji_gap_ns / 1e6,
            "quality_gate_max_alignment_drop_run": 3,
            "quality_gate_max_glove_age_ms": self.max_glove_age_us / 1000.0,
            "quality_gate_max_wrist_pose_age_ms": self.max_wrist_pose_age_us / 1000.0,
            "quality_gate_max_gemini_age_ms": self.max_gemini_age_us / 1000.0,
            "quality_gate_max_camera_pose_age_ms": self.max_camera_pose_age_us / 1000.0,
            "quality_gate_max_tracker_age_ms": self.max_tracker_age_us / 1000.0,
            "quality_gate_max_joint_step_rad": self.max_joint_step_rad,
            "emg_enabled": self.emg_enabled,
            "wavletech_emg_enabled": self.emg_enabled,
            "myo_enabled": self.myo_enabled,
            "quality_gate_wavletech_max_alignment_age_ms": self.max_emg_age_us / 1000.0,
            "quality_gate_wavletech_min_native_hz": self.min_emg_hz,
            "quality_gate_wavletech_max_missing_ratio": self.max_emg_missing_ratio,
            "quality_gate_wavletech_max_gap_ms": self.max_emg_gap_ns / 1e6,
            "quality_gate_myo_max_alignment_age_ms": self.max_myo_age_us / 1000.0,
            "quality_gate_myo_min_native_hz": self.min_myo_hz,
            "quality_gate_myo_max_missing_ratio": self.max_myo_missing_ratio,
            "quality_gate_myo_max_gap_ms": self.max_myo_gap_ns / 1e6,
            "quality_gate_min_ego_sharpness_p05": self.min_ego_sharpness,
            "quality_warning_ego_sharpness_below_threshold": bool(
                self.camera.capture_ego and sharpness["p05"] < self.min_ego_sharpness
            ),
            "right_hand_joints_max_step_rad": joint_step_max,
            "right_hand_joints_max_step_joint": FTP1_HAND_NAMES[joint_step_joint],
            "wuji_joint_limit_max_fraction": joint_limit_max_fraction,
            "wuji_joint_limit_max_joint": WUJI_21_NAMES[joint_limit_max_index],
            "wuji_joint_limit_fraction_by_joint": {
                name: float(fraction)
                for name, fraction in zip(WUJI_21_NAMES, joint_limit_fraction)
            },
            "camera_ego_rgb_role": "first_person_vr" if self.camera.capture_ego and self.camera.ego_compressed else ("first_person_realsense" if self.camera.capture_ego else "disabled"),
            "camera_ego_rgb_source": self.camera.ego_source if self.camera.capture_ego else "",
            "camera_ego_rgb_source_topic": self.camera.ego_topic if self.camera.capture_ego else "",
            "camera_ego_rgb_timestamp_semantics": (
                "host decode-completion time with cadence smoothing; native PICO TCP packets "
                "contain no exposure timestamp; camera/network latency is not calibrated"
                if self.camera.capture_ego and "PICO native" in self.camera.ego_source
                else "source ROS header timestamp" if self.camera.capture_ego else "disabled"
            ),
            "camera_ego_rgb_transport": "sensor_msgs/CompressedImage (JPEG)" if self.camera.ego_compressed else ("sensor_msgs/Image" if self.camera.capture_ego else "disabled"),
            "camera_rgb_zarr_compression": (
                f"imagecodecs JPEG quality={_fast_rgb_jpeg_quality() if self.fast_save else 90} "
                "subsampling=4:2:0; transparent uint8 ndarray decode"
            ),
            "camera_ego_rgb_intrinsics_valid": bool(self.camera.capture_ego and self.camera.ego_require_camera_info and color_info),
            "camera_ego_rgb_geometry_note": "PICO native VST stereo CameraHandle H.264 side-by-side frame; display UI and tracker panels are not present; not calibrated with ROS CameraInfo" if self.camera.capture_ego and self.camera.ego_compressed else "ROS camera image with CameraInfo" if self.camera.capture_ego else "disabled",
            "camera_ego_rgb_intrinsics": color_info or {},
            "camera_main_rgb_role": "third_person_gemini_independent_timeline",
            "camera_main_rgb_raw_hz": _rate(gemini_ts_ns),
            "camera_main_rgb_raw_nominal_period_ns": gemini_timeline["nominal_period_ns"],
            "camera_main_rgb_raw_max_gap_ns": gemini_timeline["max_gap_ns"],
            "camera_main_rgb_raw_missing_slot_count": gemini_timeline["missing_slot_count"],
            "camera_main_rgb_raw_missing_slot_ratio": gemini_timeline["missing_slot_ratio"],
            "camera_ego_pose_raw_stream_path": "streams/camera_ego_pose_raw",
            "camera_ego_pose_raw_hz": _rate(camera_pose_ts_ns),
            "right_wrist_pose_semantics": "[tracker1_x,tracker1_y,tracker1_z,wuji_roll,wuji_pitch,wuji_yaw]; XYZ from wrist tracker PC2310MLKC190056G in PICO tracking coordinates, orientation from Wuji dynamic tf waist->r_wrist; metres and radians, ROS XYZ intrinsic RPY" if aligned_tracker_wrist is not None else "[x,y,z,roll,pitch,yaw] from Wuji dynamic tf waist->r_wrist; metres and radians, ROS XYZ intrinsic RPY (wrist tracker disabled)",
            "right_wrist_pose_translation_source": "PICO wrist tracker1 PC2310MLKC190056G /pico/right_wrist/raw_pose" if aligned_tracker_wrist is not None else "Wuji dynamic tf waist->r_wrist (wrist tracker disabled)",
            "right_wrist_pose_orientation_source": "Wuji dynamic tf waist->r_wrist",
            "camera_ego_pose_semantics": "[0,0,0,roll,pitch,yaw] from Gemini 2 accel+gyro on its own optical origin; metres and radians. Roll/pitch are gravity-stabilized, yaw is relative to process start and drifts without a magnetometer/tracker.",
            "camera_tracker_pose_semantics": "[x,y,z+0.1m,roll,pitch,yaw] from tracker PC2310MLKC190573G; metres and radians, ROS XYZ intrinsic RPY.",
            "right_wrist_tracker_pose_semantics": "[x,y,z,roll,pitch,yaw] from tracker PC2310MLKC190056G; metres and radians, ROS XYZ intrinsic RPY.",
            "camera_tracker_pose_source_topic": self.camera.tracker_camera_topic if self.camera.trackers_enabled else "",
            "right_wrist_tracker_pose_source_topic": self.camera.tracker_wrist_topic if self.camera.trackers_enabled else "",
            "camera_tracker_pose_z_offset_m": self.camera.tracker_camera_z_offset_m if self.camera.trackers_enabled else 0.0,
            "camera_tracker_pose_raw_hz": _rate(tracker_camera_ts_ns),
            "right_wrist_tracker_pose_raw_hz": _rate(tracker_wrist_ts_ns),
            "camera_tracker_pose_xyz_quality": tracker_camera_xyz_quality,
            "right_wrist_tracker_pose_xyz_quality": tracker_wrist_xyz_quality,
            "camera_tracker_pose_alignment": "nearest real native pose; a source sample may serve adjacent canonical rows; raw stream and per-row source seq/timestamp are retained",
            "right_wrist_tracker_pose_alignment": "nearest real native pose; a source sample may serve adjacent canonical rows; raw stream and per-row source seq/timestamp are retained",
            "camera_tracker_pose_alignment_reused_rows": int(len(tracker_camera_idx) - len(np.unique(tracker_camera_idx))) if tracker_camera_idx is not None else 0,
            "right_wrist_tracker_pose_alignment_reused_rows": int(len(tracker_wrist_idx) - len(np.unique(tracker_wrist_idx))) if tracker_wrist_idx is not None else 0,
            "right_hand_joints_unit": "radian",
            "right_hand_joints_layout": "ftp1_human_canonical_from_wuji_hand_skeleton_mediapipe",
            "right_hand_joints_adduction_method": "stable_mcp_to_pip_palm_plane_axis; FTP-1 layout and FAAS indices unchanged",
            "right_hand_joints_published_ftp1_note": "audit-only exact published MCP-to-tip projected-axis equations; may be ill-conditioned during deep finger curl",
            "right_hand_joint_names": list(FTP1_HAND_NAMES),
            "wuji_hand_joints_raw_wuji_5x5_note": "audit/raw SDK (T,5,5); index/middle/ring/pinky slot 4 is interface padding",
            "wuji_hand_joints_anatomical_names": list(WUJI_21_NAMES),
            "wuji_hand_skeleton_layout": "MediaPipe 21 landmarks in wrist frame; SDK documented order",
            "right_hand_joints_idx_semantics": "ftp1_mano_human_canonical_faas_index",
            "right_hand_joints_faas_verified": True,
            "right_hand_joints_idx_note": "FTP-1 MANO human canonical slots; values were calculated from Wuji hand_skeleton, not copied from anatomical angles",
            "right_tactile_data_wuji_unit": "sdk_calibrated_raw_not_newton",
            "right_tactile_data_wuji_invalid_value": 0.0,
            "right_tactile_data_wuji_geometry": "one whole-hand 24x31 Wuji pressure matrix; invalid cells zeroed for FTP-1, raw -1 mask in audit",
            "wuji_tactile_official_active_taxels": WUJI_OFFICIAL_ACTIVE_TAXELS,
            "right_tactile_area_wuji_semantics": "5=whole_hand fallback; vendor physical-zone map not supplied",
            "right_tactile_type_wuji_note": "FTP-1 MatrixCNN tactile encoder",
            "right_tactile_area_wuji_zones_semantics": "vendor tactile_zones order: thumb=0,index=1,middle=2,ring=3,pinky=4,palm=5",
            "right_tactile_data_wuji_zones_feature_order": ["mean", "max", "sum"],
            "right_tactile_data_wuji_zones_unit": "sdk_calibrated_raw_not_newton; baseline-subtracted vendor zone aggregation",
            "right_tactile_data_wuji_zones_health_mask_applied": False,
            "right_tactile_data_wuji_zones_health_mask_note": "vendor zone arrays have no public raw-cell coordinate map; use whole-hand matrix for health-masked tactile training",
            "tactile_baseline_subtracted": True,
            "tactile_baseline_frame_count": self.tactile_baseline_count,
            "tactile_baseline_captured_unix_ns": self.tactile_baseline_captured_ns,
            "operator_protocol_confirmed_unix_ns": self.protocol_confirmed_ns,
            "source_glove_angle_hz": _rate(angle_ts_us * 1_000),
            "source_glove_tactile_hz": _rate(tactile_ts_us * 1_000),
            "source_glove_skeleton_hz": _rate(skeleton_ts_us * 1_000),
            "source_glove_tactile_zones_hz": _rate(zone_ts_us * 1_000),
            "right_forearm_emg_wavletech_layout": f"{self.emg.emg_layout}; canonical row is nearest native sample" if self.emg_enabled else "disabled",
            "right_forearm_emg_wavletech_unit": self.emg.emg_unit if self.emg_enabled else "disabled",
            "right_forearm_emg_wavletech_source": self.emg.source_name if self.emg_enabled else "disabled",
            "right_forearm_emg_wavletech_driver": self.emg.driver if self.emg is not None else "disabled",
            "right_forearm_emg_wavletech_native_hz": emg_hz,
            "right_forearm_emg_wavletech_native_sample_count": len(emg_frames),
            "right_forearm_emg_wavletech_missing_count": emg_missing,
            "right_forearm_emg_wavletech_missing_ratio": emg_missing_ratio,
            "right_forearm_emg_wavletech_clip_fraction": emg_clip_fraction,
            "right_forearm_emg_wavletech_clip_fraction_by_channel": emg_clip_fraction_by_channel.tolist(),
            "right_forearm_emg_wavletech_max_gap_ms": emg_max_gap_ns / 1e6 if self.emg_enabled else 0.0,
            "right_forearm_emg_wavletech_clock_offset_ms": emg_clock_offset_ns / 1e6 if self.emg_enabled else 0.0,
            "right_forearm_emg_wavletech_timestamp_semantics": "host serial packet arrival timeline with affine source/arrival correction; raw and arrival timestamps retained" if self.emg_enabled else "disabled",
            "right_forearm_emg_wavletech_raw_stream_path": "streams/right_forearm_emg_wavletech_raw" if self.emg_enabled else "",
            "right_forearm_imu_wavletech_raw_stream_path": "streams/right_forearm_imu_wavletech_raw" if "right_forearm_imu_wavletech_raw" in streams else "",
            "right_forearm_imu_wavletech_layout": "[gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z]" if self.emg_enabled else "disabled",
            "right_forearm_imu_wavletech_unit": "[rad/s,rad/s,rad/s,m/s^2,m/s^2,m/s^2]" if self.emg_enabled else "disabled",
            "right_forearm_emg_layout": "Myo RAW EMG, 8 signed int8 channels; canonical row is nearest native sample" if self.myo_enabled else "disabled",
            "right_forearm_emg_unit": "raw_adc_not_millivolt" if self.myo_enabled else "disabled",
            "right_forearm_emg_source": "Myo Armband via BLED112/BGAPI" if self.myo_enabled else "disabled",
            "right_forearm_emg_driver": self.myo.driver if self.myo is not None else "disabled",
            "right_forearm_emg_native_hz": myo_hz,
            "right_forearm_emg_native_sample_count": len(myo_frames),
            "right_forearm_emg_missing_count": myo_missing,
            "right_forearm_emg_missing_ratio": myo_missing_ratio,
            "right_forearm_emg_clip_fraction": myo_clip_fraction,
            "right_forearm_emg_clip_fraction_by_channel": myo_clip_fraction_by_channel.tolist(),
            "right_forearm_emg_max_gap_ms": myo_max_gap_ns / 1e6 if self.myo_enabled else 0.0,
            "right_forearm_emg_clock_offset_ms": myo_clock_offset_ns / 1e6 if self.myo_enabled else 0.0,
            "right_forearm_emg_timestamp_semantics": "smooth Myo native sample timeline plus affine arrival correction; raw clock and arrivals retained" if self.myo_enabled else "disabled",
            "right_forearm_emg_raw_stream_path": "streams/right_forearm_emg_raw" if self.myo_enabled else "",
            "alignment": "nearest source frame to each first-person RGB timestamp for non-image state; camera_main_rgb_raw remains on its own source timeline",
            "max_alignment_age_us": max_age_us,
            "started_unix_ns": self.started_ns,
            "camera_ego_rgb_sharpness_laplacian": sharpness if self.camera.capture_ego else {},
            "privacy_face_blur_enabled": False,
            "privacy_face_processing": "disabled; RGB is saved without face detection or blur",
        })
        if self.tactile_health_mask is not None:
            group.attrs.update(self.tactile_health_mask.metadata())
        if gemini_rgb_raw is not None:
            group.attrs.update({
                "camera_main_rgb_source_topic": self.camera.gemini_topic,
                "camera_main_rgb_raw_stream_path": "streams/camera_main_rgb_raw",
                "camera_main_rgb_timestamp_stream_path": "streams/camera_main_rgb_timestamp_ns",
            })
        if video_executor is not None:
            try:
                for name, path, future in video_results:
                    ok, detail = future.result()
                    group.attrs[f"{name}_mp4"] = str(path.relative_to(self.output_dir)) if ok else ""
                    group.attrs[f"{name}_mp4_status"] = detail
            finally:
                video_executor.shutdown(wait=True)
        io_seconds = time.perf_counter() - io_started
        prepare_seconds = io_started - save_started
        save_seconds = time.perf_counter() - save_started
        total_episode_seconds = time.perf_counter() - episode_started_mono
        group.attrs["zarr_and_mp4_seconds"] = io_seconds
        group.attrs["save_duration_seconds"] = save_seconds
        group.attrs["episode_total_duration_seconds"] = total_episode_seconds
        group.attrs["episode_capture_duration_seconds"] = capture_seconds
        wavletech_summary = (
            f"Wavletech={group.attrs['right_forearm_emg_wavletech_native_hz']:.1f} Hz"
            if self.emg_enabled else "Wavletech=disabled"
        )
        myo_summary = (
            f"Myo={group.attrs['right_forearm_emg_native_hz']:.1f} Hz"
            if self.myo_enabled else "Myo=disabled"
        )
        print(
            f"[SAVE] {final_episode_path.name}: T={t}, axis={group.attrs['canonical_hz']:.1f} Hz, "
            f"glove source angles={group.attrs['source_glove_angle_hz']:.1f} Hz, skeleton={group.attrs['source_glove_skeleton_hz']:.1f} Hz, "
            f"tactile={group.attrs['source_glove_tactile_hz']:.1f} Hz, "
            f"{myo_summary}, {wavletech_summary}, "
            f"max-align-age={max_age_us / 1000.0:.1f} ms, capture={capture_seconds:.1f}s, "
            f"prepare={prepare_seconds:.1f}s, zarr+mp4={io_seconds:.1f}s, "
            f"save={save_seconds:.1f}s, total={total_episode_seconds:.1f}s"
        )
        if gemini_rgb_raw is not None:
            print(f"       Gemini third-person independent source={group.attrs['camera_main_rgb_raw_hz']:.1f} Hz")
        if camera_poses:
            print(f"       Gemini IMU pose native source={group.attrs['camera_ego_pose_raw_hz']:.1f} Hz")
        if tracker_camera:
            print(f"       Tracker camera native source={_rate(tracker_camera_ts_ns):.1f} Hz, wrist={_rate(tracker_wrist_ts_ns):.1f} Hz")
        if self.save_mp4:
            if aligned_color is not None:
                print(f"       MP4 ego={group.attrs['camera_ego_rgb_mp4'] or 'FAILED'} ({group.attrs['camera_ego_rgb_mp4_status']})")
            if gemini_rgb_raw is not None:
                print(f"       MP4 main={group.attrs['camera_main_rgb_mp4'] or 'FAILED'} ({group.attrs['camera_main_rgb_mp4_status']})")
        print(
            f"       quality: dropped-align={alignment_dropped}/{raw_t}, "
            f"missing-slots={timeline['missing_slot_count']} ({timeline['missing_slot_ratio']:.2%}), "
            f"max-gap={timeline['max_gap_ns'] / 1e6:.2f} ms, "
            f"max-joint-step={joint_step_max:.3f} rad ({FTP1_HAND_NAMES[joint_step_joint]})"
        )
        if ego_store is not None:
            ego_store.close()
        if gemini_store is not None:
            gemini_store.close()
        # Publish only a fully written episode. Readers never see a partial
        # Zarr directory under the normal episode_*.zarr naming scheme.
        episode_path.rename(final_episode_path)
        self._active_temp_episode = None

    def refuse(self, reason: str = "操作员主动拒收", *, automatic: bool = False) -> None:
        # Stop the camera before clearing the glove buffers.  Both this path
        # and save() use the same lock as the drain thread, so no stale batch
        # can leak into the next episode.
        with self.state_lock:
            if not self.recording:
                if not automatic:
                    print("当前没有 episode")
                return
            self.recording = False
            self.last_refusal_reason = reason
        self.camera.discard_episode()
        temp_episode = getattr(self, "_active_temp_episode", None)
        if temp_episode is not None:
            import shutil
            shutil.rmtree(temp_episode, ignore_errors=True)
            self._active_temp_episode = None
        with self.state_lock:
            self.glove.clear()
            self.angle_frames = []
            self.tactile_frames = []
            self.skeleton_frames = []
            self.zone_frames = []
            self.right_wrist_pose_frames = []
            if self.emg is not None:
                self.emg.clear()
            myo = getattr(self, "myo", None)
            if myo is not None:
                myo.clear()
            self.emg_frames = []
            self.emg_imu_frames = []
            self.myo_frames = []
            self.emg_last_episode_ts_ns = 0
            self.emg_invalid_reason = ""
            self.emg_warning_reason = ""
            self.emg_alignment_warning_count = 0
            self.emg_alignment_warning_max_gap_ns = 0
            self.emg_clock_violation_started_mono = 0.0
            self.emg_next_clock_warning_mono = 0.0
            self.myo_last_episode_ts_ns = 0
            self.myo_invalid_reason = ""
            self.live_last_canonical_joints = None
            self.live_alignment_violation_started = {}
            self.live_next_alignment_check_mono = 0.0
        label = "AUTO REFUSE" if automatic else "REFUSE"
        print(f"\n[{label}] 当前 episode 已终止并丢弃：{reason}")

    def discard(self) -> None:
        self.refuse("操作员主动丢弃")

    def status(self) -> None:
        with self.state_lock:
            recording = self.recording
            counts = (
                len(self.angle_frames), len(self.skeleton_frames), len(self.tactile_frames),
                len(self.zone_frames), len(self.right_wrist_pose_frames),
                len(self.myo_frames), len(self.emg_frames),
            )
        print("glove :", self.glove.status())
        print("camera:", self.camera.status())
        myo = getattr(self, "myo", None)
        print("Myo EMG       :", myo.status() if myo is not None else "disabled")
        print("Wavletech EMG :", self.emg.status() if self.emg is not None else "disabled")
        print("baseline:", "OK" if self.tactile_baseline is not None else "未标定", f"frames={self.tactile_baseline_count}")
        print(
            "episode:", "采集中" if recording else "未采集",
            f"angles={counts[0]} skeleton={counts[1]} tactile={counts[2]} "
            f"zones={counts[3]} wrist_tf={counts[4]} myo={counts[5]} "
            f"wavletech={counts[6]}",
        )
        if self.last_refusal_reason:
            print("last refusal:", self.last_refusal_reason)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        if self.online_bridge is not None:
            self.online_bridge.close()
        if self._save_thread is not None:
            self._save_thread.join()
        temp_episode = getattr(self, "_active_temp_episode", None)
        if temp_episode is not None:
            import shutil
            shutil.rmtree(temp_episode, ignore_errors=True)
            self._active_temp_episode = None
        if self.emg is not None:
            self.emg.close()
        myo = getattr(self, "myo", None)
        if myo is not None:
            myo.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="FTP-1 domain directory; one episode_*.zarr is written per demo")
    parser.add_argument("--glove-sn", default=DEFAULT_GLOVE_SN)
    parser.add_argument("--glove-hz", type=float, default=120.0)
    parser.add_argument("--camera-name", default="camera")
    parser.add_argument("--camera-namespace", default="camera")
    parser.add_argument("--no-ego-camera", action="store_true", help="disable first-person RGB capture")
    parser.add_argument("--ego-topic", default="", help="override first-person RGB topic")
    parser.add_argument("--ego-compressed", action="store_true", help="first-person topic uses sensor_msgs/CompressedImage")
    parser.add_argument("--ego-source", default="RealSense D435", help="human-readable first-person camera source stored in metadata")
    parser.add_argument("--gemini-topic", default="/gemini2/color/image_raw/compressed", help="third-person Gemini CompressedImage topic; pass '' to disable")
    parser.add_argument("--gemini-imu-topic", default="/gemini2/gyro_accel/sample", help="Gemini synchronized sensor_msgs/Imu topic; pass '' to disable camera_ego_pose")
    parser.add_argument("--tracker-camera-topic", default="", help="PoseStamped topic for camera tracker; empty disables tracker capture")
    parser.add_argument("--tracker-wrist-topic", default="", help="PoseStamped topic for wrist tracker; empty disables tracker capture")
    parser.add_argument("--tracker-camera-z-offset-m", type=float, default=0.10, help="camera tracker z offset in metres")
    parser.add_argument("--max-tracker-age-ms", type=float, default=25.0, help="maximum tracker-to-canonical alignment age")
    parser.add_argument("--max-glove-age-ms", type=float, default=10.0, help="maximum |Wuji-axis| alignment age retained")
    parser.add_argument("--max-wrist-pose-age-ms", type=float, default=15.0, help="maximum |waist->r_wrist TF - RGB| alignment age retained")
    parser.add_argument("--max-gemini-age-ms", type=float, default=10.0, help="maximum |Gemini-ego| alignment age retained")
    parser.add_argument("--max-camera-pose-age-ms", type=float, default=20.0, help="maximum |Gemini IMU - RGB| alignment age retained")
    parser.add_argument("--max-rgb-gap-ms", type=float, default=50.0, help="refuse an episode if canonical ego RGB has a larger gap")
    parser.add_argument("--max-missing-ratio", type=float, default=0.02, help="refuse if estimated missing canonical RGB slots exceed this ratio")
    parser.add_argument("--max-align-drop-ratio", type=float, default=0.05, help="maximum fraction of canonical rows automatically removed for stale non-image sources")
    parser.add_argument("--max-wuji-gap-ms", type=float, default=500.0, help="maximum recoverable Wuji stream pause during an episode")
    parser.add_argument("--max-joint-step-rad", type=float, default=2.5, help="refuse if canonical hand state jumps farther in one RGB step")
    parser.add_argument("--joint-preflight-seconds", type=float, default=2.0, help="operator flexion-check duration before each episode")
    parser.add_argument("--joint-preflight-min-range-deg", type=float, default=3.0, help="minimum motion required from key PIP/DIP channels")
    parser.add_argument("--baseline-seconds", type=float, default=2.0, help="unloaded tactile-baseline capture duration after command b")
    parser.add_argument(
        "--tactile-health-mask",
        type=Path,
        default=None,
        help="reviewed device-specific bad-row/column JSON",
    )
    parser.add_argument("--no-save-mp4", action="store_true", help="do not write ego/main RGB MP4 sidecars after each successful episode")
    parser.add_argument("--fast-save", dest="fast_save", action="store_true", default=True, help="skip post-hoc quality analysis at e (default)")
    parser.add_argument("--full-save", dest="fast_save", action="store_false", help="enable post-hoc quality analysis at e")
    parser.add_argument("--mp4-codec", default="mp4v", help="fourcc used for MP4 sidecars (default: mp4v)")
    parser.add_argument("--min-ego-sharpness", type=float, default=70.0, help="refuse if ego RGB Laplacian-variance p05 is lower")
    parser.add_argument("--no-wavletech-emg", "--no-emg", dest="no_wavletech_emg",
                        action="store_true", help="disable the Wavletech serial EMG receiver")
    parser.add_argument(
        "--wavletech-python", "--emg-python", dest="wavletech_python",
        default=str(Path(__file__).resolve().parent / ".venv" / "bin" / "python"),
        help="Python containing pyserial (defaults to this project's .venv)",
    )
    parser.add_argument("--wavletech-tty", "--emg-tty", dest="wavletech_tty",
                        default="/dev/ttyUSB0", help="Wavletech USB serial receiver")
    parser.add_argument("--wavletech-baud", "--emg-baud", dest="wavletech_baud",
                        type=int, default=921600, help="receiver baud rate from the device manual")
    parser.add_argument("--max-wavletech-emg-age-ms", type=float, default=100.0,
                        help="maximum |Wavletech EMG-RGB| nearest-sample alignment age")
    parser.add_argument("--min-wavletech-emg-hz", type=float, default=1950.0,
                        help="minimum accepted Wavletech native EMG sample rate")
    parser.add_argument("--max-wavletech-emg-missing-ratio", type=float, default=0.02,
                        help="maximum Wavletech packet sequence-gap ratio")
    parser.add_argument("--max-wavletech-emg-gap-ms", type=float, default=350.0,
                        help="Wavletech outage threshold for discarding the active episode")
    parser.add_argument("--wavletech-start-wait-s", type=float, default=20.0)
    parser.add_argument("--wavletech-silence-timeout-s", type=float, default=0.35)
    parser.add_argument("--no-myo", action="store_true", help="disable the original Myo/BLED112 EMG chain")
    parser.add_argument(
        "--myo-python",
        default=str(Path(__file__).resolve().parent / ".venv" / "bin" / "python"),
        help="Python containing pyomyo (defaults to this project's .venv)",
    )
    parser.add_argument(
        "--myo-tty",
        default="/dev/serial/by-id/usb-Bluegiga_Low_Energy_Dongle_1-if00",
        help="Bluegiga BLED112 serial port",
    )
    parser.add_argument("--myo-mac", default="auto", help="Myo MAC address or auto")
    parser.add_argument("--max-emg-age-ms", type=float, default=100.0,
                        help="maximum |Myo EMG-RGB| nearest-sample alignment age")
    parser.add_argument("--min-emg-hz", type=float, default=180.0,
                        help="minimum accepted Myo native EMG sample rate")
    parser.add_argument("--max-emg-missing-ratio", type=float, default=0.02,
                        help="maximum Myo callback sequence-gap ratio")
    parser.add_argument("--max-emg-gap-ms", type=float, default=350.0,
                        help="Myo outage threshold for discarding the active episode")
    parser.add_argument("--myo-start-wait-s", type=float, default=20.0)
    parser.add_argument("--myo-silence-timeout-s", type=float, default=0.35)
    args = parser.parse_args()
    if not 0 < args.glove_hz <= 120:
        parser.error("--glove-hz 必须在 (0, 120]")
    if min(args.max_glove_age_ms, args.max_wrist_pose_age_ms, args.max_gemini_age_ms, args.max_camera_pose_age_ms, args.max_tracker_age_ms, args.max_rgb_gap_ms, args.max_wuji_gap_ms, args.max_joint_step_rad, args.joint_preflight_seconds, args.joint_preflight_min_range_deg, args.baseline_seconds, args.min_ego_sharpness, args.max_emg_age_ms, args.min_emg_hz, args.max_emg_gap_ms, args.myo_start_wait_s, args.myo_silence_timeout_s, args.max_wavletech_emg_age_ms, args.min_wavletech_emg_hz, args.max_wavletech_emg_gap_ms, args.wavletech_start_wait_s, args.wavletech_silence_timeout_s, args.wavletech_baud) <= 0:
        parser.error("质量阈值的毫秒/弧度值必须为正")
    if not 0 <= args.max_missing_ratio < 1:
        parser.error("质量比例阈值必须在 [0, 1)")
    if not 0 <= args.max_align_drop_ratio < 1:
        parser.error("--max-align-drop-ratio 必须在 [0, 1)")
    if not 0 <= args.max_emg_missing_ratio < 1:
        parser.error("--max-emg-missing-ratio 必须在 [0, 1)")
    if not 0 <= args.max_wavletech_emg_missing_ratio < 1:
        parser.error("--max-wavletech-emg-missing-ratio 必须在 [0, 1)")
    if len(args.mp4_codec) != 4:
        parser.error("--mp4-codec 必须是四字符 fourcc，例如 mp4v")
    if (
        not args.no_myo
        and not args.no_wavletech_emg
        and Path(args.myo_tty).exists()
        and Path(args.wavletech_tty).exists()
        and Path(args.myo_tty).resolve() == Path(args.wavletech_tty).resolve()
    ):
        parser.error(
            "Myo 与 Wavletech 不能使用同一个物理串口；"
            "Myo 应连接 Bluegiga BLED112，Wavletech 应连接 QinHeng USB Serial"
        )

    try:
        health_mask = (load_tactile_health_mask(args.tactile_health_mask, glove_sn=args.glove_sn)
                       if args.tactile_health_mask is not None else None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"invalid --tactile-health-mask: {exc}")
    _configure_wuji_sdk_logging()
    print(
        f"Wuji health mask: rows={list(health_mask.rows)}, cols={list(health_mask.cols)}, "
        f"grid cells={int(health_mask.mask.sum())}; raw audit=-1, training tactile=0"
    )

    namespace = args.camera_namespace.strip("/")
    camera_name = args.camera_name.strip("/")
    prefix = "/" + "/".join(part for part in (namespace, camera_name) if part)
    ego_topic = args.ego_topic.strip() or f"{prefix}/color/image_raw"
    print("[INIT] 准备 Wuji、ROS、Myo 与 Wavletech 数据源...", flush=True)
    glove = WujiGloveSource(
        args.glove_sn,
        args.glove_hz,
        capture_skeleton=True,
        capture_zones=True,
        capture_right_wrist_pose=True,
    )
    emg = None if args.no_wavletech_emg else SerialEmgSource(
        args.wavletech_python, args.wavletech_tty, baud=args.wavletech_baud,
        silence_timeout_s=args.wavletech_silence_timeout_s,
    )
    myo = None if args.no_myo else MyoEmgSource(
        args.myo_python, args.myo_tty, args.myo_mac,
        silence_timeout_s=args.myo_silence_timeout_s,
    )
    camera = D435Node(
        ego_topic,
        f"{prefix}/aligned_depth_to_color/image_raw",
        f"{prefix}/color/camera_info",
        f"{prefix}/aligned_depth_to_color/camera_info",
        capture_depth=False,
        capture_ego=not args.no_ego_camera,
        gemini_topic=args.gemini_topic,
        gemini_imu_topic=args.gemini_imu_topic,
        tracker_camera_topic=args.tracker_camera_topic,
        tracker_wrist_topic=args.tracker_wrist_topic,
        tracker_camera_z_offset_m=args.tracker_camera_z_offset_m,
        ego_compressed=args.ego_compressed,
        ego_require_camera_info=not args.ego_compressed,
        ego_source=args.ego_source,
    )
    collector: FTP1Collector | None = None
    try:
        if emg is not None:
            print(
                f"[INIT] 连接 Wavletech 串口接收器（{args.wavletech_tty} @ {args.wavletech_baud}）...",
                flush=True,
            )
            emg.start(wait_for_stability=False)
            print("[INIT] 串口 EMG 后台连接已启动；连接失败会自动重试，采集器不会退出", flush=True)
        else:
            print("[SKIP] Wavletech 串口 EMG 已禁用", flush=True)
        if myo is not None:
            print(f"[INIT] 连接 Myo（{args.myo_tty}, MAC={args.myo_mac}）...", flush=True)
            myo.start(wait_for_stability=False)
            print("[INIT] Myo 后台连接已启动；连接失败会自动重试", flush=True)
        else:
            print("[SKIP] Myo EMG 已禁用", flush=True)
        print(f"[INIT] 连接 Wuji 手套 {args.glove_sn}...", flush=True)
        glove.start()
        print("[PASS] Wuji 手套数据源已启动", flush=True)
        collector = FTP1Collector(
            args.output_dir,
            glove,
            camera,
            emg,
            myo,
            max_glove_age_ms=args.max_glove_age_ms,
            max_wrist_pose_age_ms=args.max_wrist_pose_age_ms,
            max_gemini_age_ms=args.max_gemini_age_ms,
            max_camera_pose_age_ms=args.max_camera_pose_age_ms,
            max_rgb_gap_ms=args.max_rgb_gap_ms,
            max_missing_ratio=args.max_missing_ratio,
            max_align_drop_ratio=args.max_align_drop_ratio,
            max_wuji_gap_ms=args.max_wuji_gap_ms,
            max_joint_step_rad=args.max_joint_step_rad,
            baseline_seconds=args.baseline_seconds,
            save_mp4=not args.no_save_mp4,
            fast_save=args.fast_save,
            mp4_codec=args.mp4_codec,
            min_ego_sharpness=args.min_ego_sharpness,
            max_emg_age_ms=args.max_wavletech_emg_age_ms,
            min_emg_hz=args.min_wavletech_emg_hz,
            max_emg_missing_ratio=args.max_wavletech_emg_missing_ratio,
            max_emg_gap_ms=args.max_wavletech_emg_gap_ms,
            emg_start_wait_s=args.wavletech_start_wait_s,
            max_myo_age_ms=args.max_emg_age_ms,
            min_myo_hz=args.min_emg_hz,
            max_myo_missing_ratio=args.max_emg_missing_ratio,
            max_myo_gap_ms=args.max_emg_gap_ms,
            myo_start_wait_s=args.myo_start_wait_s,
            max_tracker_age_ms=args.max_tracker_age_ms,
            joint_preflight_seconds=args.joint_preflight_seconds,
            joint_preflight_min_range_deg=args.joint_preflight_min_range_deg,
            tactile_health_mask=health_mask,
        )
        print("=" * 78)
        print(
            "Wuji glove + " + ("Myo EMG + " if myo is not None else "")
            + ("Wavletech serial EMG/IMU + " if emg is not None else "")
            + (f"{args.ego_source}(ego) + " if not args.no_ego_camera else "")
            + "Gemini(main) -> FTP-1 standardized Zarr"
        )
        print("output:", args.output_dir.expanduser())
        print("commands: b=无接触触觉基线  s=开始  e=保存成功 episode  r=立即拒收  d=丢弃  status=状态  q=退出（p=可选规范确认）")
        print("=" * 78)
        episode_layout_selection = os.environ.get("WUJI_EPISODE_LAYOUT_SELECTION", "") == "1"
        last_participant = ""
        retry_selected_layout = False
        while True:
            try:
                command = input("ftp1-glove> ").strip().lower()
            except EOFError:
                print()
                break
            if command == "b":
                collector.capture_baseline()
            elif command == "p":
                collector.confirm_protocol()
            elif command == "s":
                if episode_layout_selection:
                    if retry_selected_layout:
                        print(
                            f"[RETRY] 沿用 人员={collector.participant_name}，task={collector.task_id}，"
                            f"条件={collector.glove_condition}"
                        )
                        retry_selected_layout = not collector.start(f"任务 {collector.task_id}")
                        continue
                    name_prompt = "被采集人员姓名"
                    if last_participant:
                        name_prompt += f"（回车沿用 {last_participant}）"
                    participant = input(name_prompt + ": ").strip() or last_participant
                    glove_choice = input("手套条件 [1=有手套, 2=无手套]: ").strip()
                    task_choice = input(
                        "task 编号 [正整数，例如 15=t15]: "
                    ).strip()
                    condition_by_choice = {"1": "有手套", "2": "无手套"}
                    if glove_choice not in condition_by_choice:
                        print("[未开始] 手套条件只需输入 1 或 2")
                        continue
                    if (
                        not task_choice.isdecimal()
                        or int(task_choice) < 1
                        or task_choice != str(int(task_choice))
                    ):
                        print("[未开始] task 请输入正整数，例如 1 或 15")
                        continue
                    try:
                        collector.select_episode_layout(
                            participant,
                            f"t{task_choice}",
                            condition_by_choice[glove_choice],
                        )
                    except ValueError as exc:
                        print(f"[未开始] {exc}")
                        continue
                    last_participant = participant
                    print(
                        f"[EPISODE] 人员={collector.participant_name}，task={collector.task_id}，"
                        f"条件={collector.glove_condition}\n"
                        f"          保存目录={collector.output_dir}"
                    )
                    retry_selected_layout = not collector.start(f"任务 {collector.task_id}")
                else:
                    collector.start(input("任务描述: "))
            elif command == "e":
                collector.save()
            elif command == "d":
                collector.discard()
            elif command in {"r", "refuse"}:
                collector.refuse()
            elif command == "status":
                collector.status()
            elif command == "q":
                break
            elif command:
                print("未知命令：b | p | s | e | r | d | status | q")
    except KeyboardInterrupt:
        print()
    finally:
        if collector is not None:
            collector.close()
        camera.close()
        glove.close()
        if collector is None and emg is not None:
            emg.close()
        if collector is None and myo is not None:
            myo.close()


if __name__ == "__main__":
    main()
