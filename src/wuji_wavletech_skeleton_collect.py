#!/usr/bin/env python3
"""Collect Wavletech EMG and Wuji glove skeleton into episode Zarr stores.

The Wuji 21x3 MediaPipe skeleton is the aligned (training) time axis.  Every
native Wavletech EMG sample, its transport timestamps, and the receiver IMU
are retained under ``streams`` so alignment never destroys source data.
"""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import argparse
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from wuji_glove_d435_collect import DEFAULT_GLOVE_SN, WujiGloveSource
from wuji_serial_emg_source import SerialEmgSource


FORMAT = "wuji_wavletech_emg_skeleton_zarr_v1"
EMG_CHANNELS = 8
SKELETON_SHAPE = (21, 3)


def configure_wuji_logging() -> None:
    """Keep periodic SDK clock/status messages out of the collection console."""
    level = os.environ.get("WUJI_SDK_LOG_LEVEL", "error").strip()
    if not level:
        return
    try:
        from wuji_sdk import set_log_level

        set_log_level(level)
    except (ImportError, ValueError, RuntimeError) as exc:
        print(f"[WARN] 无法设置 wuji_sdk 日志级别为 {level!r}: {exc}")


class TimedWujiGloveSource(WujiGloveSource):
    """Timestamp and health-check only the requested skeleton stream."""

    def _append(self, queue, item, kind: str) -> None:
        if kind == "skeleton":
            item = (*item, time.time_ns())
        super()._append(queue, item, kind)

    def _transport_stale(self, now: float) -> bool:
        with self.lock:
            stamp = self.last_skeleton_mono
            started = self.connection_started_mono
        return bool(started and now - (stamp if stamp >= started else started) >= self.reconnect_after_s)

    def healthy(self, max_age_s: float = 0.25) -> bool:
        with self.lock:
            return bool(
                self.last_skeleton_mono > 0.0
                and time.monotonic() - self.last_skeleton_mono <= max_age_s
                and self.device is not None
            )


def rate_hz(timestamps_ns: np.ndarray) -> float:
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    if len(timestamps_ns) < 2:
        return 0.0
    duration = (int(timestamps_ns[-1]) - int(timestamps_ns[0])) / 1e9
    return float((len(timestamps_ns) - 1) / max(duration, 1e-9))


def nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest source indices and signed source-minus-target ages."""
    source_ns = np.asarray(source_ns, dtype=np.int64)
    target_ns = np.asarray(target_ns, dtype=np.int64)
    if source_ns.ndim != 1 or not len(source_ns):
        raise ValueError("empty source timeline")
    right = np.clip(np.searchsorted(source_ns, target_ns, side="left"), 0, len(source_ns) - 1)
    left = np.clip(right - 1, 0, len(source_ns) - 1)
    use_left = np.abs(source_ns[left] - target_ns) <= np.abs(source_ns[right] - target_ns)
    indices = np.where(use_left, left, right).astype(np.int64)
    return indices, (source_ns[indices] - target_ns).astype(np.int64)


def corrected_emg_timestamps_ns(
    frames: list[tuple[int, int, int, np.ndarray, int]],
) -> tuple[np.ndarray, int]:
    """Map the reconstructed EMG sample clock onto host wall-clock time."""
    if not frames:
        return np.empty(0, dtype=np.int64), 0
    raw = np.asarray([frame[0] for frame in frames], dtype=np.int64)
    arrival = np.asarray([frame[4] for frame in frames], dtype=np.int64)
    residual = arrival - raw
    if len(raw) < 40 or raw[-1] <= raw[0]:
        rank = min(max(int(len(residual) * 0.10), 0), len(residual) - 1)
        offset_ns = int(np.partition(residual, rank)[rank])
        return raw + offset_ns, offset_ns

    anchors_x: list[float] = []
    anchors_y: list[float] = []
    for indices in np.array_split(np.arange(len(raw)), min(20, max(2, len(raw) // 100))):
        if not len(indices):
            continue
        local = residual[indices]
        low_latency = indices[local <= np.quantile(local, 0.20)]
        anchors_x.append(float(np.median(raw[low_latency] - raw[0])))
        anchors_y.append(float(np.median(residual[low_latency])))
    slope, intercept = np.polyfit(np.asarray(anchors_x), np.asarray(anchors_y), 1)
    slope = float(np.clip(slope, -0.05, 0.05))
    if abs(slope) < 1e-4:
        slope = 0.0
    relative = np.rint((raw - raw[0]).astype(np.float64) * (1.0 + slope) + intercept)
    corrected = raw[0] + relative.astype(np.int64)
    if np.any(np.diff(corrected) <= 0):
        raise ValueError("corrected EMG timestamps are not strictly increasing")
    return corrected, int(round(intercept))


def corrected_glove_timestamps_ns(
    frames: list[tuple[int, int, np.ndarray] | tuple[int, int, np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Map the Wuji SDK clock to host time using its lower arrival envelope."""
    source = np.asarray([frame[0] for frame in frames], dtype=np.int64) * 1_000
    arrival = np.asarray(
        [frame[3] if len(frame) >= 4 else frame[0] * 1_000 for frame in frames],
        dtype=np.int64,
    )
    if np.any(np.diff(source) <= 0):
        raise ValueError("Wuji 骨架时间戳不是严格递增")
    relative = (source - source[0]).astype(np.float64)
    residual = (arrival - source).astype(np.float64)
    anchors_x: list[float] = []
    anchors_y: list[float] = []
    second_bins = ((source - source[0]) // 1_000_000_000).astype(np.int64)
    for second in np.unique(second_bins):
        indices = np.flatnonzero(second_bins == second)
        local = residual[indices]
        low_latency = indices[local <= np.quantile(local, 0.15)]
        anchors_x.append(float(np.median(relative[low_latency])))
        anchors_y.append(float(np.median(residual[low_latency])))
    slope = 0.0
    if len(anchors_x) >= 4:
        slope = float(np.polyfit(np.asarray(anchors_x), np.asarray(anchors_y), 1)[0])
        slope = float(np.clip(slope, -0.005, 0.005))
    offset = int(round(float(np.quantile(residual - slope * relative, 0.05))))
    mapped = source + np.rint(offset + slope * relative).astype(np.int64)
    # A decoded sample cannot exist after its host enqueue timestamp.
    mapped = np.minimum(mapped, arrival)
    if np.any(np.diff(mapped) <= 0):
        raise ValueError("Wuji 映射后的主机时间戳不是严格递增")
    return mapped, arrival, offset, slope


@dataclass(frozen=True)
class EpisodeData:
    data: dict[str, np.ndarray]
    audit: dict[str, np.ndarray]
    streams: dict[str, np.ndarray]
    attrs: dict[str, object]


def build_episode(
    skeleton_frames: list[tuple[int, int, np.ndarray] | tuple[int, int, np.ndarray, int]],
    emg_frames: list[tuple[int, int, int, np.ndarray, int]],
    imu_frames: list[tuple[int, int, np.ndarray, int]],
    *,
    instruction: str,
    max_alignment_age_ms: float,
) -> EpisodeData:
    """Validate and align one in-memory capture without accessing hardware."""
    if len(skeleton_frames) < 2:
        raise ValueError("Wuji 骨架不足 2 帧")
    if len(emg_frames) < 2:
        raise ValueError("Wavletech EMG 不足 2 帧")

    skeleton = np.stack([frame[2] for frame in skeleton_frames]).astype(np.float32)
    if skeleton.shape[1:] != SKELETON_SHAPE or not np.all(np.isfinite(skeleton)):
        raise ValueError(f"Wuji 骨架必须为有限值 (T,{SKELETON_SHAPE[0]},3)，得到 {skeleton.shape}")
    skeleton_ts_us = np.asarray([frame[0] for frame in skeleton_frames], dtype=np.int64)
    skeleton_ts_ns_all, skeleton_arrival_ns, glove_clock_offset_ns, glove_clock_slope = (
        corrected_glove_timestamps_ns(skeleton_frames)
    )
    skeleton_ts_ns = skeleton_ts_ns_all.copy()

    emg = np.stack([frame[3] for frame in emg_frames]).astype(np.int32)
    if emg.shape != (len(emg_frames), EMG_CHANNELS):
        raise ValueError(f"Wavletech EMG 必须为 (N,{EMG_CHANNELS})，得到 {emg.shape}")
    emg_ts_ns, clock_offset_ns = corrected_emg_timestamps_ns(emg_frames)
    emg_raw_ts_ns = np.asarray([frame[0] for frame in emg_frames], dtype=np.int64)
    emg_arrival_ns = np.asarray([frame[4] for frame in emg_frames], dtype=np.int64)
    emg_seq = np.asarray([frame[1] for frame in emg_frames], dtype=np.int64)

    # Only aligned rows inside the true common time span are training data.
    overlap = (skeleton_ts_ns >= emg_ts_ns[0]) & (skeleton_ts_ns <= emg_ts_ns[-1])
    if not np.any(overlap):
        raise ValueError("Wuji 骨架与 Wavletech EMG 没有重叠时间段")
    skeleton = skeleton[overlap]
    skeleton_ts_us = skeleton_ts_us[overlap]
    skeleton_ts_ns = skeleton_ts_ns[overlap]
    skeleton_seq = np.asarray([frame[1] for frame in skeleton_frames], dtype=np.int64)[overlap]

    emg_idx, emg_age_ns = nearest_indices(emg_ts_ns, skeleton_ts_ns)
    valid = np.abs(emg_age_ns) <= int(max_alignment_age_ms * 1e6)
    if not np.any(valid):
        raise ValueError(f"没有 EMG 对齐年龄 <= {max_alignment_age_ms:g} ms 的骨架帧")
    skeleton = skeleton[valid]
    skeleton_ts_us = skeleton_ts_us[valid]
    skeleton_ts_ns = skeleton_ts_ns[valid]
    skeleton_seq = skeleton_seq[valid]
    emg_idx = emg_idx[valid]
    emg_age_ns = emg_age_ns[valid]

    width = max(1, len(instruction))
    data = {
        "timestamps": skeleton_ts_ns,
        "wuji_hand_skeleton_mediapipe": skeleton,
        "right_forearm_emg_wavletech": emg[emg_idx],
        "sub_task_instruction": np.full(len(skeleton), instruction, dtype=f"<U{width}"),
    }
    audit = {
        "wuji_hand_skeleton_mediapipe": skeleton,
        "wuji_hand_skeleton_source_timestamp_us": skeleton_ts_us,
        "wuji_hand_skeleton_mapped_timestamp_ns": skeleton_ts_ns,
        "wuji_hand_skeleton_source_seq": skeleton_seq,
        "right_forearm_emg_wavletech_source_timestamp_ns": emg_ts_ns[emg_idx],
        "right_forearm_emg_wavletech_source_seq": emg_seq[emg_idx],
        "right_forearm_emg_wavletech_age_us": (emg_age_ns // 1_000).astype(np.int64),
    }
    streams = {
        "wuji_hand_skeleton_raw": np.stack([frame[2] for frame in skeleton_frames]).astype(np.float32),
        "wuji_hand_skeleton_timestamp_us": np.asarray([frame[0] for frame in skeleton_frames], dtype=np.int64),
        "wuji_hand_skeleton_mapped_timestamp_ns": skeleton_ts_ns_all,
        "wuji_hand_skeleton_host_enqueue_timestamp_ns": skeleton_arrival_ns,
        "wuji_hand_skeleton_seq": np.asarray([frame[1] for frame in skeleton_frames], dtype=np.int64),
        "right_forearm_emg_wavletech_raw": emg,
        "right_forearm_emg_wavletech_timestamp_ns": emg_ts_ns,
        "right_forearm_emg_wavletech_clock_raw_timestamp_ns": emg_raw_ts_ns,
        "right_forearm_emg_wavletech_arrival_timestamp_ns": emg_arrival_ns,
        "right_forearm_emg_wavletech_seq": emg_seq,
    }
    if imu_frames:
        streams.update({
            "right_forearm_imu_wavletech_raw": np.stack([frame[2] for frame in imu_frames]).astype(np.float32),
            "right_forearm_imu_wavletech_timestamp_ns": np.asarray([frame[0] for frame in imu_frames], dtype=np.int64),
            "right_forearm_imu_wavletech_seq": np.asarray([frame[1] for frame in imu_frames], dtype=np.int64),
            "right_forearm_imu_wavletech_arrival_timestamp_ns": np.asarray([frame[3] for frame in imu_frames], dtype=np.int64),
        })

    missing = int(np.maximum(np.diff(emg_seq) - 1, 0).sum())
    max_emg_gap_ns = int(np.diff(emg_ts_ns).max())
    max_skeleton_gap_ns = int(np.diff(np.asarray(streams["wuji_hand_skeleton_timestamp_us"])) .max()) * 1_000
    attrs: dict[str, object] = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "success": True,
        "instruction": instruction,
        "canonical_time_axis": "native Wuji hand_skeleton timestamps; no interpolation",
        "canonical_hz": rate_hz(skeleton_ts_ns),
        "canonical_sample_count": len(skeleton_ts_ns),
        "alignment_rows_dropped": int(len(skeleton_frames) - len(skeleton_ts_ns)),
        "alignment_max_abs_age_ms": float(np.abs(emg_age_ns).max(initial=0) / 1e6),
        "right_forearm_emg_wavletech_layout": "8 signed 24-bit channels; nearest native sample per skeleton row",
        "right_forearm_emg_wavletech_unit": "microvolt",
        "right_forearm_emg_wavletech_native_hz": rate_hz(emg_ts_ns),
        "right_forearm_emg_wavletech_native_sample_count": len(emg_ts_ns),
        "right_forearm_emg_wavletech_missing_count": missing,
        "right_forearm_emg_wavletech_missing_ratio": missing / max(len(emg_seq) + missing, 1),
        "right_forearm_emg_wavletech_max_gap_ms": max_emg_gap_ns / 1e6,
        "right_forearm_emg_wavletech_clock_offset_ms": clock_offset_ns / 1e6,
        "right_forearm_emg_wavletech_timestamp_semantics": "host serial packet arrival timeline with affine source/arrival correction; raw and arrival timestamps retained",
        "source_glove_skeleton_hz": rate_hz(np.asarray(streams["wuji_hand_skeleton_timestamp_us"]) * 1_000),
        "source_glove_skeleton_sample_count": len(skeleton_frames),
        "source_glove_skeleton_max_gap_ms": max_skeleton_gap_ns / 1e6,
        "source_glove_skeleton_clock_offset_ms": glove_clock_offset_ns / 1e6,
        "source_glove_skeleton_clock_rate_correction_ppm": glove_clock_slope * 1e6,
        "source_glove_skeleton_timestamp_semantics": "Wuji SDK timestamp mapped to host time from the lower host-enqueue latency envelope; raw SDK and host enqueue timestamps retained",
        "wuji_hand_skeleton_layout": "MediaPipe 21 landmarks in wrist frame; SDK documented order",
        "wuji_hand_skeleton_unit": "metre",
        "right_forearm_imu_wavletech_layout": "[gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z]",
        "right_forearm_imu_wavletech_unit": "[rad/s,rad/s,rad/s,m/s^2,m/s^2,m/s^2]",
    }
    return EpisodeData(data=data, audit=audit, streams=streams, attrs=attrs)


def _create_array(group: zarr.Group, name: str, value: np.ndarray) -> None:
    value = np.asarray(value)
    rows = max(1, min(len(value), 2048)) if value.ndim else 1
    chunks = (rows, *value.shape[1:]) if value.ndim else None
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    group.create_dataset(
        name, data=value, shape=value.shape, chunks=chunks,
        dtype=value.dtype, compressor=compressor,
    )


def write_episode(path: Path, episode: EpisodeData) -> None:
    """Atomically write a complete episode directory."""
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        root = zarr.open_group(str(temporary), mode="w")
        for group_name, arrays in (
            ("data", episode.data), ("audit", episode.audit), ("streams", episode.streams)
        ):
            group = root.create_group(group_name)
            for name, value in arrays.items():
                _create_array(group, name, value)
        meta = root.create_group("meta")
        _create_array(meta, "episode_ends", np.asarray([len(episode.data["timestamps"])], dtype=np.int64))
        root.attrs.update(episode.attrs)
        # Compatibility sidecar for the existing EMG-to-skeleton training
        # package. It keeps both native-rate streams on one shared origin.
        emg_ns = episode.streams["right_forearm_emg_wavletech_timestamp_ns"]
        pose_ns = episode.streams["wuji_hand_skeleton_mapped_timestamp_ns"]
        common_start = max(int(emg_ns[0]), int(pose_ns[0]))
        common_end = min(int(emg_ns[-1]), int(pose_ns[-1]))
        emg_keep = (emg_ns >= common_start) & (emg_ns <= common_end)
        pose_keep = (pose_ns >= common_start) & (pose_ns <= common_end)
        np.savez_compressed(
            temporary / "calibration.npz",
            emg=episode.streams["right_forearm_emg_wavletech_raw"][emg_keep],
            emg_time_seconds=(emg_ns[emg_keep] - common_start).astype(np.float64) / 1e9,
            raw_glove_skeleton=episode.streams["wuji_hand_skeleton_raw"][pose_keep],
            pose_time_seconds=(pose_ns[pose_keep] - common_start).astype(np.float64) / 1e9,
            original_emg_time_seconds=emg_ns[emg_keep].astype(np.float64) / 1e9,
            original_pose_time_seconds=pose_ns[pose_keep].astype(np.float64) / 1e9,
            time_origin_seconds=np.asarray(common_start / 1e9),
            units=np.asarray("m"),
        )
        temporary.rename(path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class Collector:
    def __init__(
        self,
        output_dir: Path,
        glove: WujiGloveSource,
        emg: SerialEmgSource,
        *,
        max_alignment_age_ms: float,
        min_emg_hz: float,
        max_emg_missing_ratio: float,
        max_emg_gap_ms: float,
        max_skeleton_gap_ms: float,
    ) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.glove = glove
        self.emg = emg
        self.max_alignment_age_ms = max_alignment_age_ms
        self.min_emg_hz = min_emg_hz
        self.max_emg_missing_ratio = max_emg_missing_ratio
        self.max_emg_gap_ms = max_emg_gap_ms
        self.max_skeleton_gap_ms = max_skeleton_gap_ms
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.recording = False
        self.instruction = ""
        self.started_ns = 0
        self.skeleton_frames: list[
            tuple[int, int, np.ndarray] | tuple[int, int, np.ndarray, int]
        ] = []
        self.emg_frames: list[tuple[int, int, int, np.ndarray, int]] = []
        self.imu_frames: list[tuple[int, int, np.ndarray, int]] = []
        self.faults: list[str] = []
        self.emg_connection_id = 0
        self.emg_interruption_id = 0
        self.glove_connection_count = 0
        self.start_dropped = (0, 0)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _drain_locked(self) -> None:
        # WujiGloveSource currently shares angle/tactile transport with the
        # skeleton. Drain those queues, but deliberately do not save them.
        self.glove.drain()
        skeleton = self.glove.drain_skeleton()
        emg = self.emg.drain()
        imu = self.emg.drain_imu()
        if self.recording:
            self.skeleton_frames.extend(skeleton)
            self.emg_frames.extend(emg)
            self.imu_frames.extend(imu)
            if self.emg.current_connection_id() != self.emg_connection_id:
                self.faults.append("Wavletech EMG 在 episode 中重连")
            interruption_id, reason = self.emg.interruption_snapshot()
            if interruption_id != self.emg_interruption_id:
                self.faults.append(f"Wavletech EMG 中断: {reason}")
                self.emg_interruption_id = interruption_id
            if self.glove.connection_count != self.glove_connection_count:
                self.faults.append("Wuji 手套在 episode 中重连")
                self.glove_connection_count = self.glove.connection_count

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                self._drain_locked()
            self.stop_event.wait(0.005)

    def start(self, instruction: str) -> bool:
        if not self.glove.healthy(max_age_s=0.5):
            print("[未开始] Wuji 数据源未就绪:", self.glove.status())
            return False
        if not self.emg.healthy(max_age_s=0.5):
            print("[未开始] Wavletech EMG 未就绪:", self.emg.status())
            return False
        with self.lock:
            if self.recording:
                print("[未开始] 已在采集中")
                return False
            self._drain_locked()
            self.skeleton_frames = []
            self.emg_frames = []
            self.imu_frames = []
            self.faults = []
            self.instruction = instruction.strip()
            self.started_ns = time.time_ns()
            self.emg_connection_id = self.emg.current_connection_id()
            self.emg_interruption_id = self.emg.interruption_snapshot()[0]
            self.glove_connection_count = self.glove.connection_count
            self.start_dropped = (self.glove.skeleton_dropped, self.emg.dropped)
            self.recording = True
        print(f"[REC] 开始采集: {self.instruction or '(无任务标签)'}")
        return True

    def _next_path(self) -> Path:
        indices = []
        for path in self.output_dir.glob("episode_*.zarr"):
            try:
                indices.append(int(path.stem.split("_")[-1]))
            except ValueError:
                pass
        return self.output_dir / f"episode_{max(indices, default=-1) + 1:06d}.zarr"

    def save(self) -> Path | None:
        with self.lock:
            if not self.recording:
                print("[未保存] 当前没有采集中的 episode")
                return None
            self._drain_locked()
            self.recording = False
            skeleton = list(self.skeleton_frames)
            emg = list(self.emg_frames)
            imu = list(self.imu_frames)
            faults = list(dict.fromkeys(self.faults))
            dropped = (
                self.glove.skeleton_dropped - self.start_dropped[0],
                self.emg.dropped - self.start_dropped[1],
            )
            instruction = self.instruction
        if dropped[0] or dropped[1]:
            faults.append(f"采集队列溢出: skeleton={dropped[0]}, emg={dropped[1]}")
        try:
            episode = build_episode(
                skeleton, emg, imu, instruction=instruction,
                max_alignment_age_ms=self.max_alignment_age_ms,
            )
        except ValueError as exc:
            print(f"[SAVE REFUSED] {exc}")
            return None

        attrs = episode.attrs
        quality_faults = list(faults)
        if float(attrs["right_forearm_emg_wavletech_native_hz"]) < self.min_emg_hz:
            quality_faults.append(
                f"EMG 频率 {attrs['right_forearm_emg_wavletech_native_hz']:.1f} < {self.min_emg_hz:.1f} Hz"
            )
        if float(attrs["right_forearm_emg_wavletech_missing_ratio"]) > self.max_emg_missing_ratio:
            quality_faults.append(
                f"EMG 丢包率 {attrs['right_forearm_emg_wavletech_missing_ratio']:.2%} > {self.max_emg_missing_ratio:.2%}"
            )
        if float(attrs["right_forearm_emg_wavletech_max_gap_ms"]) > self.max_emg_gap_ms:
            quality_faults.append(
                f"EMG 最大间断 {attrs['right_forearm_emg_wavletech_max_gap_ms']:.1f} > {self.max_emg_gap_ms:.1f} ms"
            )
        if float(attrs["source_glove_skeleton_max_gap_ms"]) > self.max_skeleton_gap_ms:
            quality_faults.append(
                f"骨架最大间断 {attrs['source_glove_skeleton_max_gap_ms']:.1f} > {self.max_skeleton_gap_ms:.1f} ms"
            )
        if quality_faults:
            print("[SAVE REFUSED] 数据质量未通过:")
            for fault in quality_faults:
                print("  -", fault)
            print("  本轮未写盘；可用 d 清空，检查设备后重新采集。")
            return None

        attrs.update({
            "started_unix_ns": self.started_ns,
            "capture_duration_seconds": (time.time_ns() - self.started_ns) / 1e9,
            "glove_sn": self.glove.sn,
            "wavletech_tty": self.emg.resolved_tty,
            "wavletech_baud": self.emg.baud,
            "quality_gate_max_alignment_age_ms": self.max_alignment_age_ms,
            "quality_gate_min_emg_hz": self.min_emg_hz,
            "quality_gate_max_emg_missing_ratio": self.max_emg_missing_ratio,
            "quality_gate_max_emg_gap_ms": self.max_emg_gap_ms,
            "quality_gate_max_skeleton_gap_ms": self.max_skeleton_gap_ms,
        })
        path = self._next_path()
        write_episode(path, episode)
        print(
            f"[SAVE] {path}: skeleton={attrs['source_glove_skeleton_sample_count']} "
            f"({attrs['source_glove_skeleton_hz']:.1f} Hz), EMG="
            f"{attrs['right_forearm_emg_wavletech_native_sample_count']} "
            f"({attrs['right_forearm_emg_wavletech_native_hz']:.1f} Hz), "
            f"aligned={attrs['canonical_sample_count']}"
        )
        return path

    def discard(self) -> None:
        with self.lock:
            self.recording = False
            self.skeleton_frames = []
            self.emg_frames = []
            self.imu_frames = []
            self.faults = []
        print("[DROP] 当前 episode 已丢弃")

    def status(self) -> None:
        with self.lock:
            state = "采集中" if self.recording else "未采集"
            counts = (len(self.skeleton_frames), len(self.emg_frames), len(self.imu_frames))
            faults = list(dict.fromkeys(self.faults))
        print("Wuji      :", self.glove.status())
        print("Wavletech :", self.emg.status())
        print(f"episode   : {state}, skeleton={counts[0]}, emg={counts[1]}, imu={counts[2]}")
        if faults:
            print("faults    :", "; ".join(faults))

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def _positive(parser: argparse.ArgumentParser, **values: float) -> None:
    bad = [name for name, value in values.items() if value <= 0]
    if bad:
        parser.error("以下参数必须为正数: " + ", ".join(bad))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/wavletech_skeleton"))
    parser.add_argument("--glove-sn", default=DEFAULT_GLOVE_SN)
    parser.add_argument("--glove-hz", type=float, default=120.0)
    parser.add_argument("--wavletech-python", default=sys.executable)
    parser.add_argument("--wavletech-tty", default="/dev/ttyUSB0")
    parser.add_argument("--wavletech-baud", type=int, default=921600)
    parser.add_argument("--wavletech-silence-timeout-s", type=float, default=0.35)
    parser.add_argument("--start-wait-s", type=float, default=20.0)
    parser.add_argument("--min-emg-hz", type=float, default=1950.0)
    parser.add_argument("--max-emg-missing-ratio", type=float, default=0.02)
    parser.add_argument("--max-emg-gap-ms", type=float, default=350.0)
    parser.add_argument("--max-skeleton-gap-ms", type=float, default=500.0)
    parser.add_argument("--max-alignment-age-ms", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=0.0,
                        help="non-interactive: record one episode for this duration")
    parser.add_argument("--instruction", default="", help="task label for non-interactive mode")
    args = parser.parse_args()
    _positive(
        parser, glove_hz=args.glove_hz, baud=args.wavletech_baud,
        silence=args.wavletech_silence_timeout_s, start_wait=args.start_wait_s,
        min_emg_hz=args.min_emg_hz, max_emg_gap=args.max_emg_gap_ms,
        max_skeleton_gap=args.max_skeleton_gap_ms, alignment=args.max_alignment_age_ms,
    )
    if args.glove_hz > 120:
        parser.error("--glove-hz 必须 <= 120")
    if not 0 <= args.max_emg_missing_ratio < 1:
        parser.error("--max-emg-missing-ratio 必须在 [0,1)")
    if args.duration_seconds < 0:
        parser.error("--duration-seconds 不能为负")

    configure_wuji_logging()
    glove = TimedWujiGloveSource(args.glove_sn, args.glove_hz, capture_skeleton=True)
    emg = SerialEmgSource(
        args.wavletech_python, args.wavletech_tty, baud=args.wavletech_baud,
        silence_timeout_s=args.wavletech_silence_timeout_s,
    )
    collector: Collector | None = None
    try:
        print("[INIT] 连接 Wuji 骨架与 Wavletech EMG...", flush=True)
        glove.start()
        emg.start(timeout_s=args.start_wait_s, wait_for_stability=False)
        deadline = time.monotonic() + args.start_wait_s
        while time.monotonic() < deadline:
            if glove.healthy(max_age_s=0.5) and emg.stable(
                duration_s=min(3.0, args.start_wait_s / 2), min_hz=args.min_emg_hz,
                max_age_s=args.wavletech_silence_timeout_s,
            ):
                break
            time.sleep(0.1)
        else:
            print("[FAILED] 数据源在等待时间内未稳定")
            print("Wuji      :", glove.status())
            print("Wavletech :", emg.status())
            return 2

        collector = Collector(
            args.output_dir, glove, emg,
            max_alignment_age_ms=args.max_alignment_age_ms,
            min_emg_hz=args.min_emg_hz,
            max_emg_missing_ratio=args.max_emg_missing_ratio,
            max_emg_gap_ms=args.max_emg_gap_ms,
            max_skeleton_gap_ms=args.max_skeleton_gap_ms,
        )
        print(f"[READY] 输出目录: {collector.output_dir}")
        if args.duration_seconds:
            if not collector.start(args.instruction):
                return 2
            deadline = time.monotonic() + args.duration_seconds
            while time.monotonic() < deadline:
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
            return 0 if collector.save() is not None else 3

        print("命令: s=开始, e=结束并保存, d=丢弃, status=状态, q=退出")
        while True:
            command = input("> ").strip().lower()
            if command == "s":
                collector.start(input("任务描述（可空）: "))
            elif command == "e":
                collector.save()
            elif command == "d":
                collector.discard()
            elif command == "status":
                collector.status()
            elif command == "q":
                if collector.recording:
                    print("[提示] 正在采集；先输入 e 保存或 d 丢弃")
                    continue
                break
            elif command:
                print("未知命令: s | e | d | status | q")
    except (KeyboardInterrupt, EOFError):
        print()
        if collector is not None and collector.recording:
            print("[DROP] 中断时的未完成 episode 不保存")
            collector.discard()
    finally:
        if collector is not None:
            collector.close()
        emg.close()
        glove.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
