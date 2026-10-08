#!/usr/bin/env python3
"""Record only a Wuji glove and a RealSense D435 into an episode Zarr store.

This recorder deliberately does not create a LinkerHand API object, ROS hand
subscriber, teleoperation process, or robot command.  The glove and camera are
kept as independent source-rate streams.  RGB-only is the default: the D415/D435
color camera can then run at 60 Hz while the Wuji tactile stream may be 120 Hz.
Storing a separate timestamp per stream avoids inventing duplicated camera
frames.

Prerequisite: start the official ROS D435 driver in a separate terminal.  The
default topics are produced by ``realsense2_camera/rs_launch.py`` with camera
name ``camera``.
"""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import argparse
import json
import math
import os
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr


GLOVE_SHAPE = (24, 31)
ANGLE_SHAPE = (5, 5)
DEFAULT_GLOVE_SN = "WG1KA06260622532"
# Wuji's current public tactile contract is a 24x31 grid with 526 physical
# taxels.  ``-1`` is the only valid geometry sentinel; never infer an extra
# mask from low/no pressure values.
WUJI_OFFICIAL_ACTIVE_TAXELS = 526
WUJI_TACTILE_ZONE_NAMES = ("thumb", "index", "middle", "ring", "pinky", "palm")


class TactileHealthMask:
    """Device-specific taxels excluded after a physical health diagnosis."""

    def __init__(self, rows: list[int], cols: list[int], *, source: str = ""):
        self.rows = tuple(sorted(set(map(int, rows))))
        self.cols = tuple(sorted(set(map(int, cols))))
        if any(row < 0 or row >= GLOVE_SHAPE[0] for row in self.rows):
            raise ValueError(f"health-mask row out of range: {self.rows}")
        if any(col < 0 or col >= GLOVE_SHAPE[1] for col in self.cols):
            raise ValueError(f"health-mask column out of range: {self.cols}")
        self.source = str(source)
        self.mask = np.zeros(GLOVE_SHAPE, dtype=bool)
        self.mask[self.rows, :] = True
        self.mask[:, self.cols] = True

    def apply_batch(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 3 or raw.shape[1:] != GLOVE_SHAPE:
            raise ValueError(f"expected (T,{GLOVE_SHAPE[0]},{GLOVE_SHAPE[1]}), got {raw.shape}")
        output = raw.copy()
        output[:, self.mask] = -1.0
        return output

    def metadata(self) -> dict[str, object]:
        return {
            "tactile_health_mask_applied": True,
            "tactile_health_mask_rows": list(self.rows),
            "tactile_health_mask_cols": list(self.cols),
            "tactile_health_mask_cell_count": int(self.mask.sum()),
            "tactile_health_mask_source": self.source,
            "tactile_health_mask_invalid_value": -1.0,
        }


def load_tactile_health_mask(path: str | Path, *, glove_sn: str | None = None) -> TactileHealthMask:
    """Load a reviewed health mask and reject a file for a different glove."""
    path = Path(path).expanduser()
    document = json.loads(path.read_text(encoding="utf-8"))
    declared_sn = str(document.get("glove_sn", "")).strip()
    if glove_sn and declared_sn and declared_sn != glove_sn:
        raise ValueError(f"health mask is for {declared_sn}, not connected glove {glove_sn}")
    return TactileHealthMask(
        list(document.get("bad_rows", [])) + list(document.get("weak_rows", [])),
        list(document.get("weak_cols", [])),
        source=f"{path.name}; {document.get('diagnosis', 'reviewed tactile health diagnosis')}",
    )


def _stamp_us(frame: Any) -> int:
    try:
        return int(frame.header.timestamp_us)
    except Exception:
        return 0


def _quaternion_to_rpy(rotation: Any) -> np.ndarray | None:
    """Convert an SDK quaternion to ROS-compatible intrinsic XYZ RPY radians."""
    try:
        x, y, z, w = (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w))
    except (AttributeError, TypeError, ValueError):
        return None
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-8:
        return None
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float32)


def _pose_stamped_to_xyz_rpy(msg: Any, *, z_offset_m: float = 0.0) -> np.ndarray | None:
    """Decode geometry_msgs/PoseStamped as metres + ROS XYZ intrinsic RPY."""
    try:
        position = np.asarray(
            (float(msg.pose.position.x), float(msg.pose.position.y), float(msg.pose.position.z) + float(z_offset_m)),
            dtype=np.float32,
        )
        rpy = _quaternion_to_rpy(msg.pose.orientation)
    except (AttributeError, TypeError, ValueError):
        return None
    if rpy is None or not np.all(np.isfinite(position)):
        return None
    return np.concatenate((position, rpy)).astype(np.float32)


def _right_wrist_pose(frame: Any) -> tuple[int, np.ndarray] | None:
    """Extract the Wuji dynamic ``waist -> r_wrist`` transform as xyz+rpy."""
    try:
        transforms = frame.transforms
    except AttributeError:
        return None
    for transform in transforms:
        if (
            str(getattr(transform, "parent_frame_id", "")) != "waist"
            or str(getattr(transform, "child_frame_id", "")) != "r_wrist"
        ):
            continue
        try:
            translation = np.asarray(transform.translation, dtype=np.float32).reshape(3)
            timestamp_us = int(transform.timestamp_us)
        except (AttributeError, TypeError, ValueError):
            continue
        rpy = _quaternion_to_rpy(getattr(transform, "rotation", None))
        if rpy is not None and np.all(np.isfinite(translation)):
            return timestamp_us, np.concatenate((translation, rpy)).astype(np.float32)
    return None


def _frame_seq(frame: Any, fallback: int) -> int:
    """Use a device sequence if the SDK exposes one, otherwise local order."""
    for owner in (getattr(frame, "header", None), frame):
        for name in ("sequence", "seq", "frame_id"):
            try:
                value = getattr(owner, name)
                if isinstance(value, (int, np.integer)):
                    return int(value)
            except Exception:
                pass
    return int(fallback)


def _glove_angles(frame: Any) -> np.ndarray | None:
    try:
        value = np.asarray(
            [list(finger.angles) for finger in frame.fingers], dtype=np.float32
        )
    except Exception:
        return None
    if value.shape != ANGLE_SHAPE or not np.all(np.isfinite(value)):
        return None
    return value


def _xyz(position: Any) -> np.ndarray | None:
    """Decode an SDK position object or a three-value sequence."""
    try:
        value = np.asarray([position.x, position.y, position.z], dtype=np.float32)
    except Exception:
        try:
            value = np.asarray(position, dtype=np.float32).reshape(3)
        except Exception:
            return None
    return value if np.all(np.isfinite(value)) else None


def _glove_skeleton(frame: Any) -> np.ndarray | None:
    """Return the documented 21-point MediaPipe skeleton in SDK list order."""
    try:
        joints = list(frame.joints)
    except Exception:
        return None
    if len(joints) != 21:
        return None
    points = [_xyz(joint.pose.position) for joint in joints]
    if any(point is None for point in points):
        return None
    value = np.stack(points).astype(np.float32)
    return value if value.shape == (21, 3) else None


def _glove_tactile(frame: Any) -> np.ndarray | None:
    try:
        value = np.asarray(frame.data, dtype=np.float32).reshape(GLOVE_SHAPE)
    except Exception:
        return None
    return value if np.all(np.isfinite(value)) else None


def _tactile_features(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve ``-1`` geometry sentinels in every stored tactile map."""
    valid = raw >= 0.0
    pressure = np.where(valid, raw, -1.0).astype(np.float32)
    values = raw[valid]
    summary = np.asarray(
        (
            int(valid.sum()),
            int(np.count_nonzero(values > 0.0)),
            float(values.sum()) if values.size else 0.0,
            float(values.max()) if values.size else 0.0,
        ),
        dtype=np.float32,
    )
    return valid.astype(np.uint8), pressure, summary


def tactile_active_taxel_count(raw: np.ndarray) -> int:
    """Return the number of physical taxels encoded by a raw 24x31 frame."""
    raw = np.asarray(raw)
    if raw.shape != GLOVE_SHAPE:
        raise ValueError(f"unexpected tactile shape {raw.shape}; expected {GLOVE_SHAPE}")
    return int(np.count_nonzero(raw >= 0.0))


def _glove_tactile_zones(frame: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return vendor semantic-zone [mean, max, sum] and valid-taxel counts.

    The full 24x31 map remains the lossless tactile source.  These six values
    are a second, semantic FTP-1 tactile group so a whole-hand matrix is not
    incorrectly labelled as a thumb-tip token.
    """
    stats: list[list[float]] = []
    counts: list[int] = []
    try:
        for name in WUJI_TACTILE_ZONE_NAMES:
            raw = np.asarray(getattr(frame, name), dtype=np.float32).reshape(-1)
            values = raw[np.isfinite(raw) & (raw >= 0.0)]
            counts.append(int(values.size))
            if values.size:
                stats.append([float(values.mean()), float(values.max()), float(values.sum())])
            else:
                stats.append([0.0, 0.0, 0.0])
    except Exception:
        return None
    return np.asarray(stats, dtype=np.float32), np.asarray(counts, dtype=np.int32)


def _ros_stamp_ns(header: Any) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _image_array(msg: Any, want_color: bool) -> np.ndarray:
    """Decode sensor_msgs/Image without cv_bridge (NumPy-2 safe)."""
    encoding = str(msg.encoding).lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if want_color:
        if encoding not in ("rgb8", "bgr8"):
            raise ValueError(f"unsupported color encoding {msg.encoding!r}; expected rgb8/bgr8")
        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
        image = rows[:, : w * 3].reshape(h, w, 3)
        if encoding == "bgr8":
            image = image[..., ::-1]
        return np.ascontiguousarray(image)

    if encoding == "16uc1":
        rows = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, step // 2)
        return np.ascontiguousarray(rows[:, :w])
    if encoding == "32fc1":
        rows = np.frombuffer(msg.data, dtype=np.float32).reshape(h, step // 4)
        # Standard D435 32FC1 is metres.  Preserve invalid/negative values as 0.
        return np.clip(rows[:, :w] * 1000.0, 0.0, 65535.0).astype(np.uint16)
    raise ValueError(f"unsupported depth encoding {msg.encoding!r}; expected 16UC1/32FC1")


def _compressed_image_array(msg: Any) -> np.ndarray:
    """Decode sensor_msgs/CompressedImage to contiguous RGB after capture."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV unavailable for Gemini MJPEG decode"
        ) from exc

    encoded = np.frombuffer(msg.data, dtype=np.uint8)

    if encoded.size == 0:
        raise ValueError("empty Gemini compressed frame")

    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("invalid Gemini JPEG frame")

    # OpenCV JPEG decode = BGR; FTP-1 camera_main_rgb = RGB.
    return np.ascontiguousarray(bgr[..., ::-1])


def _decode_compressed_item(item: tuple[Any, ...]) -> tuple[Any, ...]:
    """Decode the final message in a timestamp/sequence tuple."""
    return (*item[:-1], _compressed_image_array(item[-1]))


def _decode_compressed_messages(
    messages: list[tuple[Any, ...]],
    max_workers: int = 8,
) -> list[tuple[Any, ...]]:
    """Decode captured JPEG messages concurrently while preserving order."""
    if not messages:
        return []
    with ThreadPoolExecutor(max_workers=min(max(int(max_workers), 1), len(messages))) as executor:
        return list(executor.map(_decode_compressed_item, messages))


class _MappedFrameStore:
    """Disk-backed RGB frames used to keep episode-save memory bounded."""

    def __init__(self, path: Path, array: np.memmap):
        self.path = path
        self.array: np.memmap | None = array

    def close(self) -> None:
        array, self.array = self.array, None
        if array is not None:
            array.flush()
            del array
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __del__(self):
        # Save() has several quality-gate returns.  CPython normally releases
        # this local immediately; the fallback also removes a file on error.
        try:
            self.close()
        except Exception:
            pass


def _cleanup_stale_mapped_frame_stores(max_age_s: float = 24.0 * 3600.0) -> int:
    """Remove old mmap files left by a process killed during episode save."""
    cutoff = time.time() - max(float(max_age_s), 0.0)
    removed = 0
    try:
        candidates = Path(tempfile.gettempdir()).glob("wuji_rgb_*.mmap")
        for path in candidates:
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                removed += 1
            except (FileNotFoundError, OSError):
                continue
    except OSError:
        pass
    return removed


def _decode_messages_to_memmap(
    messages: list[tuple[Any, ...]],
    *,
    message_index: int,
    compressed: bool,
    batch_size: int = 32,
) -> _MappedFrameStore | None:
    """Decode image messages in small batches into a temporary mmap file.

    The old save path materialized every decoded RGB frame, then made another
    full copy with ``np.stack``.  A mmap keeps the same NumPy/Zarr interface
    while bounding resident decode memory to one small batch.
    """
    if not messages:
        return None

    def decode(item: tuple[Any, ...]) -> np.ndarray:
        value = item[message_index]
        return (
            _compressed_image_array(value)
            if compressed
            else _image_array(value, want_color=True)
        )

    first = decode(messages[0])
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"expected RGB frames, got {first.shape}")
    fd, raw_path = tempfile.mkstemp(prefix="wuji_rgb_", suffix=".mmap")
    os.close(fd)
    path = Path(raw_path)
    shape = (len(messages), *first.shape)
    try:
        array = np.memmap(path, mode="w+", dtype=np.uint8, shape=shape)
        array[0] = first
        batch_size = max(int(batch_size), 1)
        # A bounded batch keeps JPEG decode memory predictable while avoiding
        # a synchronous mmap flush for every eight frames. The previous small
        # batch made short episodes pay dozens of unnecessary writeback waits.
        workers = min(4, batch_size)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(1, len(messages), batch_size):
                batch = messages[start:start + batch_size]
                for offset, frame in enumerate(
                    executor.map(decode, batch), start
                ):
                    if frame.shape != first.shape:
                        raise ValueError(
                            f"inconsistent RGB frame shape {frame.shape}; expected {first.shape}"
                        )
                    array[offset] = frame
                # Push each small batch through the OS page cache so a long
                # episode cannot accumulate a large dirty mmap footprint.
                array.flush()
        array.flush()
        return _MappedFrameStore(path, array)
    except Exception:
        try:
            del array
        except UnboundLocalError:
            pass
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


class WujiGloveSource:
    """Read Wuji streams without involving teleop or robot control."""

    def __init__(
        self,
        sn: str,
        requested_hz: float,
        *,
        capture_skeleton: bool = False,
        capture_zones: bool = False,
        capture_right_wrist_pose: bool = False,
    ):
        self.sn = sn
        self.requested_hz = float(requested_hz)
        self.capture_skeleton = bool(capture_skeleton)
        self.capture_zones = bool(capture_zones)
        self.capture_right_wrist_pose = bool(capture_right_wrist_pose)
        self.lock = threading.Lock()
        self.angle_frames: deque[tuple[int, int, np.ndarray]] = deque(maxlen=4096)
        self.tactile_frames: deque[tuple[int, int, np.ndarray]] = deque(maxlen=4096)
        self.skeleton_frames: deque[tuple[int, int, np.ndarray]] = deque(maxlen=4096)
        self.zone_frames: deque[tuple[int, int, np.ndarray, np.ndarray]] = deque(maxlen=4096)
        self.right_wrist_pose_frames: deque[tuple[int, int, np.ndarray]] = deque(maxlen=4096)
        self.angle_dropped = 0
        self.tactile_dropped = 0
        self.skeleton_dropped = 0
        self.zone_dropped = 0
        self.right_wrist_pose_dropped = 0
        self.angle_count = 0
        self.tactile_count = 0
        self.skeleton_count = 0
        self.zone_count = 0
        self.right_wrist_pose_count = 0
        self.last_angle_mono = 0.0
        self.last_tactile_mono = 0.0
        self.last_skeleton_mono = 0.0
        self.last_zone_mono = 0.0
        self.last_right_wrist_pose_mono = 0.0
        self.last_tactile_active_taxels: int | None = None
        self.error = "not started"
        self.device_name = "glove_d435_collect"
        self.manager = None
        self.connection_count = 0
        self.reconnects = 0
        self.connection_started_mono = 0.0
        self.reconnect_after_s = 0.5
        self.next_reconnect_mono = 0.0
        self.reconnect_retry_s = 1.0
        self.stop_event = threading.Event()
        self.device = None
        self.angle_sub = None
        self.tactile_sub = None
        self.skeleton_sub = None
        self.zone_sub = None
        self.tf_sub = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            from wuji_sdk import SdkManager
        except ImportError as exc:
            raise SystemExit("找不到 wuji_sdk；请使用可运行遥操的 Python 环境。") from exc

        self.manager = SdkManager.instance()
        self._open_transport()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open_transport(self) -> None:
        """Connect and create a complete, fresh set of stream subscriptions."""
        if self.manager is None:
            raise RuntimeError("Wuji SDK manager is not initialized")
        device = self.manager.connect(sn=self.sn, device_name=self.device_name)
        angle_sub = tactile_sub = skeleton_sub = zone_sub = tf_sub = None
        try:
            angle_sub = device.hand_joint_angles().subscribe()
            tactile_sub = device.tactile().subscribe()
            if self.capture_skeleton:
                skeleton_sub = device.hand_skeleton().subscribe()
            if self.capture_zones:
                zone_sub = device.tactile_zones().subscribe()
            if self.capture_right_wrist_pose:
                # ``tf`` is a manager-level, cross-device topic. It carries
                # the dynamic waist -> wrist transform.
                tf_sub = self.manager.tf().subscribe()
        except Exception:
            for subscription in (angle_sub, tactile_sub, skeleton_sub, zone_sub, tf_sub):
                try:
                    if subscription is not None:
                        subscription.close()
                except Exception:
                    pass
            try:
                device.disconnect()
            except Exception:
                pass
            raise
        self.device = device
        self.angle_sub = angle_sub
        self.tactile_sub = tactile_sub
        self.skeleton_sub = skeleton_sub
        self.zone_sub = zone_sub
        self.tf_sub = tf_sub
        # Hand tracking derivatives (angles/skeleton) follow the EMF source
        # rate and cannot be set individually.  Tactile is independently
        # configurable, so only request its rate here.
        try:
            # The vendor binding accepts an integer frequency.  Passing the
            # argparse float (e.g. 120.0) raises a TypeError and silently
            # leaves the stream at its firmware default, which makes the
            # requested-rate metadata misleading.
            actual = tactile_sub.set_rate(int(round(self.requested_hz)))
            print(f"Wuji tactile requested={self.requested_hz:g} Hz, actual={actual} Hz")
        except Exception as exc:
            print(f"Wuji tactile rate left at device default: {exc}")
        suffix = " + skeleton" if self.capture_skeleton else ""
        suffix += " + tactile zones" if self.capture_zones else ""
        suffix += " + waist->r_wrist TF" if self.capture_right_wrist_pose else ""
        print("Wuji hand tracking (angles%s) follows the EMF source rate." % suffix)
        with self.lock:
            self.connection_count += 1
            self.connection_started_mono = time.monotonic()
            self.error = "waiting for glove frames"

    def _close_transport(self) -> None:
        for subscription in (
            self.angle_sub,
            self.tactile_sub,
            self.skeleton_sub,
            self.zone_sub,
            self.tf_sub,
        ):
            try:
                if subscription is not None:
                    subscription.close()
            except Exception:
                pass
        self.angle_sub = self.tactile_sub = self.skeleton_sub = None
        self.zone_sub = self.tf_sub = None
        try:
            if self.device is not None:
                self.device.disconnect()
        except Exception:
            pass
        self.device = None

    def _transport_stale(self, now: float) -> bool:
        with self.lock:
            stamps = [self.last_angle_mono, self.last_tactile_mono]
            if self.capture_skeleton:
                stamps.append(self.last_skeleton_mono)
            if self.capture_zones:
                stamps.append(self.last_zone_mono)
            if self.capture_right_wrist_pose:
                stamps.append(self.last_right_wrist_pose_mono)
            started = self.connection_started_mono
        return bool(
            started
            and any(
                now - (stamp if stamp >= started else started) >= self.reconnect_after_s
                for stamp in stamps
            )
        )

    def _reconnect_transport(self, now: float) -> None:
        if now < self.next_reconnect_mono or self.stop_event.is_set():
            return
        with self.lock:
            self.error = "Wuji stream stalled; reconnecting SDK session"
        print(f"[WUJI] 数据流超过 {self.reconnect_after_s:g}s 未更新，正在自动重连…", flush=True)
        self._close_transport()
        try:
            self._open_transport()
        except Exception as exc:
            with self.lock:
                self.error = f"Wuji reconnect failed: {type(exc).__name__}: {exc}"
            retry_delay_s = self.reconnect_retry_s
            self.next_reconnect_mono = time.monotonic() + retry_delay_s
            self.reconnect_retry_s = min(8.0, self.reconnect_retry_s * 2.0)
            print(
                f"[WUJI] 自动重连失败，{retry_delay_s:g}s 后继续重试：{exc}",
                flush=True,
            )
            return
        self.reconnects += 1
        self.reconnect_retry_s = 1.0
        self.next_reconnect_mono = 0.0
        print(f"[WUJI] 已重建连接和全部订阅（reconnects={self.reconnects}）", flush=True)

    def _append(self, queue: deque, item: tuple[int, int, np.ndarray], kind: str) -> None:
        if len(queue) == queue.maxlen:
            if kind == "angle":
                self.angle_dropped += 1
            elif kind == "tactile":
                self.tactile_dropped += 1
            elif kind == "zone":
                self.zone_dropped += 1
            elif kind == "right_wrist_pose":
                self.right_wrist_pose_dropped += 1
            else:
                self.skeleton_dropped += 1
        queue.append(item)

    def _loop(self) -> None:
        angle_local_seq = 0
        tactile_local_seq = 0
        skeleton_local_seq = 0
        zone_local_seq = 0
        right_wrist_pose_local_seq = 0
        while not self.stop_event.is_set():
            received = False
            if self.angle_sub is None or self.tactile_sub is None:
                self._reconnect_transport(time.monotonic())
                time.sleep(0.01)
                continue
            try:
                angle_frame = self.angle_sub.recv()
                if angle_frame is not None:
                    angle = _glove_angles(angle_frame)
                    if angle is not None:
                        angle_local_seq += 1
                        with self.lock:
                            self._append(
                                self.angle_frames,
                                (_stamp_us(angle_frame), _frame_seq(angle_frame, angle_local_seq), angle),
                                "angle",
                            )
                            self.angle_count += 1
                            self.last_angle_mono = time.monotonic()
                        received = True

                tactile_frame = self.tactile_sub.recv()
                if tactile_frame is not None:
                    tactile = _glove_tactile(tactile_frame)
                    if tactile is not None:
                        tactile_local_seq += 1
                        with self.lock:
                            self._append(
                                self.tactile_frames,
                                (_stamp_us(tactile_frame), _frame_seq(tactile_frame, tactile_local_seq), tactile),
                                "tactile",
                            )
                            self.tactile_count += 1
                            self.last_tactile_mono = time.monotonic()
                            self.last_tactile_active_taxels = tactile_active_taxel_count(tactile)
                        received = True

                if self.skeleton_sub is not None:
                    skeleton_frame = self.skeleton_sub.recv()
                    if skeleton_frame is not None:
                        skeleton = _glove_skeleton(skeleton_frame)
                        if skeleton is not None:
                            skeleton_local_seq += 1
                            with self.lock:
                                self._append(
                                    self.skeleton_frames,
                                    (_stamp_us(skeleton_frame), _frame_seq(skeleton_frame, skeleton_local_seq), skeleton),
                                    "skeleton",
                                )
                                self.skeleton_count += 1
                                self.last_skeleton_mono = time.monotonic()
                            received = True
                if self.zone_sub is not None:
                    zone_frame = self.zone_sub.recv()
                    if zone_frame is not None:
                        zone = _glove_tactile_zones(zone_frame)
                        if zone is not None:
                            zone_local_seq += 1
                            stats, counts = zone
                            with self.lock:
                                self._append(
                                    self.zone_frames,
                                    (_stamp_us(zone_frame), _frame_seq(zone_frame, zone_local_seq), stats, counts),
                                    "zone",
                                )
                                self.zone_count += 1
                                self.last_zone_mono = time.monotonic()
                                received = True
                if self.tf_sub is not None:
                    tf_frame = self.tf_sub.recv()
                    if tf_frame is not None:
                        wrist_pose = _right_wrist_pose(tf_frame)
                        if wrist_pose is not None:
                            timestamp_us, pose = wrist_pose
                            right_wrist_pose_local_seq += 1
                            with self.lock:
                                self._append(
                                    self.right_wrist_pose_frames,
                                    (timestamp_us, right_wrist_pose_local_seq, pose),
                                    "right_wrist_pose",
                                )
                                self.right_wrist_pose_count += 1
                                self.last_right_wrist_pose_mono = time.monotonic()
                            received = True
                if received:
                    self.error = ""
                else:
                    time.sleep(0.0005)
            except Exception as exc:
                self.error = str(exc)
                time.sleep(0.01)
            now = time.monotonic()
            if self._transport_stale(now):
                self._reconnect_transport(now)

    def drain(self) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
        with self.lock:
            angles, tactile = list(self.angle_frames), list(self.tactile_frames)
            self.angle_frames.clear()
            self.tactile_frames.clear()
        return angles, tactile

    def drain_skeleton(self) -> list[tuple[int, int, np.ndarray]]:
        with self.lock:
            skeleton = list(self.skeleton_frames)
            self.skeleton_frames.clear()
        return skeleton

    def drain_zones(self) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
        with self.lock:
            zones = list(self.zone_frames)
            self.zone_frames.clear()
        return zones

    def drain_right_wrist_pose(self) -> list[tuple[int, int, np.ndarray]]:
        with self.lock:
            poses = list(self.right_wrist_pose_frames)
            self.right_wrist_pose_frames.clear()
        return poses

    def clear(self) -> None:
        self.drain()
        self.drain_skeleton()
        self.drain_zones()
        self.drain_right_wrist_pose()

    def healthy(self, max_age_s: float = 0.25) -> bool:
        """Return true only while every requested glove stream is fresh."""
        now = time.monotonic()
        with self.lock:
            stamps = [self.last_angle_mono, self.last_tactile_mono]
            if self.capture_skeleton:
                stamps.append(self.last_skeleton_mono)
            if self.capture_zones:
                stamps.append(self.last_zone_mono)
            if self.capture_right_wrist_pose:
                stamps.append(self.last_right_wrist_pose_mono)
            return (
                all(stamp > 0.0 and now - stamp <= max_age_s for stamp in stamps)
                and self.last_tactile_active_taxels == WUJI_OFFICIAL_ACTIVE_TAXELS
            )

    def status(self) -> str:
        with self.lock:
            now = time.monotonic()
            angle_age = (now - self.last_angle_mono) * 1000.0 if self.last_angle_mono else float("inf")
            tactile_age = (now - self.last_tactile_mono) * 1000.0 if self.last_tactile_mono else float("inf")
            skeleton_age = (now - self.last_skeleton_mono) * 1000.0 if self.last_skeleton_mono else float("inf")
            zone_age = (now - self.last_zone_mono) * 1000.0 if self.last_zone_mono else float("inf")
            wrist_age = (now - self.last_right_wrist_pose_mono) * 1000.0 if self.last_right_wrist_pose_mono else float("inf")
            return (
                f"angles={self.angle_count}, tactile={self.tactile_count}, skeleton={self.skeleton_count}, zones={self.zone_count}, wrist_tf={self.right_wrist_pose_count}, "
                f"queued=({len(self.angle_frames)},{len(self.tactile_frames)},{len(self.skeleton_frames)},{len(self.zone_frames)},{len(self.right_wrist_pose_frames)}), "
                f"dropped=({self.angle_dropped},{self.tactile_dropped},{self.skeleton_dropped},{self.zone_dropped},{self.right_wrist_pose_dropped}), "
                f"age_ms=({angle_age:.0f},{tactile_age:.0f},{skeleton_age:.0f},{zone_age:.0f},{wrist_age:.0f}), "
                f"tactile_active={self.last_tactile_active_taxels}/{GLOVE_SHAPE[0] * GLOVE_SHAPE[1]} "
                f"(official={WUJI_OFFICIAL_ACTIVE_TAXELS}), connection={self.connection_count}, "
                f"reconnects={self.reconnects} {self.error}"
            )

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._close_transport()


class D435Node:
    """ROS subscriber that queues RGB and, only when requested, depth frames."""

    def __init__(
        self,
        color_topic: str,
        depth_topic: str,
        color_info_topic: str,
        depth_info_topic: str,
        capture_depth: bool,
        gemini_topic: str | None = None,
        gemini_imu_topic: str | None = None,
        capture_ego: bool = True,
        tracker_camera_topic: str | None = None,
        tracker_wrist_topic: str | None = None,
        tracker_camera_z_offset_m: float = 0.1,
        ego_compressed: bool = False,
        ego_require_camera_info: bool = True,
        ego_source: str = "RealSense D435",
    ):
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
            from geometry_msgs.msg import PoseStamped
            from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu
        except ImportError as exc:
            raise SystemExit("找不到 ROS 2 Python 环境；先 source /opt/ros/humble/setup.bash") from exc

        self._rclpy = rclpy
        rclpy.init(args=None)
        self.node = Node("wuji_glove_d435_collect")
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            # RGB callbacks only retain a message reference. A deep queue is
            # harmful here: during a CPU burst it replays stale pixels later,
            # even though their ROS callback time looks current enough for an
            # episode boundary. Keep only a short latest-frame window.
            depth=6,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.capture_depth = bool(capture_depth)
        self.capture_ego = bool(capture_ego)
        self.ego_compressed = bool(ego_compressed)
        self.ego_require_camera_info = bool(ego_require_camera_info)
        self.ego_source = str(ego_source).strip() or "first-person camera"
        self.ego_topic = str(color_topic)
        self.gemini_topic = (gemini_topic or "").strip()
        self.gemini_enabled = bool(self.gemini_topic)
        self.gemini_compressed = self.gemini_topic.endswith("/compressed")
        self.gemini_imu_topic = (gemini_imu_topic or "").strip()
        self.gemini_imu_enabled = bool(self.gemini_imu_topic)
        self.tracker_camera_topic = (tracker_camera_topic or "").strip()
        self.tracker_wrist_topic = (tracker_wrist_topic or "").strip()
        self.trackers_enabled = bool(self.tracker_camera_topic or self.tracker_wrist_topic)
        if self.trackers_enabled and not (self.tracker_camera_topic and self.tracker_wrist_topic):
            raise ValueError("tracker camera and wrist topics must be configured together")
        self.tracker_camera_z_offset_m = float(tracker_camera_z_offset_m)
        self.lock = threading.Lock()
        self.recording = False
        # Keep ROS messages in the callbacks and decode only after an episode
        # stops.  Copying/converting two uncompressed 60 Hz RGB streams inside
        # one Python executor used enough callback time to lose camera frames.
        # Retaining the message object is safe and makes each live callback a
        # small timestamp/list operation.
        self.color_frames: list[tuple[int, Any]] = []
        self.depth_frames: list[tuple[int, Any]] = []
        self.gemini_frames: list[tuple[int, int, Any]] = []
        self.gemini_pose_frames: list[tuple[int, int, np.ndarray]] = []
        self.tracker_camera_frames: list[tuple[int, int, np.ndarray]] = []
        self.tracker_wrist_frames: list[tuple[int, int, np.ndarray]] = []
        self.color_info: dict[str, Any] | None = None
        self.depth_info: dict[str, Any] | None = None
        self.color_count = 0
        self.depth_count = 0
        self.gemini_count = 0
        self.gemini_pose_count = 0
        self.tracker_camera_count = 0
        self.tracker_wrist_count = 0
        self.color_last_mono = 0.0
        self.episode_rgb_timestamps_ns: list[int] = []
        self.episode_rgb_fault = ""
        self.episode_source_fault = ""
        self.episode_max_rgb_gap_ns = 0
        self.episode_missing_ratio = 0.0
        self.live_max_rgb_gap_ns = 0
        self.live_max_missing_ratio = 0.0
        self.depth_last_mono = 0.0
        self.gemini_last_mono = 0.0
        self.gemini_pose_last_mono = 0.0
        self.tracker_camera_last_mono = 0.0
        self.tracker_wrist_last_mono = 0.0
        self._gemini_pose_rpy: np.ndarray | None = None
        self._gemini_pose_last_stamp_ns = 0
        self.error = f"waiting for {self.ego_source} frames" if self.capture_ego else "waiting for Gemini frames"
        # Camera callbacks retain message objects only. The shallow best-effort
        # queue favors freshness and drops old RGB during scheduling bursts.
        if self.capture_ego:
            ego_msg_type = CompressedImage if self.ego_compressed else Image
            # Match the PICO/USB camera publishers' best-effort sensor QoS so
            # a slow callback queue cannot backpressure the camera source.
            self.node.create_subscription(ego_msg_type, color_topic, self._color_cb, image_qos)
            if self.ego_require_camera_info:
                self.node.create_subscription(CameraInfo, color_info_topic, self._color_info_cb, 3)
        if self.gemini_enabled:
            gemini_msg_type = CompressedImage if self.gemini_compressed else Image
            self.node.create_subscription(
                gemini_msg_type,
                self.gemini_topic,
                self._gemini_cb,
                image_qos,
            )
        if self.gemini_imu_enabled:
            self.node.create_subscription(Imu, self.gemini_imu_topic, self._gemini_imu_cb, 240)
        if self.trackers_enabled:
            # The publisher emits only new SDK samples; a large depth keeps
            # native messages available during short executor scheduling
            # bursts without changing or fabricating samples.
            self.node.create_subscription(PoseStamped, self.tracker_camera_topic, self._tracker_camera_cb, 240)
            self.node.create_subscription(PoseStamped, self.tracker_wrist_topic, self._tracker_wrist_cb, 240)
        if self.capture_depth:
            self.node.create_subscription(Image, depth_topic, self._depth_cb, 30)
            self.node.create_subscription(CameraInfo, depth_info_topic, self._depth_info_cb, 3)
        self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()

    @staticmethod
    def _info(msg: Any) -> dict[str, Any]:
        return {
            "width": int(msg.width), "height": int(msg.height),
            "distortion_model": str(msg.distortion_model),
            "k": [float(v) for v in msg.k], "d": [float(v) for v in msg.d],
            "r": [float(v) for v in msg.r], "p": [float(v) for v in msg.p],
        }

    def _color_info_cb(self, msg: Any) -> None:
        with self.lock:
            self.color_info = self._info(msg)
            # With align_depth enabled, depth pixels are expressed in the color
            # optical frame.  Some driver versions omit the aligned CameraInfo.
            if self.capture_depth and self.depth_info is None:
                self.depth_info = dict(self.color_info)

    def _depth_info_cb(self, msg: Any) -> None:
        with self.lock:
            self.depth_info = self._info(msg)

    def _color_cb(self, msg: Any) -> None:
        try:
            stamp_ns = _ros_stamp_ns(msg.header)
            with self.lock:
                if self.recording:
                    self.color_frames.append((stamp_ns, msg))
                    self._update_live_rgb_quality(stamp_ns)
                self.color_count += 1
                self.color_last_mono = time.monotonic()
                self.error = ""
        except Exception as exc:
            self.error = str(exc)

    def configure_live_rgb_quality(self, max_gap_ns: int, max_missing_ratio: float) -> None:
        """Configure quality limits used to refuse a bad episode while recording."""
        with self.lock:
            self.live_max_rgb_gap_ns = max(int(max_gap_ns), 0)
            self.live_max_missing_ratio = max(float(max_missing_ratio), 0.0)

    def _update_live_rgb_quality(self, stamp_ns: int) -> None:
        """Update live ego timing quality. Caller must hold ``self.lock``."""
        timestamps = self.episode_rgb_timestamps_ns
        if timestamps:
            gap_ns = stamp_ns - timestamps[-1]
            self.episode_max_rgb_gap_ns = max(self.episode_max_rgb_gap_ns, gap_ns)
            if gap_ns <= 0:
                self.episode_rgb_fault = f"第一人称 RGB 时间戳倒退或重复：gap={gap_ns / 1e6:.2f} ms"
            elif self.live_max_rgb_gap_ns and gap_ns > self.live_max_rgb_gap_ns:
                self.episode_rgb_fault = (
                    f"第一人称 RGB 中断：gap={gap_ns / 1e6:.2f} ms "
                    f"> {self.live_max_rgb_gap_ns / 1e6:.2f} ms"
                )
        timestamps.append(stamp_ns)
        # Do not reject an episode from this live ratio. Callback delivery can
        # jitter independently of the publisher, while the complete raw stream
        # is checked at save time. A clearly oversized gap above is sufficient
        # to stop a genuinely disconnected camera immediately.

    def live_rgb_fault(self) -> str:
        with self.lock:
            return self.episode_rgb_fault

    def live_tracker_frames(self) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
        """Return shallow live snapshots for recording-time quality gates."""
        with self.lock:
            return list(self.tracker_camera_frames), list(self.tracker_wrist_frames)

    def live_source_fault(self) -> str:
        with self.lock:
            return self.episode_source_fault

    def live_source_timestamps_ns(self) -> dict[str, int]:
        """Latest acquisition stamps used by live cross-source alignment gates."""
        with self.lock:
            return {
                "ego": self.color_frames[-1][0] if self.color_frames else 0,
                "gemini_imu": self.gemini_pose_frames[-1][0] if self.gemini_pose_frames else 0,
                "tracker_camera": self.tracker_camera_frames[-1][0] if self.tracker_camera_frames else 0,
                "tracker_wrist": self.tracker_wrist_frames[-1][0] if self.tracker_wrist_frames else 0,
            }

    def live_alignment_timestamp_windows_ns(self) -> dict[str, list[int]]:
        """Recent source stamps for delayed-watermark alignment checks."""
        with self.lock:
            return {
                "ego": [frame[0] for frame in self.color_frames[-32:]],
                "gemini_imu": [frame[0] for frame in self.gemini_pose_frames[-64:]],
                "tracker_camera": [frame[0] for frame in self.tracker_camera_frames[-64:]],
                "tracker_wrist": [frame[0] for frame in self.tracker_wrist_frames[-64:]],
            }

    def _depth_cb(self, msg: Any) -> None:
        try:
            with self.lock:
                if self.recording:
                    self.depth_frames.append((_ros_stamp_ns(msg.header), msg))
                self.depth_count += 1
                self.depth_last_mono = time.monotonic()
                self.error = ""
        except Exception as exc:
            self.error = str(exc)

    def _gemini_cb(self, msg: Any) -> None:
        try:
            stamp_ns = _ros_stamp_ns(msg.header)
            with self.lock:
                self.gemini_count += 1
                if self.recording:
                    if self.gemini_frames and stamp_ns <= self.gemini_frames[-1][0]:
                        self.episode_source_fault = f"Gemini RGB timestamp invalid: {stamp_ns} <= {self.gemini_frames[-1][0]}"
                    self.gemini_frames.append((stamp_ns, self.gemini_count, msg))
                self.gemini_last_mono = time.monotonic()
                self.error = ""
        except Exception as exc:
            self.error = str(exc)

    def _gemini_imu_cb(self, msg: Any) -> None:
        """Track Gemini's gravity-stabilized, start-relative orientation.

        Gemini 2 supplies acceleration and angular velocity, not a fused
        world-frame pose. Its optical-origin translation is therefore zero and
        yaw is gyro-integrated from process start; the saved metadata records
        those limits explicitly.
        """
        try:
            stamp_ns = _ros_stamp_ns(msg.header)
            accel = np.asarray(
                (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z), dtype=np.float64
            )
            gyro = np.asarray(
                (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z), dtype=np.float64
            )
            if not np.all(np.isfinite(accel)) or not np.all(np.isfinite(gyro)) or float(np.linalg.norm(accel)) < 1e-5:
                return
            accel_roll = math.atan2(float(accel[1]), float(accel[2]))
            accel_pitch = math.atan2(-float(accel[0]), math.hypot(float(accel[1]), float(accel[2])))
            if self._gemini_pose_rpy is None:
                rpy = np.asarray((accel_roll, accel_pitch, 0.0), dtype=np.float64)
            else:
                dt = min(max((stamp_ns - self._gemini_pose_last_stamp_ns) / 1e9, 0.0), 0.05)
                rpy = self._gemini_pose_rpy.astype(np.float64, copy=True) + gyro * dt
                correction = min(0.04, 2.0 * dt)
                rpy[0] = (1.0 - correction) * rpy[0] + correction * accel_roll
                rpy[1] = (1.0 - correction) * rpy[1] + correction * accel_pitch
                rpy[2] = (rpy[2] + math.pi) % (2.0 * math.pi) - math.pi
            pose = np.concatenate((np.zeros(3, dtype=np.float32), rpy.astype(np.float32)))
            with self.lock:
                self._gemini_pose_rpy = rpy.astype(np.float32)
                self._gemini_pose_last_stamp_ns = stamp_ns
                self.gemini_pose_count += 1
                if self.recording:
                    if self.gemini_pose_frames and stamp_ns <= self.gemini_pose_frames[-1][0]:
                        self.episode_source_fault = f"Gemini IMU timestamp invalid: {stamp_ns} <= {self.gemini_pose_frames[-1][0]}"
                    self.gemini_pose_frames.append((stamp_ns, self.gemini_pose_count, pose))
                self.gemini_pose_last_mono = time.monotonic()
                self.error = ""
        except Exception as exc:
            self.error = str(exc)

    def _tracker_cb(self, msg: Any, *, camera: bool) -> None:
        try:
            pose = _pose_stamped_to_xyz_rpy(
                msg,
                z_offset_m=self.tracker_camera_z_offset_m if camera else 0.0,
            )
            if pose is None:
                return
            stamp_ns = _ros_stamp_ns(msg.header)
            with self.lock:
                if camera:
                    self.tracker_camera_count += 1
                    seq = self.tracker_camera_count
                    if self.recording:
                        if self.tracker_camera_frames and stamp_ns <= self.tracker_camera_frames[-1][0]:
                            self.episode_source_fault = f"PICO camera Tracker timestamp invalid: {stamp_ns} <= {self.tracker_camera_frames[-1][0]}"
                        self.tracker_camera_frames.append((stamp_ns, seq, pose))
                    self.tracker_camera_last_mono = time.monotonic()
                else:
                    self.tracker_wrist_count += 1
                    seq = self.tracker_wrist_count
                    if self.recording:
                        if self.tracker_wrist_frames and stamp_ns <= self.tracker_wrist_frames[-1][0]:
                            self.episode_source_fault = f"PICO wrist Tracker timestamp invalid: {stamp_ns} <= {self.tracker_wrist_frames[-1][0]}"
                        self.tracker_wrist_frames.append((stamp_ns, seq, pose))
                    self.tracker_wrist_last_mono = time.monotonic()
                self.error = ""
        except Exception as exc:
            self.error = str(exc)

    def _tracker_camera_cb(self, msg: Any) -> None:
        self._tracker_cb(msg, camera=True)

    def _tracker_wrist_cb(self, msg: Any) -> None:
        self._tracker_cb(msg, camera=False)

    def start_episode(self) -> None:
        with self.lock:
            self.color_frames = []
            self.depth_frames = []
            self.gemini_frames = []
            self.gemini_pose_frames = []
            self.tracker_camera_frames = []
            self.tracker_wrist_frames = []
            self.episode_rgb_timestamps_ns = []
            self.episode_rgb_fault = ""
            self.episode_source_fault = ""
            self.episode_max_rgb_gap_ns = 0
            self.episode_missing_ratio = 0.0
            self.recording = True

    def finish_episode(
        self, *, decode: bool = True
    ) -> tuple[list[tuple[int, Any]], list[tuple[int, Any]], dict[str, Any] | None, dict[str, Any] | None]:
        with self.lock:
            self.recording = False
            color_messages = self.color_frames
            depth_messages = self.depth_frames
            self.color_frames = []
            self.depth_frames = []
            color_info = self.color_info
            depth_info = self.depth_info
        if not decode:
            # FTP-1 save() uses a disk-backed decoder after applying its
            # episode-boundary and alignment gates.  Keep the legacy decoded
            # return value for the standalone collector and existing callers.
            return color_messages, depth_messages, color_info, depth_info
        # Decoding happens after recording has stopped, off the ROS callback
        # path.  This can take a moment when the user presses ``e`` but cannot
        # make the next live camera callback miss a frame.
        if self.ego_compressed:
            color = _decode_compressed_messages(color_messages)
        else:
            color = [(stamp, _image_array(msg, want_color=True)) for stamp, msg in color_messages]
        depth = [(stamp, _image_array(msg, want_color=False)) for stamp, msg in depth_messages]
        return color, depth, color_info, depth_info

    def discard_episode(self) -> None:
        with self.lock:
            self.recording = False
            self.color_frames = []
            self.depth_frames = []
            self.gemini_frames = []
            self.gemini_pose_frames = []
            self.tracker_camera_frames = []
            self.tracker_wrist_frames = []

    def finish_gemini_episode(self, *, decode: bool = True) -> list[tuple[int, int, Any]]:
        """Return Gemini frames captured in the most recently finished episode."""
        with self.lock:
            messages = self.gemini_frames
            self.gemini_frames = []
        if not decode:
            return messages
        # Gemini recording uses compressed MJPEG whenever available.
        # Decode only after recording has stopped so JPEG decoding cannot
        # steal CPU from the live RealSense 60 Hz callback path.
        if self.gemini_compressed:
            return _decode_compressed_messages(messages)

        # Legacy raw bgr8 fallback.
        return [
            (stamp, seq, _image_array(msg, want_color=True))
            for stamp, seq, msg in messages
        ]

    def finish_gemini_pose_episode(self) -> list[tuple[int, int, np.ndarray]]:
        with self.lock:
            poses = self.gemini_pose_frames
            self.gemini_pose_frames = []
        return poses

    def finish_tracker_episode(self) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
        with self.lock:
            camera = self.tracker_camera_frames
            wrist = self.tracker_wrist_frames
            self.tracker_camera_frames = []
            self.tracker_wrist_frames = []
        return camera, wrist

    def ready(self, max_age_s: float = 2.0) -> bool:
        with self.lock:
            now = time.monotonic()
            fresh = lambda last: last > 0 and now - last <= max(float(max_age_s), 0.0)
            color_ready = not self.capture_ego or (
                fresh(self.color_last_mono)
                and (not self.ego_require_camera_info or self.color_info is not None)
            )
            depth_ready = (
                not self.capture_depth
                or (self.depth_info is not None and fresh(self.depth_last_mono))
            )
            gemini_ready = not self.gemini_enabled or fresh(self.gemini_last_mono)
            gemini_pose_ready = not self.gemini_imu_enabled or fresh(self.gemini_pose_last_mono)
            tracker_camera_ready = not self.trackers_enabled or fresh(self.tracker_camera_last_mono)
            tracker_wrist_ready = not self.trackers_enabled or fresh(self.tracker_wrist_last_mono)
            return color_ready and depth_ready and gemini_ready and gemini_pose_ready and tracker_camera_ready and tracker_wrist_ready

    def status(self) -> str:
        with self.lock:
            now = time.monotonic()
            color_age = (now - self.color_last_mono) * 1000.0 if self.color_last_mono else float("inf")
            if not self.capture_ego:
                result = "RGB-disabled"
                if self.gemini_enabled:
                    gemini_age = (now - self.gemini_last_mono) * 1000.0 if self.gemini_last_mono else float("inf")
                    result += f" | Gemini={self.gemini_count}, age_ms={gemini_age:.0f}"
                if self.gemini_imu_enabled:
                    pose_age = (now - self.gemini_pose_last_mono) * 1000.0 if self.gemini_pose_last_mono else float("inf")
                    result += f" | GeminiIMU={self.gemini_pose_count}, age_ms={pose_age:.0f}"
                if self.trackers_enabled:
                    camera_age = (now - self.tracker_camera_last_mono) * 1000.0 if self.tracker_camera_last_mono else float("inf")
                    wrist_age = (now - self.tracker_wrist_last_mono) * 1000.0 if self.tracker_wrist_last_mono else float("inf")
                    result += f" | Trackers=({self.tracker_camera_count},{self.tracker_wrist_count}), age_ms=({camera_age:.0f},{wrist_age:.0f})"
                return f"{result} {self.error}"
            if not self.capture_depth:
                result = f"RGB-only color={self.color_count}, age_ms={color_age:.0f}"
                if self.gemini_enabled:
                    gemini_age = (now - self.gemini_last_mono) * 1000.0 if self.gemini_last_mono else float("inf")
                    result += f" | Gemini={self.gemini_count}, age_ms={gemini_age:.0f}"
                if self.gemini_imu_enabled:
                    pose_age = (now - self.gemini_pose_last_mono) * 1000.0 if self.gemini_pose_last_mono else float("inf")
                    result += f" | GeminiIMU={self.gemini_pose_count}, age_ms={pose_age:.0f}"
                if self.trackers_enabled:
                    camera_age = (now - self.tracker_camera_last_mono) * 1000.0 if self.tracker_camera_last_mono else float("inf")
                    wrist_age = (now - self.tracker_wrist_last_mono) * 1000.0 if self.tracker_wrist_last_mono else float("inf")
                    result += f" | Trackers=({self.tracker_camera_count},{self.tracker_wrist_count}), age_ms=({camera_age:.0f},{wrist_age:.0f})"
                return f"{result} {self.error}"
            depth_age = (now - self.depth_last_mono) * 1000.0 if self.depth_last_mono else float("inf")
            return f"color={self.color_count}, depth={self.depth_count}, age_ms=({color_age:.0f},{depth_age:.0f}) {self.error}"

    def close(self) -> None:
        try:
            self.node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass
        self.thread.join(timeout=2.0)


class Collector:
    def __init__(self, output: Path, glove: WujiGloveSource, camera: D435Node):
        self.output = output.expanduser()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(self.output), mode="a")
        self.episodes = self.root.require_group("episodes")
        self.glove = glove
        self.camera = camera
        self.recording = False
        self.angle_frames: list[tuple[int, int, np.ndarray]] = []
        self.tactile_frames: list[tuple[int, int, np.ndarray]] = []
        self.root.attrs.update({
            "format": "wuji_glove_d435_zarr_v1",
            "description": "Wuji glove + Intel RealSense RGB; no robot data",
            "glove_tactile_shape": list(GLOVE_SHAPE),
            "glove_tactile_invalid_value": -1.0,
            "glove_tactile_pressure_invalid_value": -1.0,
            "glove_tactile_pressure_unit": "sdk_calibrated_raw_not_newton",
            "streams_are_independent": True,
        })
        if self.camera.capture_depth:
            self.root.attrs["d435_depth_unit"] = "millimetre_uint16; zero means invalid/unavailable"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._drain_loop, daemon=True)
        self.thread.start()

    def _drain_loop(self) -> None:
        while not self.stop_event.is_set():
            angles, tactile = self.glove.drain()
            if self.recording:
                self.angle_frames.extend(angles)
                self.tactile_frames.extend(tactile)
            time.sleep(0.001)

    def _episode_count(self) -> int:
        return sum(key.startswith("episode_") for key in self.episodes.group_keys())

    def start(self, instruction: str) -> None:
        if self.recording:
            print("已经在采集中")
            return
        if not self.camera.ready():
            required = "RGB、深度和内参" if self.camera.capture_depth else "RGB 和内参"
            print(f"[REFUSED] D435 尚未收到 {required}；先检查相机 ROS 驱动。")
            return
        instruction = " ".join(instruction.split())
        if not instruction:
            print("[REFUSED] 请输入任务描述")
            return
        self.glove.clear()
        self.angle_frames = []
        self.tactile_frames = []
        self.camera.start_episode()
        self.recording = True
        self.instruction = instruction
        self.started_ns = time.time_ns()
        print("[REC] episode 开始：", instruction)

    @staticmethod
    def _rate(timestamps: np.ndarray, scale: float) -> float:
        if len(timestamps) < 2:
            return 0.0
        return float((len(timestamps) - 1) / max((timestamps[-1] - timestamps[0]) * scale, 1e-9))

    def save(self) -> None:
        if not self.recording:
            print("当前没有 episode")
            return
        self.recording = False
        # One final drain after the flag closes the recording window.
        angles, tactile = self.glove.drain()
        self.angle_frames.extend(angles)
        self.tactile_frames.extend(tactile)
        color, depth, color_info, depth_info = self.camera.finish_episode()
        if (
            not self.angle_frames
            or not self.tactile_frames
            or not color
            or (self.camera.capture_depth and not depth)
        ):
            required = "手套角度、手套触觉、D435 RGB 和深度" if self.camera.capture_depth else "手套角度、手套触觉和 D435 RGB"
            print(f"[SAVE REFUSED] 需要同时有{required}数据")
            return

        name = f"episode_{self._episode_count():06d}"
        group = self.episodes.create_group(name)
        angle_ts = np.asarray([item[0] for item in self.angle_frames], dtype=np.int64)
        tactile_ts = np.asarray([item[0] for item in self.tactile_frames], dtype=np.int64)
        color_ts = np.asarray([item[0] for item in color], dtype=np.int64)
        tactile_raw = np.stack([item[2] for item in self.tactile_frames]).astype(np.float32)
        tactile_valid, tactile_pressure, tactile_summary = zip(*[_tactile_features(frame) for frame in tactile_raw])
        arrays = {
            "glove_angles": np.stack([item[2] for item in self.angle_frames]).astype(np.float32),
            "glove_angle_timestamp_us": angle_ts,
            "glove_angle_seq": np.asarray([item[1] for item in self.angle_frames], dtype=np.int64),
            "glove_tactile_raw": tactile_raw,
            "glove_tactile_valid_mask": np.stack(tactile_valid).astype(np.uint8),
            "glove_tactile_pressure": np.stack(tactile_pressure).astype(np.float32),
            "glove_tactile_summary": np.stack(tactile_summary).astype(np.float32),
            "glove_tactile_timestamp_us": tactile_ts,
            "glove_tactile_seq": np.asarray([item[1] for item in self.tactile_frames], dtype=np.int64),
            "d435_color_rgb": np.stack([item[1] for item in color]).astype(np.uint8),
            "d435_color_timestamp_ns": color_ts,
        }
        if self.camera.capture_depth:
            depth_ts = np.asarray([item[0] for item in depth], dtype=np.int64)
            arrays["d435_depth_mm"] = np.stack([item[1] for item in depth]).astype(np.uint16)
            arrays["d435_depth_timestamp_ns"] = depth_ts
        for key, value in arrays.items():
            chunks = (1, *value.shape[1:]) if value.ndim >= 3 else (min(max(len(value), 1), 256), *value.shape[1:])
            group.create_dataset(key, data=value, shape=value.shape, chunks=chunks, dtype=value.dtype)
        group.attrs.update({
            "format": "wuji_glove_d435_episode_v1",
            "success": True,
            "instruction": self.instruction,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_unix_ns": self.started_ns,
            "glove_angle_hz": self._rate(angle_ts, 1e-6),
            "glove_tactile_hz": self._rate(tactile_ts, 1e-6),
            "d435_color_hz": self._rate(color_ts, 1e-9),
            "d435_color_intrinsics": json.dumps(color_info or {}),
        })
        message = (
            f"[SAVE] {name}: angles={len(angle_ts)} ({group.attrs['glove_angle_hz']:.1f} Hz), "
            f"tactile={len(tactile_ts)} ({group.attrs['glove_tactile_hz']:.1f} Hz), "
            f"RGB={len(color_ts)} ({group.attrs['d435_color_hz']:.1f} Hz)"
        )
        if self.camera.capture_depth:
            group.attrs["d435_depth_hz"] = self._rate(depth_ts, 1e-9)
            group.attrs["d435_depth_intrinsics"] = json.dumps(depth_info or {})
            message += f", depth={len(depth_ts)} ({group.attrs['d435_depth_hz']:.1f} Hz)"
        print(message)

    def discard(self) -> None:
        if not self.recording:
            print("当前没有 episode")
            return
        self.recording = False
        self.glove.clear()
        self.angle_frames = []
        self.tactile_frames = []
        self.camera.discard_episode()
        print("[DROP] 当前 episode 已丢弃")

    def status(self) -> None:
        print("glove :", self.glove.status())
        print("D435  :", self.camera.status())
        print("episode:", "采集中" if self.recording else "未采集", f"angles={len(self.angle_frames)} tactile={len(self.tactile_frames)}")

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glove-sn", default=DEFAULT_GLOVE_SN)
    parser.add_argument("--glove-hz", type=float, default=120.0)
    parser.add_argument("--camera-name", default="camera", help="RealSense ROS camera_name")
    parser.add_argument("--camera-namespace", default="camera", help="RealSense ROS camera_namespace")
    parser.add_argument("--with-depth", action="store_true", help="also collect depth (reduces maximum RGB throughput)")
    args = parser.parse_args()
    if not 0 < args.glove_hz <= 120:
        parser.error("--glove-hz 必须在 (0, 120]")

    namespace = args.camera_namespace.strip("/")
    camera_name = args.camera_name.strip("/")
    prefix = "/" + "/".join(part for part in (namespace, camera_name) if part)
    glove = WujiGloveSource(args.glove_sn, args.glove_hz)
    camera = D435Node(
        f"{prefix}/color/image_raw",
        f"{prefix}/aligned_depth_to_color/image_raw",
        f"{prefix}/color/camera_info",
        f"{prefix}/aligned_depth_to_color/camera_info",
        capture_depth=args.with_depth,
    )
    collector = None
    try:
        glove.start()
        collector = Collector(args.output, glove, camera)
        print("=" * 76)
        print("Wuji glove + RealSense RGB collector (no L20/G20, no teleop)")
        print("output:", args.output.expanduser())
        print("commands: s=开始  e=保存成功 episode  d=丢弃  status=状态  q=退出")
        print("=" * 76)
        while True:
            command = input("glove-d435> ").strip().lower()
            if command == "s":
                collector.start(input("任务描述: "))
            elif command == "e":
                collector.save()
            elif command == "d":
                collector.discard()
            elif command == "status":
                collector.status()
            elif command == "q":
                break
            elif command:
                print("未知命令：s | e | d | status | q")
    except KeyboardInterrupt:
        print()
    finally:
        if collector is not None:
            collector.close()
        camera.close()
        glove.close()


if __name__ == "__main__":
    main()
