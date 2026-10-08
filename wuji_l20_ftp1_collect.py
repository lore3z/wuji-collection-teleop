#!/usr/bin/env python3
"""Collect LinkerHand L20/G20 demonstrations in FTP-1 Zarr format.

The collector subscribes to the official LinkerHand ROS 2 topics, samples the
latest synchronized-enough observations at a fixed rate, and appends only
episodes explicitly marked successful.  Failed episodes stay in memory and can
be discarded without modifying the dataset.

FTP-1 reference contract:
    <dataset>.zarr/
      data/<time-major arrays>
      meta/episode_ends

No standalone ``actions`` key is needed by FTP-1.  Its loader constructs action
targets from future state trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FINGER_KEYS = (
    "thumb_matrix",
    "index_matrix",
    "middle_matrix",
    "ring_matrix",
    "little_matrix",
)

# G20 raw20 positions that are real/readable actuators.  Positions 11..14 are
# named Reserved by the vendor SDK and are intentionally excluded from FTP-1's
# canonical hand state.  The complete raw20 vector is preserved separately.
ACTIVE_RAW20_INDICES = np.asarray(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19],
    dtype=np.int64,
)

# FTP-1 FAAS 32-slot mapping for the 16 active G20 channels, in the same order
# as ACTIVE_RAW20_INDICES:
#   thumb/index/middle/ring/pinky flexion base,
#   thumb + four-finger abduction,
#   thumb horizontal abduction,
#   thumb/index/middle/ring/pinky distal flexion.
FTP1_HAND_JOINT_INDICES = np.asarray(
    [2, 7, 12, 17, 22, 26, 6, 11, 16, 21, 1, 3, 8, 13, 18, 23],
    dtype=np.int32,
)

TACTILE_AREAS = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
TACTILE_SENSOR_NAME = "LinkerHandG20Matrix6x12"
TACTILE_TYPE = "matrix"
WUJI_GLOVE_SENSOR_NAME = "WujiGloveTactile24x31"
WUJI_GLOVE_MATRIX_SHAPE = (24, 31)
WUJI_GLOVE_ZONE_NAMES = ("thumb", "index", "middle", "ring", "pinky", "palm")
WUJI_GLOVE_ZONE_AREAS = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
DEFAULT_GLOVE_SN = "WG1KA06260622532"
INSTRUCTION_DTYPE = "<U256"
SENSOR_DTYPE = "<U64"
TYPE_DTYPE = "<U16"

DEFAULT_STATE_TOPIC = "/cb_right_hand_state"
DEFAULT_COMMAND_TOPIC = "/cb_right_hand_control_cmd"
DEFAULT_TACTILE_TOPIC = "/cb_right_hand_matrix_touch"

REQUIRED_FTP1_KEYS = {
    "timestamps",
    "camera_main_rgb",
    "right_hand_joints",
    "right_hand_joints_idx",
    "right_tactile_data_fingers",
    "right_tactile_area_fingers",
    "right_tactile_sensor_fingers",
    "right_tactile_type_fingers",
    "sub_task_instruction",
}

REQUIRED_GLOVE_KEYS = {
    "right_tactile_data_wuji_glove_zones",
    "right_tactile_area_wuji_glove_zones",
    "right_tactile_sensor_wuji_glove_zones",
    "right_tactile_type_wuji_glove_zones",
    "wuji_glove_pressure_matrix_raw",
}


def _dependency_error(package: str, install: str) -> SystemExit:
    return SystemExit(
        f"缺少 Python 包 {package!r}。请先运行：\n"
        f"  /usr/bin/python3 -m pip install --user {install}"
    )


def _import_zarr():
    try:
        import zarr  # type: ignore
        from numcodecs import Blosc  # type: ignore
    except ImportError as exc:
        # ROS 2 Humble on this machine uses Python 3.10, while current zarr 3
        # releases require Python 3.11.  Zarr 2 writes the same format-2 store
        # consumed by FTP-1's Python 3.11/zarr 3 training environment.
        raise _dependency_error("zarr/numcodecs", '"zarr<3" "numcodecs<0.16"') from exc
    return zarr, Blosc


def _stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _chunks_for(value: np.ndarray) -> tuple[int, ...]:
    """Time-only chunks, with one-frame image chunks to bound memory use."""
    if value.ndim == 1:
        return (min(max(len(value), 1), 1024),)
    if value.ndim >= 4 and value.shape[-1] == 3:
        return (1,) + tuple(value.shape[1:])
    if value.ndim >= 4:
        return (min(max(len(value), 1), 32),) + tuple(value.shape[1:])
    return (min(max(len(value), 1), 256),) + tuple(value.shape[1:])


class FTP1ZarrWriter:
    """Append complete episodes with rollback and crash-tail repair."""

    def __init__(self, path: Path, metadata: dict[str, Any] | None = None):
        zarr, Blosc = _import_zarr()
        self.zarr = zarr
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # FTP-1's current writer emits Zarr format 2.  zarr-python 2.x does so
        # by default and FTP-1's Python 3.11/zarr 3 loader reads it directly.
        self.root = zarr.open_group(str(self.path), mode="a")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")
        if "episode_ends" not in self.meta:
            self.meta.create_dataset(
                "episode_ends",
                shape=(0,),
                chunks=(1024,),
                dtype=np.int64,
                compressor=None,
            )
        self.episode_ends = self.meta["episode_ends"]
        self.compressor = Blosc(
            cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE
        )
        self._repair_uncommitted_tail()
        if metadata:
            for key, value in metadata.items():
                if key not in self.root.attrs:
                    self.root.attrs[key] = value

    @property
    def n_steps(self) -> int:
        if self.episode_ends.shape[0] == 0:
            return 0
        return int(self.episode_ends[-1])

    @property
    def n_episodes(self) -> int:
        return int(self.episode_ends.shape[0])

    def _repair_uncommitted_tail(self) -> None:
        committed = self.n_steps
        for key in list(self.data.array_keys()):
            array = self.data[key]
            if array.shape[0] < committed:
                raise RuntimeError(
                    f"数据集损坏：data/{key} 长度 {array.shape[0]} "
                    f"小于已提交长度 {committed}"
                )
            if array.shape[0] > committed:
                array.resize((committed,) + array.shape[1:])

    def append_episode(self, episode: dict[str, np.ndarray]) -> int:
        if not episode:
            raise ValueError("episode is empty")
        lengths = {key: int(value.shape[0]) for key, value in episode.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"episode arrays are not time-aligned: {lengths}")
        episode_len = next(iter(lengths.values()))
        if episode_len < 2:
            raise ValueError("episode must contain at least two frames")

        old_len = self.n_steps
        new_len = old_len + episode_len
        existing_keys = set(self.data.array_keys())
        incoming_keys = set(episode)
        if existing_keys and existing_keys != incoming_keys:
            raise ValueError(
                "新 episode 字段与已有数据集不一致。"
                f" missing={sorted(existing_keys - incoming_keys)},"
                f" extra={sorted(incoming_keys - existing_keys)}"
            )

        touched: list[str] = []
        try:
            for key, value in episode.items():
                value = np.asarray(value)
                if key not in self.data:
                    self.data.create_dataset(
                        key,
                        shape=(new_len,) + value.shape[1:],
                        chunks=_chunks_for(value),
                        dtype=value.dtype,
                        compressor=self.compressor,
                    )
                    array = self.data[key]
                else:
                    array = self.data[key]
                    if array.shape[1:] != value.shape[1:]:
                        raise ValueError(
                            f"data/{key} shape mismatch: existing "
                            f"{array.shape[1:]}, new {value.shape[1:]}"
                        )
                    if np.dtype(array.dtype) != np.dtype(value.dtype):
                        raise ValueError(
                            f"data/{key} dtype mismatch: existing "
                            f"{array.dtype}, new {value.dtype}"
                        )
                    array.resize((new_len,) + array.shape[1:])
                touched.append(key)
                array[old_len:new_len] = value

            old_n_episodes = self.n_episodes
            self.episode_ends.resize(old_n_episodes + 1)
            self.episode_ends[old_n_episodes] = new_len
            return old_n_episodes
        except Exception:
            for key in touched:
                if key in self.data:
                    array = self.data[key]
                    if old_len == 0 and key not in existing_keys:
                        del self.data[key]
                    elif array.shape[0] > old_len:
                        array.resize((old_len,) + array.shape[1:])
            raise


@dataclass
class CachedValue:
    data: Any = None
    source_stamp: float = math.nan
    receive_monotonic: float = -math.inf
    receive_unix_ns: int = 0
    count: int = 0
    first_monotonic: float = math.nan

    def update(self, data: Any, source_stamp: float = math.nan) -> None:
        now = time.monotonic()
        self.data = data
        self.source_stamp = source_stamp
        self.receive_monotonic = now
        self.receive_unix_ns = time.time_ns()
        self.count += 1
        if not math.isfinite(self.first_monotonic):
            self.first_monotonic = now

    def age(self) -> float:
        return time.monotonic() - self.receive_monotonic

    def rate(self) -> float:
        if self.count < 2 or not math.isfinite(self.first_monotonic):
            return 0.0
        elapsed = time.monotonic() - self.first_monotonic
        return float(self.count - 1) / elapsed if elapsed > 0 else 0.0


class CameraReader:
    def __init__(
        self,
        device: str,
        image_size: int,
        requested_fps: float,
    ):
        self.device = device
        self.image_size = image_size
        self.requested_fps = requested_fps
        self.cache = CachedValue()
        self.error = "尚未启动"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture = None
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise _dependency_error("opencv-python", "opencv-python") from exc

        source: int | str = self.device
        if self.device.isdigit():
            source = int(self.device)
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开相机 {self.device}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, max(self.requested_fps, 10.0))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        self.error = "等待第一帧"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import cv2  # type: ignore

        failures = 0
        while not self._stop.is_set():
            ok, bgr = self._capture.read()
            if not ok or bgr is None:
                failures += 1
                self.error = f"读取失败 ({failures})"
                time.sleep(0.02)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(
                rgb,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
            with self._lock:
                self.cache.update(np.ascontiguousarray(rgb, dtype=np.uint8))
                self.error = ""

    def snapshot(self) -> tuple[np.ndarray | None, float, int]:
        with self._lock:
            if self.cache.data is None:
                return None, math.inf, 0
            return (
                self.cache.data.copy(),
                self.cache.age(),
                self.cache.receive_unix_ns,
            )

    def status(self) -> str:
        with self._lock:
            if self.cache.data is None:
                return self.error
            return (
                f"OK {self.cache.rate():.1f} Hz, "
                f"age={self.cache.age() * 1000:.0f} ms"
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()


class WujiGloveTactileReader:
    """Read the Wuji glove 24x31 tactile map and six semantic zones."""

    def __init__(self, sn: str, device_name: str):
        self.sn = sn
        self.device_name = device_name
        self.matrix = CachedValue()
        self.zones = CachedValue()
        self.error = "尚未启动"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._device = None
        self._matrix_sub = None
        self._zones_sub = None
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            from wuji_sdk import SdkManager
        except ImportError as exc:
            raise _dependency_error("wuji-sdk", "wuji-sdk") from exc

        manager = SdkManager.instance()
        self.error = f"正在连接 {self.sn}"
        self._device = manager.connect(sn=self.sn, device_name=self.device_name)
        self._matrix_sub = self._device.tactile().subscribe()
        self._zones_sub = self._device.tactile_zones().subscribe()
        self.error = "等待手套触觉第一帧"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _drain_latest(subscription: Any) -> Any:
        latest = subscription.recv()
        if latest is None:
            return None
        while True:
            newer = subscription.recv()
            if newer is None:
                return latest
            latest = newer

    def _loop(self) -> None:
        while not self._stop.is_set():
            received = False
            try:
                matrix_frame = self._drain_latest(self._matrix_sub)
                if matrix_frame is not None:
                    raw = np.asarray(matrix_frame.data, dtype=np.float32)
                    if raw.size != int(np.prod(WUJI_GLOVE_MATRIX_SHAPE)):
                        raise ValueError(
                            f"手套触觉长度={raw.size}，期望 "
                            f"{int(np.prod(WUJI_GLOVE_MATRIX_SHAPE))}"
                        )
                    raw = raw.reshape(WUJI_GLOVE_MATRIX_SHAPE)
                    stamp = float(matrix_frame.header.timestamp_us) * 1e-6
                    with self._lock:
                        self.matrix.update(raw, stamp)
                    received = True

                zones_frame = self._drain_latest(self._zones_sub)
                if zones_frame is not None:
                    # Per-area state vector: [mean, max, sum].  Invalid taxels
                    # (<0, including the SDK's -1 sentinel) are excluded.
                    stats = []
                    valid_counts = []
                    for name in WUJI_GLOVE_ZONE_NAMES:
                        values = np.asarray(
                            getattr(zones_frame, name), dtype=np.float32
                        ).reshape(-1)
                        valid = values[np.isfinite(values) & (values >= 0.0)]
                        valid_counts.append(valid.size)
                        if valid.size:
                            stats.append(
                                [float(valid.mean()), float(valid.max()), float(valid.sum())]
                            )
                        else:
                            stats.append([0.0, 0.0, 0.0])
                    stamp = float(zones_frame.header.timestamp_us) * 1e-6
                    with self._lock:
                        self.zones.update(
                            (
                                np.asarray(stats, dtype=np.float32),
                                np.asarray(valid_counts, dtype=np.int32),
                            ),
                            stamp,
                        )
                    received = True
                if received:
                    self.error = ""
                else:
                    time.sleep(0.001)
            except Exception as exc:
                self.error = str(exc)
                time.sleep(0.01)

    def snapshot(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float, float]:
        with self._lock:
            matrix = None if self.matrix.data is None else self.matrix.data.copy()
            if self.zones.data is None:
                zone_stats = None
                zone_counts = None
            else:
                zone_stats = self.zones.data[0].copy()
                zone_counts = self.zones.data[1].copy()
            return matrix, zone_stats, zone_counts, self.matrix.age(), self.zones.age()

    def status(self) -> str:
        with self._lock:
            if self.matrix.data is None or self.zones.data is None:
                return self.error
            valid = self.matrix.data[self.matrix.data >= 0.0]
            value_range = "no-valid-taxel"
            if valid.size:
                value_range = f"range=[{float(valid.min()):.3f}, {float(valid.max()):.3f}]"
            return (
                f"matrix={self.matrix.rate():.1f} Hz, zones={self.zones.rate():.1f} Hz, "
                f"age={max(self.matrix.age(), self.zones.age())*1000:.0f} ms, "
                f"{value_range}"
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for subscription in (self._matrix_sub, self._zones_sub):
            if subscription is not None:
                try:
                    subscription.close()
                except Exception:
                    pass
        if self._device is not None:
            try:
                self._device.disconnect()
            except Exception:
                pass


class EpisodeBuffer:
    def __init__(self, instruction: str):
        instruction = " ".join(instruction.strip().split())
        if not instruction:
            raise ValueError("任务描述不能为空")
        if len(instruction) > 256:
            raise ValueError("任务描述最长 256 个字符")
        if not instruction.endswith((".", "。", "!", "！", "?", "？")):
            instruction += "."
        self.instruction = instruction
        self.start_monotonic = time.monotonic()
        self.frames: dict[str, list[Any]] = {}
        self.dropped = 0

    def append(self, sample: dict[str, Any]) -> None:
        for key, value in sample.items():
            self.frames.setdefault(key, []).append(value)

    @property
    def length(self) -> int:
        if not self.frames:
            return 0
        return len(next(iter(self.frames.values())))

    def as_arrays(self) -> dict[str, np.ndarray]:
        arrays = {key: np.asarray(values) for key, values in self.frames.items()}
        arrays["sub_task_instruction"] = np.asarray(
            [self.instruction] * self.length, dtype=INSTRUCTION_DTYPE
        )
        arrays["right_tactile_sensor_fingers"] = np.asarray(
            [TACTILE_SENSOR_NAME] * self.length, dtype=SENSOR_DTYPE
        )
        arrays["right_tactile_type_fingers"] = np.asarray(
            [TACTILE_TYPE] * self.length, dtype=TYPE_DTYPE
        )
        arrays["right_tactile_sensor_wuji_glove_zones"] = np.asarray(
            [WUJI_GLOVE_SENSOR_NAME] * self.length, dtype=SENSOR_DTYPE
        )
        arrays["right_tactile_type_wuji_glove_zones"] = np.asarray(
            ["state"] * self.length, dtype=TYPE_DTYPE
        )
        return arrays


def _parse_tactile_json(text: str) -> tuple[np.ndarray, float]:
    obj = json.loads(text)
    matrices = []
    for key in FINGER_KEYS:
        matrix = np.asarray(obj[key])
        if matrix.shape != (12, 6):
            raise ValueError(f"{key} shape={matrix.shape}, expected (12, 6)")
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError(f"{key} is not numeric")
        if np.any(matrix < 0) or np.any(matrix > 255):
            raise ValueError(f"{key} contains values outside [0, 255]")
        matrices.append(matrix.astype(np.uint8))
    stamp_obj = obj.get("stamp", {})
    stamp = float(stamp_obj.get("secs", 0)) + float(stamp_obj.get("nsecs", 0)) * 1e-9
    return np.stack(matrices, axis=0), stamp


def _joint20(msg: Any) -> tuple[np.ndarray, np.ndarray, float]:
    position = np.asarray(msg.position, dtype=np.float32)
    if position.shape != (20,):
        raise ValueError(f"JointState.position shape={position.shape}, expected (20,)")
    velocity = np.asarray(msg.velocity, dtype=np.float32)
    if velocity.shape != (20,):
        velocity = np.full((20,), np.nan, dtype=np.float32)
    return position, velocity, _stamp_seconds(msg.header.stamp)


def _build_collector_node(
    args: argparse.Namespace,
    camera: CameraReader | None,
    glove: WujiGloveTactileReader,
):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
    except ImportError as exc:
        raise SystemExit(
            "找不到 ROS 2 Python 环境。先 source /opt/ros/humble/setup.bash "
            "和 LinkerHand install/setup.bash。"
        ) from exc

    class CollectorNode(Node):
        def __init__(self):
            super().__init__("wuji_l20_ftp1_collector")
            self.args = args
            self.camera = camera
            self.glove = glove
            self.lock = threading.RLock()
            self.state = CachedValue()
            self.command = CachedValue()
            self.tactile = CachedValue()
            self.state_names: list[str] = []
            self.command_names: list[str] = []
            self.last_errors = {"state": "", "command": "", "tactile": ""}
            self.episode: EpisodeBuffer | None = None
            self.last_drop_report = 0.0

            self.create_subscription(JointState, args.state_topic, self._state_cb, 20)
            self.create_subscription(JointState, args.command_topic, self._command_cb, 20)
            self.create_subscription(String, args.tactile_topic, self._tactile_cb, 20)
            self.timer = self.create_timer(1.0 / args.fps, self._sample)

        def _state_cb(self, msg: Any) -> None:
            try:
                position, velocity, stamp = _joint20(msg)
                with self.lock:
                    self.state.update((position, velocity), stamp)
                    self.state_names = list(msg.name)
                    self.last_errors["state"] = ""
            except Exception as exc:
                self.last_errors["state"] = str(exc)

        def _command_cb(self, msg: Any) -> None:
            try:
                position, velocity, stamp = _joint20(msg)
                with self.lock:
                    self.command.update((position, velocity), stamp)
                    self.command_names = list(msg.name)
                    self.last_errors["command"] = ""
            except Exception as exc:
                self.last_errors["command"] = str(exc)

        def _tactile_cb(self, msg: Any) -> None:
            try:
                tactile, stamp = _parse_tactile_json(msg.data)
                with self.lock:
                    self.tactile.update(tactile, stamp)
                    self.last_errors["tactile"] = ""
            except Exception as exc:
                self.last_errors["tactile"] = str(exc)

        def readiness_errors(self) -> list[str]:
            errors = []
            with self.lock:
                if self.state.data is None:
                    errors.append("没有收到关节状态")
                elif self.state.age() > self.args.max_age:
                    errors.append(f"关节状态过期 {self.state.age():.2f}s")
                if self.tactile.data is None:
                    errors.append("没有收到 5×12×6 触觉矩阵")
                elif self.tactile.age() > self.args.max_age:
                    errors.append(f"触觉数据过期 {self.tactile.age():.2f}s")
                for source in ("state", "tactile"):
                    if self.last_errors[source]:
                        errors.append(f"{source}: {self.last_errors[source]}")
            if self.camera is not None:
                image, image_age, _ = self.camera.snapshot()
                if image is None:
                    errors.append(f"没有相机图像 ({self.camera.status()})")
                elif image_age > self.args.max_age:
                    errors.append(f"相机图像过期 {image_age:.2f}s")
            glove_matrix, glove_zones, _, glove_matrix_age, glove_zones_age = (
                self.glove.snapshot()
            )
            if glove_matrix is None or glove_zones is None:
                errors.append(f"没有 Wuji 手套触觉 ({self.glove.status()})")
            elif max(glove_matrix_age, glove_zones_age) > self.args.max_age:
                errors.append(
                    "Wuji 手套触觉过期 "
                    f"{max(glove_matrix_age, glove_zones_age):.2f}s"
                )
            return errors

        def start_episode(self, instruction: str) -> None:
            with self.lock:
                if self.episode is not None:
                    raise RuntimeError("已经在采集 episode")
                errors = self.readiness_errors()
                if errors:
                    raise RuntimeError("；".join(errors))
                self.episode = EpisodeBuffer(instruction)

        def take_episode(self) -> EpisodeBuffer:
            with self.lock:
                if self.episode is None:
                    raise RuntimeError("当前没有正在采集的 episode")
                episode = self.episode
                self.episode = None
                return episode

        def discard_episode(self) -> EpisodeBuffer:
            return self.take_episode()

        def restore_episode(self, episode: EpisodeBuffer) -> None:
            """Restore an episode after a disk/schema error so it can be retried."""
            with self.lock:
                if self.episode is not None:
                    raise RuntimeError("cannot restore over an active episode")
                self.episode = episode

        def _sample(self) -> None:
            with self.lock:
                episode = self.episode
                if episode is None:
                    return
                if self.camera is None:
                    image = None
                    image_age = 0.0
                    image_unix_ns = 0
                else:
                    image, image_age, image_unix_ns = self.camera.snapshot()
                (
                    glove_matrix,
                    glove_zone_stats,
                    glove_zone_counts,
                    glove_matrix_age,
                    glove_zones_age,
                ) = self.glove.snapshot()
                invalid = (
                    self.state.data is None
                    or self.tactile.data is None
                    or glove_matrix is None
                    or glove_zone_stats is None
                    or glove_zone_counts is None
                    or self.state.age() > self.args.max_age
                    or self.tactile.age() > self.args.max_age
                    or max(glove_matrix_age, glove_zones_age) > self.args.max_age
                    or (
                        self.camera is not None
                        and (image is None or image_age > self.args.max_age)
                    )
                )
                if invalid:
                    episode.dropped += 1
                    now = time.monotonic()
                    if now - self.last_drop_report > 2.0:
                        print("\n[WARN] 数据源缺失/过期，本帧未写入。输入 status 查看。", flush=True)
                        self.last_drop_report = now
                    return

                state20, velocity20 = self.state.data
                tactile = self.tactile.data
                if self.command.data is None or self.command.age() > self.args.max_age:
                    command20 = np.full((20,), np.nan, dtype=np.float32)
                    command_valid = False
                    command_stamp = math.nan
                    command_age = math.inf
                else:
                    command20 = self.command.data[0]
                    command_valid = True
                    command_stamp = self.command.source_stamp
                    command_age = self.command.age()

                elapsed = time.monotonic() - episode.start_monotonic
                sample = {
                    "timestamps": np.float64(elapsed),
                    "right_hand_joints": state20[ACTIVE_RAW20_INDICES].astype(np.float32),
                    "right_hand_joints_idx": FTP1_HAND_JOINT_INDICES.copy(),
                    "right_tactile_data_fingers": tactile.copy(),
                    "right_tactile_area_fingers": TACTILE_AREAS.copy(),
                    # Wuji glove pressure: the full calibrated 24x31 map is
                    # retained losslessly (including -1 invalid taxels).  The
                    # six FTP-1 state areas carry [mean, max, sum].
                    "right_tactile_data_wuji_glove_zones": glove_zone_stats.copy(),
                    "right_tactile_area_wuji_glove_zones": WUJI_GLOVE_ZONE_AREAS.copy(),
                    "wuji_glove_pressure_matrix_raw": glove_matrix.copy(),
                    "wuji_glove_zone_valid_taxels": glove_zone_counts.copy(),
                    # Diagnostics retained alongside FTP-1's required fields.
                    "right_hand_state_raw20": state20.copy(),
                    "right_hand_velocity_raw20": velocity20.copy(),
                    "right_hand_command_raw20": command20.copy(),
                    "right_hand_command_valid": np.bool_(command_valid),
                    "source_state_timestamp": np.float64(self.state.source_stamp),
                    "source_command_timestamp": np.float64(command_stamp),
                    "source_tactile_timestamp": np.float64(self.tactile.source_stamp),
                    "source_wuji_glove_matrix_timestamp": np.float64(
                        self.glove.matrix.source_stamp
                    ),
                    "source_wuji_glove_zones_timestamp": np.float64(
                        self.glove.zones.source_stamp
                    ),
                    "sample_unix_ns": np.int64(time.time_ns()),
                }
                source_ages = [
                    self.state.age(),
                    command_age,
                    self.tactile.age(),
                    max(glove_matrix_age, glove_zones_age),
                ]
                if self.camera is not None:
                    sample["camera_main_rgb"] = image
                    sample["source_camera_unix_ns"] = np.int64(image_unix_ns)
                    source_ages.append(image_age)
                sample["source_age_ms"] = (
                    np.asarray(source_ages, dtype=np.float32) * np.float32(1000.0)
                )
                episode.append(sample)

        def status_text(self) -> str:
            with self.lock:
                def source_line(name: str, cache: CachedValue) -> str:
                    if cache.data is None:
                        detail = "无数据"
                    else:
                        detail = f"{cache.rate():.1f} Hz, age={cache.age()*1000:.0f} ms"
                    if self.last_errors[name]:
                        detail += f", error={self.last_errors[name]}"
                    return detail

                tactile_range = ""
                if self.tactile.data is not None:
                    tactile_range = (
                        f", range=[{int(self.tactile.data.min())},"
                        f" {int(self.tactile.data.max())}]"
                    )
                if self.episode is None:
                    episode_text = "未采集"
                else:
                    episode_text = (
                        f"采集中 frames={self.episode.length}, "
                        f"dropped={self.episode.dropped}, "
                        f"instruction={self.episode.instruction!r}"
                    )
                return "\n".join(
                    [
                        f"state   : {source_line('state', self.state)}",
                        f"command : {source_line('command', self.command)}",
                        f"tactile : {source_line('tactile', self.tactile)}{tactile_range}",
                        f"glove   : {self.glove.status()}",
                        "camera  : 已关闭（不采集）"
                        if self.camera is None
                        else f"camera  : {self.camera.status()}",
                        f"episode : {episode_text}",
                    ]
                )

    return CollectorNode(), rclpy


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    camera_enabled = not args.no_camera
    source_age_order = ["state", "command", "l20_tactile", "wuji_glove_tactile"]
    if camera_enabled:
        source_age_order.append("camera")
    return {
        "format": "FTP-1-compatible-zarr-v2",
        "profile": "ftp1_with_rgb" if camera_enabled else "tactile_state_no_rgb",
        "collector": "wuji_l20_ftp1_collect.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "robot": "LinkerHand L20 physical hand with G20 SDK mapping",
        "hand_side": "right",
        "sample_rate_hz": float(args.fps),
        "camera_enabled": camera_enabled,
        "camera_device": args.camera if camera_enabled else "",
        "camera_key": "camera_main_rgb" if camera_enabled else "",
        "camera_color_order": "RGB" if camera_enabled else "",
        "image_size": int(args.image_size) if camera_enabled else 0,
        "state_topic": args.state_topic,
        "command_topic": args.command_topic,
        "tactile_topic": args.tactile_topic,
        "right_hand_joints_unit": "vendor_raw_0_255",
        "right_hand_active_raw20_indices": ACTIVE_RAW20_INDICES.tolist(),
        "right_hand_faas_indices": FTP1_HAND_JOINT_INDICES.tolist(),
        "tactile_shape": [5, 12, 6],
        "tactile_finger_order": ["thumb", "index", "middle", "ring", "little"],
        "tactile_function_areas": TACTILE_AREAS.tolist(),
        "tactile_unit": "raw_uint8",
        "tactile_range": [0, 255],
        "wuji_glove_sn": args.glove_sn,
        "wuji_glove_tactile_shape": list(WUJI_GLOVE_MATRIX_SHAPE),
        "wuji_glove_tactile_unit": "sdk_calibrated_pressure_raw_not_newton",
        "wuji_glove_tactile_invalid_value": -1.0,
        "wuji_glove_zone_order": list(WUJI_GLOVE_ZONE_NAMES),
        "wuji_glove_zone_feature_order": ["mean", "max", "sum"],
        "source_age_ms_order": source_age_order,
    }


def collect(args: argparse.Namespace) -> int:
    _import_zarr()
    camera = (
        None
        if args.no_camera
        else CameraReader(args.camera, args.image_size, args.fps)
    )
    glove = WujiGloveTactileReader(args.glove_sn, args.glove_device_name)

    node = None
    rclpy = None
    spin_thread = None
    try:
        if camera is not None:
            camera.start()
        try:
            glove.start()
        except Exception as exc:
            raise SystemExit(
                f"无法连接 Wuji 手套 {args.glove_sn} 的触觉接口。"
                "请确认手套已开机、与电脑在同一网络，并且 SN 正确。"
                f" 原始错误: {exc}"
            ) from exc
        try:
            import rclpy as rclpy_module
        except ImportError as exc:
            raise SystemExit(
                "找不到 ROS 2 Python 环境。先 source /opt/ros/humble/setup.bash "
                "和 LinkerHand install/setup.bash。"
            ) from exc
        rclpy = rclpy_module
        rclpy.init(args=None)
        node, rclpy = _build_collector_node(args, camera, glove)
        writer = FTP1ZarrWriter(args.output, _metadata(args))

        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()

        print("=" * 78)
        print("Wuji L20 -> FTP-1 数据采集器")
        print(f"输出: {writer.path}")
        if camera is None:
            print(f"采样: {args.fps:g} Hz | RGB: 不采集")
        else:
            print(f"采样: {args.fps:g} Hz | RGB: {args.image_size}x{args.image_size}")
        print(f"手套: {args.glove_sn} | 触觉: 24x31 + 六区统计")
        print(f"已有 episodes={writer.n_episodes}, frames={writer.n_steps}")
        print("命令: s=开始  e=成功并保存  d=失败并丢弃  status=状态  q=退出")
        print("=" * 78)

        while rclpy.ok():
            try:
                command = input("ftp1> ").strip().lower()
            except EOFError:
                command = "q"
            if command in ("status", "st", ""):
                print(node.status_text())
                continue
            if command in ("s", "start"):
                if node.episode is not None:
                    print("[ERROR] 已经在采集，请先 e 保存或 d 丢弃。")
                    continue
                instruction = args.instruction
                if not instruction:
                    instruction = input("任务描述（建议英文）: ").strip()
                try:
                    node.start_episode(instruction)
                    print(f"[REC] 开始: {node.episode.instruction}")
                except Exception as exc:
                    print(f"[ERROR] 不能开始: {exc}")
                continue
            if command in ("e", "end", "save"):
                episode = None
                try:
                    episode = node.take_episode()
                    arrays = episode.as_arrays()
                    episode_idx = writer.append_episode(arrays)
                    duration = float(arrays["timestamps"][-1])
                    print(
                        f"[SAVED] episode={episode_idx}, frames={episode.length}, "
                        f"duration={duration:.2f}s, dropped={episode.dropped}"
                    )
                except Exception as exc:
                    if episode is not None:
                        node.restore_episode(episode)
                    print(f"[ERROR] 保存失败，内存中的 episode 已保留，可再次输入 e: {exc}")
                continue
            if command in ("d", "discard", "fail"):
                try:
                    episode = node.discard_episode()
                    print(
                        f"[DISCARDED] frames={episode.length}, "
                        f"duration={time.monotonic()-episode.start_monotonic:.2f}s"
                    )
                except Exception as exc:
                    print(f"[ERROR] {exc}")
                continue
            if command in ("q", "quit", "exit"):
                if node.episode is not None:
                    print("[WARN] 当前 episode 尚未标记成功，已丢弃；用 e 才会写入数据集。")
                    node.discard_episode()
                break
            print("未知命令。可用: s | e | d | status | q")

        return 0
    except KeyboardInterrupt:
        if node is not None and node.episode is not None:
            print("\n[WARN] Ctrl+C：未完成的 episode 已丢弃。")
            node.discard_episode()
        return 130
    finally:
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        glove.close()
        if camera is not None:
            camera.close()


def verify_dataset(path: Path, quiet: bool = False) -> list[str]:
    zarr, _ = _import_zarr()
    errors: list[str] = []
    path = path.expanduser().resolve()
    try:
        root = zarr.open_group(str(path), mode="r")
    except Exception as exc:
        return [f"无法打开 {path}: {exc}"]
    if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
        return ["缺少 FTP-1 根结构 data/ 或 meta/episode_ends"]
    data = root["data"]
    keys = set(data.array_keys())
    camera_enabled = bool(root.attrs.get("camera_enabled", "camera_main_rgb" in keys))
    required_keys = REQUIRED_FTP1_KEYS | REQUIRED_GLOVE_KEYS
    if not camera_enabled:
        required_keys = required_keys - {"camera_main_rgb"}
    missing = required_keys - keys
    if missing:
        errors.append(f"缺少 FTP-1 必需字段: {sorted(missing)}")
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if ends.size == 0:
        errors.append("没有已保存 episode")
        expected_len = 0
    else:
        if np.any(np.diff(np.concatenate(([0], ends))) < 2):
            errors.append("存在少于 2 帧或非递增的 episode")
        expected_len = int(ends[-1])
    for key in sorted(keys):
        if data[key].shape[0] != expected_len:
            errors.append(
                f"data/{key} T={data[key].shape[0]}，应为 {expected_len}"
            )
    expected_shapes = {
        "right_hand_joints": (16,),
        "right_hand_joints_idx": (16,),
        "right_tactile_data_fingers": (5, 12, 6),
        "right_tactile_area_fingers": (5,),
        "right_tactile_data_wuji_glove_zones": (6, 3),
        "right_tactile_area_wuji_glove_zones": (6,),
        "wuji_glove_pressure_matrix_raw": (24, 31),
    }
    if camera_enabled:
        image_size = int(root.attrs.get("image_size", 224))
        expected_shapes["camera_main_rgb"] = (image_size, image_size, 3)
    for key, suffix in expected_shapes.items():
        if key in data and data[key].shape[1:] != suffix:
            errors.append(f"data/{key} shape={data[key].shape}，期望 (T,{suffix})")
    if "camera_main_rgb" in data and data["camera_main_rgb"].dtype != np.uint8:
        errors.append("camera_main_rgb 必须是 uint8 RGB")
    if "right_tactile_data_fingers" in data:
        tactile_array = data["right_tactile_data_fingers"]
        if tactile_array.dtype != np.uint8:
            errors.append("right_tactile_data_fingers 必须是 uint8")
    if expected_len and "right_hand_joints_idx" in data:
        check_idx = np.asarray(data["right_hand_joints_idx"][: min(expected_len, 32)])
        if not np.all(check_idx == FTP1_HAND_JOINT_INDICES):
            errors.append("right_hand_joints_idx 与 L20/G20 FAAS 映射不一致")
    if expected_len and "right_tactile_area_fingers" in data:
        check_area = np.asarray(data["right_tactile_area_fingers"][: min(expected_len, 32)])
        if not np.all(check_area == TACTILE_AREAS):
            errors.append("right_tactile_area_fingers 不是 [0,1,2,3,4]")
    if expected_len and "right_tactile_area_wuji_glove_zones" in data:
        check_area = np.asarray(
            data["right_tactile_area_wuji_glove_zones"][: min(expected_len, 32)]
        )
        if not np.all(check_area == WUJI_GLOVE_ZONE_AREAS):
            errors.append("Wuji 手套触觉区域不是 [拇,食,中,无名,小,掌]")

    if not quiet:
        print(f"dataset : {path}")
        print(f"episodes: {len(ends)}")
        print(f"frames  : {expected_len}")
        for key in sorted(keys):
            print(f"  data/{key}: shape={data[key].shape}, dtype={data[key].dtype}")
        if errors:
            print("RESULT  : FAIL")
            for error in errors:
                print(f"  - {error}")
        elif camera_enabled:
            print("RESULT  : PASS (FTP-1 structure and L20 fields are valid)")
        else:
            print(
                "RESULT  : PASS (camera-free tactile/state Zarr; "
                "FTP-1 official RGB requirement is intentionally disabled)"
            )
    return errors


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="wuji_ftp1_test_") as tmp:
        path = Path(tmp) / "synthetic.zarr"
        fake_args = argparse.Namespace(
            fps=10.0,
            camera="synthetic",
            image_size=224,
            state_topic=DEFAULT_STATE_TOPIC,
            command_topic=DEFAULT_COMMAND_TOPIC,
            tactile_topic=DEFAULT_TACTILE_TOPIC,
            glove_sn=DEFAULT_GLOVE_SN,
            no_camera=False,
        )
        writer = FTP1ZarrWriter(path, _metadata(fake_args))
        buffer = EpisodeBuffer("self test")
        for i in range(3):
            buffer.append(
                {
                    "timestamps": np.float64(i / 10.0),
                    "camera_main_rgb": np.full((224, 224, 3), i, dtype=np.uint8),
                    "right_hand_joints": np.full((16,), i, dtype=np.float32),
                    "right_hand_joints_idx": FTP1_HAND_JOINT_INDICES.copy(),
                    "right_tactile_data_fingers": np.full((5, 12, 6), i, dtype=np.uint8),
                    "right_tactile_area_fingers": TACTILE_AREAS.copy(),
                    "right_tactile_data_wuji_glove_zones": np.full(
                        (6, 3), i, dtype=np.float32
                    ),
                    "right_tactile_area_wuji_glove_zones": WUJI_GLOVE_ZONE_AREAS.copy(),
                    "wuji_glove_pressure_matrix_raw": np.full(
                        WUJI_GLOVE_MATRIX_SHAPE, i, dtype=np.float32
                    ),
                    "wuji_glove_zone_valid_taxels": np.full((6,), 1, dtype=np.int32),
                    "right_hand_state_raw20": np.full((20,), i, dtype=np.float32),
                    "right_hand_velocity_raw20": np.zeros((20,), dtype=np.float32),
                    "right_hand_command_raw20": np.full((20,), i, dtype=np.float32),
                    "right_hand_command_valid": np.bool_(True),
                    "source_state_timestamp": np.float64(i),
                    "source_command_timestamp": np.float64(i),
                    "source_tactile_timestamp": np.float64(i),
                    "source_wuji_glove_matrix_timestamp": np.float64(i),
                    "source_wuji_glove_zones_timestamp": np.float64(i),
                    "source_camera_unix_ns": np.int64(i),
                    "sample_unix_ns": np.int64(i),
                    "source_age_ms": np.zeros((5,), dtype=np.float32),
                }
            )
        arrays = buffer.as_arrays()
        writer.append_episode(arrays)
        # Exercise the existing-array append path and episode_ends contract.
        writer.append_episode(arrays)
        errors = verify_dataset(path)
        if errors:
            return 1

        no_camera_path = Path(tmp) / "synthetic_no_camera.zarr"
        fake_args.no_camera = True
        no_camera_writer = FTP1ZarrWriter(no_camera_path, _metadata(fake_args))
        no_camera_arrays = {
            key: value
            for key, value in arrays.items()
            if key not in {"camera_main_rgb", "source_camera_unix_ns"}
        }
        no_camera_arrays["source_age_ms"] = no_camera_arrays["source_age_ms"][:, :4]
        no_camera_writer.append_episode(no_camera_arrays)
        errors = verify_dataset(no_camera_path)
        if errors:
            return 1
    print("SELF TEST: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and verify LinkerHand L20 data in FTP-1 Zarr format"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect", help="interactive episode collection")
    collect_parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "wuji_datasets" / "l20_glove_nocamera.zarr",
        help="FTP-1 Zarr directory; existing compatible data is appended",
    )
    collect_parser.add_argument("--instruction", default="", help="fixed task instruction")
    collect_parser.add_argument("--fps", type=float, default=10.0)
    collect_parser.add_argument("--camera", default="/dev/video0")
    collect_parser.add_argument(
        "--no-camera",
        action="store_true",
        help="do not capture or require RGB (camera-free tactile/state profile)",
    )
    collect_parser.add_argument("--glove-sn", default=DEFAULT_GLOVE_SN)
    collect_parser.add_argument("--glove-device-name", default="glove_collector")
    collect_parser.add_argument("--image-size", type=int, default=224)
    collect_parser.add_argument("--max-age", type=float, default=0.5)
    collect_parser.add_argument("--state-topic", default=DEFAULT_STATE_TOPIC)
    collect_parser.add_argument("--command-topic", default=DEFAULT_COMMAND_TOPIC)
    collect_parser.add_argument("--tactile-topic", default=DEFAULT_TACTILE_TOPIC)

    verify_parser = sub.add_parser("verify", help="validate an existing dataset")
    verify_parser.add_argument("path", type=Path)
    sub.add_parser("self-test", help="write and verify a synthetic episode")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "collect":
        if args.fps <= 0:
            raise SystemExit("--fps must be > 0")
        if args.image_size <= 0:
            raise SystemExit("--image-size must be > 0")
        if args.max_age <= 0:
            raise SystemExit("--max-age must be > 0")
        return collect(args)
    if args.command == "verify":
        return 1 if verify_dataset(args.path) else 0
    if args.command == "self-test":
        return self_test()
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
