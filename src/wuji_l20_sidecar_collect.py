#!/usr/bin/env python3

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import argparse
from collections import deque
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import zarr


FINGER_NAMES = (
    "index",
    "middle",
    "ring",
    "pinky",
)

GLOVE_TACTILE_SHAPE = (24, 31)
ROBOT_TACTILE_SHAPE = (5, 12, 6)
GLOVE_TACTILE_SUMMARY_NAMES = (
    "valid_taxel_count",
    "active_taxel_count",
    "pressure_sum",
    "pressure_max",
)


def glove_tactile_features(raw):
    """Keep SDK raw tactile data while making invalid taxels explicit.

    The Wuji SDK uses ``-1`` for taxels that do not physically exist.  It is a
    geometry sentinel, not a negative pressure.  Keep that sentinel in both
    stored tactile arrays: a zero is a valid taxel whose measured pressure is
    zero, whereas ``-1`` means that no taxel exists at that geometry location.
    The accompanying mask remains the explicit geometry contract.
    """
    raw = np.asarray(raw, dtype=np.float32).reshape(GLOVE_TACTILE_SHAPE)
    valid_mask = np.isfinite(raw) & (raw >= 0.0)
    pressure = np.where(valid_mask, raw, -1.0).astype(np.float32)
    valid_values = raw[valid_mask]
    summary = np.asarray(
        (
            valid_mask.sum(),
            np.count_nonzero(valid_values > 0.0),
            valid_values.sum() if valid_values.size else 0.0,
            valid_values.max() if valid_values.size else 0.0,
        ),
        dtype=np.float32,
    )
    return raw, valid_mask.astype(np.uint8), pressure, summary


def source_age_ms(sample_ns, source_ns):
    """Return source age in milliseconds, or NaN for an absent timestamp."""
    if source_ns <= 0:
        return np.float32(np.nan)
    return np.float32((sample_ns - source_ns) / 1e6)


class SidecarReceiver:
    def __init__(
        self,
        host="127.0.0.1",
        port=15121,
    ):
        self.host = host
        self.port = port

        self.lock = threading.Lock()
        self.latest = None
        self.latest_rx = 0.0
        self.count = 0
        self.running = True

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.sock.bind(
            (
                self.host,
                self.port,
            )
        )

        self.sock.settimeout(0.5)

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )

        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                raw, _ = self.sock.recvfrom(
                    65535
                )
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                packet = json.loads(
                    raw.decode("utf-8")
                )
            except Exception:
                continue

            if (
                packet.get("schema")
                != "wuji_g20_sidecar_v1"
            ):
                continue

            with self.lock:
                self.latest = packet
                self.latest_rx = time.monotonic()
                self.count += 1

    def get(self):
        with self.lock:
            if self.latest is None:
                return None, None

            return (
                dict(self.latest),
                time.monotonic()
                - self.latest_rx,
            )

    def close(self):
        self.running = False

        try:
            self.sock.close()
        except Exception:
            pass


class GloveSourceReceiver:
    """Queue raw Wuji frames rather than repeatedly overwriting the latest."""

    def __init__(self, host="127.0.0.1", port=15122, max_frames=2048):
        self.port = int(port)
        self.lock = threading.Lock()
        self.frames = deque(maxlen=int(max_frames))
        self.count = 0
        self.dropped = 0
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, self.port))
        self.sock.settimeout(0.5)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                raw, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if packet.get("schema") != "wuji_glove_source_v1":
                continue
            with self.lock:
                if len(self.frames) == self.frames.maxlen:
                    self.dropped += 1
                self.frames.append(packet)
                self.count += 1

    def pop(self):
        with self.lock:
            return self.frames.popleft() if self.frames else None

    def clear(self):
        with self.lock:
            self.frames.clear()

    def status(self):
        with self.lock:
            return self.count, len(self.frames), self.dropped

    def close(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


class Recorder:
    def __init__(
        self,
        receiver,
        glove_source_receiver,
        output,
        fps,
        port,
    ):
        self.receiver = receiver
        self.glove_source_receiver = glove_source_receiver
        self.output = Path(output).expanduser()
        self.fps = float(fps)
        self.port = int(port)

        self.recording = False
        self.frames = []

        self.running = True

        self.root = zarr.open_group(
            str(self.output),
            mode="a",
        )

        # Do not overwrite the root format of an existing v2 data set: it may
        # contain older episodes.  Each newly written episode carries its own
        # v3 schema attribute below.
        self.root.attrs.setdefault("format", "wuji_g20_sidecar_zarr_v5")
        self.root.attrs.setdefault(
            "description", "Wuji glove + G20 teleoperation demonstration sidecar"
        )
        self.root.attrs.setdefault("fps", self.fps)
        self.root.attrs.setdefault("sidecar_port", self.port)
        self.root.attrs["glove_source_port"] = self.glove_source_receiver.port
        self.root.attrs["latest_episode_schema"] = "wuji_g20_sidecar_episode_v5"
        self.root.attrs["glove_tactile_shape"] = list(GLOVE_TACTILE_SHAPE)
        self.root.attrs["glove_tactile_invalid_value"] = -1.0
        self.root.attrs["glove_tactile_pressure_invalid_value"] = -1.0
        self.root.attrs["glove_tactile_pressure_unit"] = "sdk_calibrated_raw_not_newton"
        self.root.attrs["glove_tactile_summary_order"] = list(
            GLOVE_TACTILE_SUMMARY_NAMES
        )

        self.episodes = self.root.require_group(
            "episodes"
        )
        # Last pressure snapshot known to be no newer than the master frame.
        # This prevents a concurrent ROS callback from pairing a future
        # tactile sample with an older 120 Hz master timestamp.
        self._last_causal_robot_pressure = None

        self.thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )

        self.thread.start()

    @staticmethod
    def _merge_glove_source(packet, source):
        """Replace only glove fields; robot fields stay from the bridge."""
        if source is None:
            return packet
        merged = dict(packet)
        for key in (
            "glove_angles",
            "glove_tactile",
            "glove_tactile_seq",
            "glove_tactile_timestamp_us",
        ):
            if source.get(key) is not None:
                merged[key] = source[key]
        return merged

    def _causal_robot_pressure(
        self,
        matrix,
        mass_g,
        source_ns,
        sample_ns,
    ):
        """Return only a pressure sample that is causal for ``sample_ns``.

        The ROS pressure callback and UDP bridge run concurrently.  A just
        received bridge snapshot can therefore contain a source timestamp a
        few milliseconds newer than its old packet timestamp.  Keep that
        sample for the next master step instead of writing negative age.
        """
        source_ns = np.int64(source_ns)
        if source_ns > 0 and source_ns <= sample_ns:
            if np.all(np.isfinite(matrix)) and np.all(np.isfinite(mass_g)):
                self._last_causal_robot_pressure = (
                    matrix.copy(),
                    mass_g.copy(),
                    source_ns,
                )
            return matrix, mass_g, source_ns, np.uint8(0)

        if self._last_causal_robot_pressure is not None:
            previous_matrix, previous_mass, previous_ns = (
                self._last_causal_robot_pressure
            )
            return (
                previous_matrix.copy(),
                previous_mass.copy(),
                np.int64(previous_ns),
                np.uint8(1),
            )

        # Only possible during startup before one causal source sample is
        # available.  Mark it invalid instead of fabricating a timestamp.
        return (
            np.full(ROBOT_TACTILE_SHAPE, np.nan, dtype=np.float32),
            np.full(5, np.nan, dtype=np.float32),
            np.int64(0),
            np.uint8(1),
        )

    def _vec(
        self,
        packet,
        key,
        n,
    ):
        value = packet.get(key)

        if value is None:
            return np.full(
                n,
                np.nan,
                dtype=np.float32,
            )

        try:
            arr = np.asarray(
                value,
                dtype=np.float32,
            ).reshape(-1)
        except Exception:
            return np.full(
                n,
                np.nan,
                dtype=np.float32,
            )

        if arr.size != n:
            return np.full(
                n,
                np.nan,
                dtype=np.float32,
            )

        return arr

    # PRESSURE_ZARR_V2
    def _matrix(
        self,
        packet,
        key,
        shape,
    ):
        value = packet.get(key)

        if value is None:
            return np.full(
                shape,
                np.nan,
                dtype=np.float32,
            )

        try:
            arr = np.asarray(
                value,
                dtype=np.float32,
            )
        except Exception:
            return np.full(
                shape,
                np.nan,
                dtype=np.float32,
            )

        if arr.shape != shape:
            if arr.size != int(np.prod(shape)):
                return np.full(
                    shape,
                    np.nan,
                    dtype=np.float32,
                )
            arr = arr.reshape(shape)

        return arr


    def _i64(
        self,
        packet,
        key,
        default=-1,
    ):
        try:
            value = packet.get(
                key,
                default,
            )

            if value is None:
                value = default

            return np.int64(value)

        except Exception:
            return np.int64(default)


    def _curl_vec(
        self,
        packet,
        key,
    ):
        d = packet.get(key, {})

        if not isinstance(d, dict):
            d = {}

        return np.asarray(
            [
                float(
                    d.get(name, np.nan)
                )
                for name in FINGER_NAMES
            ],
            dtype=np.float32,
        )

    def _convert_frame(self, packet, packet_age_s, sample_ns=None):
        glove = np.asarray(
            packet.get(
                "glove_angles",
                [],
            ),
            dtype=np.float32,
        )

        if glove.shape != (5, 5):
            return None

        actual = self._vec(
            packet,
            "robot_actual_raw20",
            20,
        )

        command = self._vec(
            packet,
            "robot_command_raw20",
            20,
        )

        u16 = self._vec(
            packet,
            "u16",
            16,
        )

        glove_raw = self._matrix(packet, "glove_tactile", GLOVE_TACTILE_SHAPE)
        glove_raw, glove_valid_mask, glove_pressure, glove_summary = (
            glove_tactile_features(glove_raw)
        )

        sample_ns = np.int64(time.time_ns() if sample_ns is None else sample_ns)

        robot_pressure_matrix = self._matrix(
            packet,
            "robot_pressure_matrix",
            ROBOT_TACTILE_SHAPE,
        )

        robot_pressure_mass_g = self._vec(
            packet,
            "robot_pressure_mass_g",
            5,
        )

        glove_timestamp_us = self._i64(packet, "glove_tactile_timestamp_us")
        robot_timestamp_ns = self._i64(packet, "robot_pressure_timestamp_ns")
        (
            robot_pressure_matrix,
            robot_pressure_mass_g,
            robot_timestamp_ns,
            robot_pressure_causal_hold,
        ) = self._causal_robot_pressure(
            robot_pressure_matrix,
            robot_pressure_mass_g,
            robot_timestamp_ns,
            sample_ns,
        )
        glove_timestamp_ns = glove_timestamp_us * np.int64(1000)
        robot_pressure_matrix_sum = np.where(
            np.isfinite(robot_pressure_matrix), robot_pressure_matrix, 0.0
        ).sum(axis=(1, 2), dtype=np.float32)
        robot_pressure_source_valid = np.uint8(
            np.all(np.isfinite(robot_pressure_matrix))
            and np.all(np.isfinite(robot_pressure_mass_g))
        )

        return {
            "timestamp_ns": sample_ns,

            "glove_angles":
                glove.copy(),

            # v2-compatible raw vector.  -1 remains untouched here.
            "glove_tactile": glove_raw.reshape(-1),
            # v3 tactile representation for statistics and model inputs.
            "glove_tactile_valid_mask": glove_valid_mask,
            "glove_tactile_pressure": glove_pressure,
            "glove_tactile_summary": glove_summary,
            "glove_tactile_source_valid": np.uint8(np.any(glove_valid_mask)),
            "glove_tactile_seq": self._i64(packet, "glove_tactile_seq"),
            "glove_tactile_timestamp_us": glove_timestamp_us,
            "glove_tactile_age_ms": source_age_ms(sample_ns, glove_timestamp_ns),

            "robot_pressure_matrix": robot_pressure_matrix,
            "robot_pressure_mass_g": robot_pressure_mass_g,
            "robot_pressure_matrix_sum": robot_pressure_matrix_sum,
            "robot_pressure_mass_minus_matrix_sum": (
                robot_pressure_mass_g - robot_pressure_matrix_sum
            ).astype(np.float32),
            "robot_pressure_source_valid": robot_pressure_source_valid,
            "robot_pressure_timestamp_ns": robot_timestamp_ns,
            "robot_pressure_age_ms": source_age_ms(sample_ns, robot_timestamp_ns),
            "robot_pressure_causal_hold": robot_pressure_causal_hold,
            "sidecar_packet_age_ms": np.float32(packet_age_s * 1000.0),

            "u16":
                u16,

            "robot_command_raw20":
                command,

            "robot_actual_raw20":
                actual,

            "finger_curls":
                self._curl_vec(
                    packet,
                    "finger_curls",
                ),

            "finger_proximal_curls":
                self._curl_vec(
                    packet,
                    "finger_proximal_curls",
                ),

            "finger_distal_curls":
                self._curl_vec(
                    packet,
                    "finger_distal_curls",
                ),

            "palm_activation":
                np.float32(
                    packet.get(
                        "palm_activation",
                        0.0,
                    )
                ),

            "palm_progress":
                np.float32(
                    packet.get(
                        "palm_progress",
                        0.0,
                    )
                ),

            "thumb_activity":
                np.float32(
                    packet.get(
                        "thumb_activity",
                        0.0,
                    )
                ),

            "armed":
                np.uint8(
                    bool(
                        packet.get(
                            "armed",
                            False,
                        )
                    )
                ),
        }

    def _sample_loop(self):
        dt = 1.0 / self.fps

        next_t = time.monotonic()

        while self.running:
            now = time.monotonic()

            if now < next_t:
                time.sleep(
                    min(
                        next_t - now,
                        0.01,
                    )
                )
                continue

            next_t += dt

            if not self.recording:
                continue

            packet, age = self.receiver.get()

            if (
                packet is None
                or age is None
                or age > 0.5
            ):
                continue

            # The bridge packet is a latest-value snapshot.  When available,
            # use the next queued producer frame for glove fields so Zarr keeps
            # one real Wuji tactile sequence per sampling step.
            source = self.glove_source_receiver.pop()
            # Master time is the moment this recorder has both source inputs,
            # not the potentially older timestamp embedded in the UDP packet.
            sample_ns = time.time_ns()
            frame = self._convert_frame(
                self._merge_glove_source(packet, source),
                age,
                sample_ns=sample_ns,
            )

            if frame is not None:
                self.frames.append(frame)

    def episode_count(self):
        return len(
            [
                key
                for key in self.episodes.group_keys()
                if key.startswith("episode_")
            ]
        )

    def total_frames(self):
        total = 0

        for key in self.episodes.group_keys():
            if not key.startswith("episode_"):
                continue

            group = self.episodes[key]

            if "timestamp_ns" in group:
                total += int(
                    group[
                        "timestamp_ns"
                    ].shape[0]
                )

        return total

    def start(self):
        if self.recording:
            print("已经在采集中")
            return

        self.frames = []
        # Do not leak pre-episode source frames into this demonstration.
        self.glove_source_receiver.clear()
        self.recording = True

        print(
            "[REC] episode 开始"
        )

    def discard(self):
        if not self.recording:
            print("当前没有 episode")
            return

        n = len(self.frames)

        self.recording = False
        self.frames = []

        print(
            f"[DROP] 丢弃 {n} 帧"
        )

    def save(self):
        if not self.recording:
            print("当前没有 episode")
            return

        self.recording = False

        frames = self.frames
        self.frames = []

        if not frames:
            print(
                "[SAVE REFUSED] 没有有效帧"
            )
            return

        index = self.episode_count()

        name = (
            f"episode_{index:06d}"
        )

        group = self.episodes.create_group(
            name
        )

        group.attrs[
            "success"
        ] = True

        group.attrs[
            "fps"
        ] = self.fps

        group.attrs[
            "num_frames"
        ] = len(frames)

        group.attrs[
            "created_unix_ns"
        ] = time.time_ns()

        group.attrs["format"] = "wuji_g20_sidecar_episode_v5"
        group.attrs["glove_tactile_shape"] = list(GLOVE_TACTILE_SHAPE)
        group.attrs["glove_tactile_invalid_value"] = -1.0
        group.attrs["glove_tactile_pressure_invalid_value"] = -1.0
        group.attrs["glove_tactile_pressure_unit"] = "sdk_calibrated_raw_not_newton"
        group.attrs["glove_tactile_summary_order"] = list(GLOVE_TACTILE_SUMMARY_NAMES)
        group.attrs["robot_pressure_unit"] = "vendor_matrix_raw_sum_g"
        group.attrs[
            "robot_pressure_timestamp_semantics"
        ] = "causal_source_timestamp_ns; future bridge samples are deferred"

        def stack(key):
            return np.stack(
                [
                    frame[key]
                    for frame in frames
                ],
                axis=0,
            )

        datasets = {
            "timestamp_ns":
                np.asarray(
                    [
                        f["timestamp_ns"]
                        for f in frames
                    ],
                    dtype=np.int64,
                ),

            "glove_angles":
                stack(
                    "glove_angles"
                ).astype(
                    np.float32
                ),

            # PRESSURE_ZARR_V2
            "glove_tactile":
                stack(
                    "glove_tactile"
                ).astype(
                    np.float32
                ),

            "glove_tactile_valid_mask":
                stack(
                    "glove_tactile_valid_mask"
                ).astype(
                    np.uint8
                ),

            "glove_tactile_pressure":
                stack(
                    "glove_tactile_pressure"
                ).astype(
                    np.float32
                ),

            "glove_tactile_summary":
                stack(
                    "glove_tactile_summary"
                ).astype(
                    np.float32
                ),

            "glove_tactile_source_valid":
                np.asarray(
                    [f["glove_tactile_source_valid"] for f in frames],
                    dtype=np.uint8,
                ),

            "glove_tactile_seq":
                np.asarray(
                    [
                        f["glove_tactile_seq"]
                        for f in frames
                    ],
                    dtype=np.int64,
                ),

            "glove_tactile_timestamp_us":
                np.asarray(
                    [
                        f["glove_tactile_timestamp_us"]
                        for f in frames
                    ],
                    dtype=np.int64,
                ),

            "glove_tactile_age_ms":
                np.asarray(
                    [f["glove_tactile_age_ms"] for f in frames],
                    dtype=np.float32,
                ),

            "robot_pressure_matrix":
                stack(
                    "robot_pressure_matrix"
                ).astype(
                    np.float32
                ),

            "robot_pressure_mass_g":
                stack(
                    "robot_pressure_mass_g"
                ).astype(
                    np.float32
                ),

            "robot_pressure_matrix_sum":
                stack(
                    "robot_pressure_matrix_sum"
                ).astype(
                    np.float32
                ),

            "robot_pressure_mass_minus_matrix_sum":
                stack(
                    "robot_pressure_mass_minus_matrix_sum"
                ).astype(
                    np.float32
                ),

            "robot_pressure_source_valid":
                np.asarray(
                    [f["robot_pressure_source_valid"] for f in frames],
                    dtype=np.uint8,
                ),

            "robot_pressure_timestamp_ns":
                np.asarray(
                    [
                        f["robot_pressure_timestamp_ns"]
                        for f in frames
                    ],
                    dtype=np.int64,
                ),

            "robot_pressure_age_ms":
                np.asarray(
                    [f["robot_pressure_age_ms"] for f in frames],
                    dtype=np.float32,
                ),

            "robot_pressure_causal_hold":
                np.asarray(
                    [f["robot_pressure_causal_hold"] for f in frames],
                    dtype=np.uint8,
                ),

            "sidecar_packet_age_ms":
                np.asarray(
                    [f["sidecar_packet_age_ms"] for f in frames],
                    dtype=np.float32,
                ),

            "u16":
                stack(
                    "u16"
                ).astype(
                    np.float32
                ),

            "robot_command_raw20":
                stack(
                    "robot_command_raw20"
                ).astype(
                    np.float32
                ),

            "robot_actual_raw20":
                stack(
                    "robot_actual_raw20"
                ).astype(
                    np.float32
                ),

            "finger_curls":
                stack(
                    "finger_curls"
                ).astype(
                    np.float32
                ),

            "finger_proximal_curls":
                stack(
                    "finger_proximal_curls"
                ).astype(
                    np.float32
                ),

            "finger_distal_curls":
                stack(
                    "finger_distal_curls"
                ).astype(
                    np.float32
                ),

            "palm_activation":
                np.asarray(
                    [
                        f[
                            "palm_activation"
                        ]
                        for f in frames
                    ],
                    dtype=np.float32,
                ),

            "palm_progress":
                np.asarray(
                    [
                        f[
                            "palm_progress"
                        ]
                        for f in frames
                    ],
                    dtype=np.float32,
                ),

            "thumb_activity":
                np.asarray(
                    [
                        f[
                            "thumb_activity"
                        ]
                        for f in frames
                    ],
                    dtype=np.float32,
                ),

            "armed":
                np.asarray(
                    [
                        f["armed"]
                        for f in frames
                    ],
                    dtype=np.uint8,
                ),
        }

        for key, data in datasets.items():
            if data.ndim == 1:
                chunks = (
                    min(
                        len(data),
                        256,
                    ),
                )
            else:
                chunks = (
                    min(
                        data.shape[0],
                        256,
                    ),
                    *data.shape[1:],
                )

            group.create_dataset(
                key,
                data=data,
                shape=data.shape,
                dtype=data.dtype,
                chunks=chunks,
            )

        print(
            f"[SAVE] {name}: "
            f"{len(frames)} 帧"
        )

    def status(self):
        packet, age = self.receiver.get()
        source_count, source_queued, source_dropped = (
            self.glove_source_receiver.status()
        )

        print()
        print(
            "recording =",
            self.recording,
        )
        print(
            "current frames =",
            len(self.frames),
        )
        print(
            "received UDP =",
            self.receiver.count,
        )
        print(
            "glove source =",
            f"received={source_count}",
            f"queued={source_queued}",
            f"dropped={source_dropped}",
        )

        if packet is None:
            print(
                "V6 sidecar = NO DATA"
            )
        else:
            print(
                "V6 sidecar age = "
                f"{age:.3f} s"
            )

            print(
                "armed =",
                packet.get(
                    "armed",
                    False,
                ),
            )

            frame = self._convert_frame(packet, age)
            if frame is not None:
                summary = frame["glove_tactile_summary"]
                print(
                    "glove tactile = "
                    f"valid={int(summary[0])}, active={int(summary[1])}, "
                    f"sum={summary[2]:.4f}, max={summary[3]:.4f}, "
                    f"age={frame['glove_tactile_age_ms']:.1f} ms"
                )
                residual = frame["robot_pressure_mass_minus_matrix_sum"]
                finite_residual = residual[np.isfinite(residual)]
                max_residual = (
                    float(np.max(np.abs(finite_residual)))
                    if finite_residual.size
                    else float("nan")
                )
                print(
                    "G20 tactile   = "
                    f"valid={int(frame['robot_pressure_source_valid'])}, "
                    f"max |mass-matrix.sum|={max_residual:.4f}, "
                    f"age={frame['robot_pressure_age_ms']:.1f} ms"
                )

        print(
            "saved episodes =",
            self.episode_count(),
        )

        print(
            "saved frames =",
            self.total_frames(),
        )

        print()

    def close(self):
        self.recording = False
        self.running = False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=15121,
    )

    parser.add_argument(
        "--glove-source-port",
        type=int,
        default=15122,
        help="raw Wuji producer stream emitted by the latest teleop",
    )

    args = parser.parse_args()

    if not 0 < args.fps <= 120:
        parser.error("--fps 必须在 (0, 120]；120 是当前 V7 链路的最高采样率")

    receiver = SidecarReceiver(
        port=args.port,
    )

    glove_source_receiver = GloveSourceReceiver(
        port=args.glove_source_port,
    )

    recorder = Recorder(
        receiver=receiver,
        glove_source_receiver=glove_source_receiver,
        output=args.output,
        fps=args.fps,
        port=args.port,
    )

    print("=" * 78)
    print(
        "Wuji V6 -> G20 demonstration sidecar collector"
    )
    print(
        "输出:",
        Path(
            args.output
        ).expanduser(),
    )
    print(
        f"采样: {args.fps:g} Hz"
    )
    print(
        "数据源: V6 sidecar UDP "
        f"127.0.0.1:{args.port}"
    )
    print(
        "手套原始帧: UDP "
        f"127.0.0.1:{args.glove_source_port}"
    )
    print(
        "不会连接 Wuji SDK"
    )
    print(
        "命令: "
        "s=开始  "
        "e=成功并保存  "
        "d=失败并丢弃  "
        "status=状态  "
        "q=退出"
    )
    print("=" * 78)

    try:
        while True:
            cmd = input(
                "sidecar> "
            ).strip().lower()

            if cmd == "s":
                recorder.start()

            elif cmd == "e":
                recorder.save()

            elif cmd == "d":
                recorder.discard()

            elif cmd == "status":
                recorder.status()

            elif cmd == "q":
                break

            elif not cmd:
                continue

            else:
                print(
                    "未知命令。"
                    "可用: "
                    "s | e | d | status | q"
                )

    except KeyboardInterrupt:
        print()

    finally:
        recorder.close()
        receiver.close()
        glove_source_receiver.close()


if __name__ == "__main__":
    main()
