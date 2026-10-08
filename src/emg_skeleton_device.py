"""Nonblocking localhost input with the WujiGloveDevice finger-data interface.

The wire skeleton is the *raw*, right-hand Wuji/MediaPipe 21-point skeleton in
meters. It must not already have apply_mediapipe_transformations applied.
``quality.valid`` reports stream validity, not model accuracy or confidence.
No SDK, PyTorch, ROS, CAN or hardware dependencies are imported here.
"""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import json
import math
import socket
import time
from collections import Counter
from typing import Optional

import numpy as np


DEFAULT_PORT = 17621
MAX_PACKET_BYTES = 16384
FINGER_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8),
                 (0, 9, 10, 11, 12), (0, 13, 14, 15, 16),
                 (0, 17, 18, 19, 20))


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def validate_skeleton(value):
    """Reject wrong units and singular geometry before the baseline's SVD/FK."""
    arr = np.asarray(value)
    if arr.dtype.kind not in "iuf":
        raise ValueError("skeleton coordinates must be numeric, not strings or booleans")
    arr = arr.astype(np.float64)
    if arr.shape != (21, 3) or not np.all(np.isfinite(arr)):
        raise ValueError("skeleton must contain finite 21 x 3 coordinates")
    centered = arr - arr[0]
    extent = np.linalg.norm(centered, axis=1).max()
    if not 0.025 <= extent <= 0.4:
        raise ValueError("skeleton scale is incompatible with meters")
    for chain in FINGER_CHAINS:
        lengths = np.linalg.norm(np.diff(arr[list(chain)], axis=0), axis=1)
        if np.any(lengths < 1e-5) or np.any(lengths > 0.15):
            raise ValueError("skeleton contains collapsed or implausible bones")
    palm_area = np.linalg.norm(np.cross(centered[5], centered[9]))
    if palm_area < 1e-6:
        raise ValueError("wrist, index MCP and middle MCP are collinear")
    return arr.astype(np.float32)


class EMGSkeletonDevice:
    """Latest-frame provider; cached input expires using a monotonic clock.

    Wall-clock publication age rejects delayed UDP; receiver monotonic age
    expires a cached frame even after a wall-clock adjustment. Historical replay
    ``source_timestamp`` is informational; it is not compared to today's clock.
    Sequence numbers prevent duplicates from keeping a dead stream alive. A
    different publisher session may take over only once the previous one expires.
    """

    def __init__(self, hand_side: Optional[str] = "right", device_name="emg",
                 sn=None, *, host="127.0.0.1", port=DEFAULT_PORT,
                 max_age_s=0.20, future_tolerance_s=0.05,
                 max_source_age_s=0.20, monotonic=time.monotonic,
                 wall_time=time.time):
        if hand_side not in (None, "right"):
            raise ValueError("EMG2Pose backend currently supports only right hand")
        if sn is not None:
            raise ValueError("EMGSkeletonDevice uses UDP, not a glove serial number")
        if socket.gethostbyname(host) != "127.0.0.1":
            raise ValueError("EMG skeleton input is restricted to 127.0.0.1")
        if not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        for name, value in (("max_age_s", max_age_s),
                            ("max_source_age_s", max_source_age_s)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if not math.isfinite(future_tolerance_s) or future_tolerance_s < 0:
            raise ValueError("future_tolerance_s must be finite and nonnegative")
        self._mono, self._wall = monotonic, wall_time
        self.max_age_s = float(max_age_s)
        self.future_tolerance_s = float(future_tolerance_s)
        self.max_source_age_s = float(max_source_age_s)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.bind(("127.0.0.1", port))
            self._socket.setblocking(False)
        except BaseException:
            self._socket.close()
            raise
        self.address = self._socket.getsockname()
        self.device_name = device_name
        self._last = None
        self._last_rx = -math.inf
        self._last_valid_rx = -math.inf
        self._last_seq = -1
        self._last_timestamp = -math.inf
        self._session = None
        self._endpoint = None
        self._closed = False
        self._metadata = {}
        self.stats = Counter()

    def _accept(self, payload, endpoint):
        now, wall = self._mono(), self._wall()
        try:
            if len(payload) > MAX_PACKET_BYTES:
                raise ValueError("oversize packet")
            def invalid_constant(value):
                raise ValueError("nonfinite JSON constant " + value)
            msg = json.loads(payload.decode("utf-8"), parse_constant=invalid_constant)
            if not isinstance(msg, dict) or msg.get("source") != "emg2pose":
                raise ValueError("wrong source")
            stamp = _number(msg["timestamp"], "timestamp")
            if not -self.future_tolerance_s <= wall - stamp <= self.max_age_s:
                raise ValueError("publication timestamp is stale or in the future")
            source_stamp = _number(msg["source_timestamp"], "source_timestamp")
            seq = msg["seq"]
            if isinstance(seq, bool) or not isinstance(seq, int) or not 0 <= seq < 2**63:
                raise ValueError("seq must be a nonnegative integer")
            session = msg.get("session_id", "legacy")
            if not isinstance(session, str) or not 1 <= len(session) <= 128:
                raise ValueError("invalid session_id")
            quality = msg["quality"]
            if not isinstance(quality, dict) or not isinstance(quality.get("valid"), bool):
                raise ValueError("quality.valid must be a boolean")
            if "source_age_ms" in quality:
                source_age = _number(quality["source_age_ms"], "quality.source_age_ms")
                if source_age < 0 or source_age > self.max_source_age_s * 1000:
                    quality = dict(quality, valid=False, reason="source age exceeds receiver limit")
            skeleton = validate_skeleton(msg["skeleton"]) if quality["valid"] else None
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            self.stats["rejected"] += 1
            self._metadata["last_rejection"] = str(exc)
            return

        same_stream = (session, endpoint) == (self._session, self._endpoint)
        if self._session is not None and not same_stream and now - self._last_rx <= self.max_age_s:
            self.stats["other_publisher"] += 1
            return
        if same_stream and (seq <= self._last_seq or stamp < self._last_timestamp):
            self.stats["out_of_order"] += 1
            return
        if same_stream and seq > self._last_seq + 1:
            self.stats["missing_sequences"] += seq - self._last_seq - 1
        if self._session is not None and not same_stream:
            self.stats["session_changes"] += 1
        self._session, self._endpoint = session, endpoint
        self._last_seq, self._last_timestamp = seq, stamp
        self._last_rx = now
        self._last = skeleton
        if skeleton is not None:
            self._last_valid_rx = now
            self.stats["accepted"] += 1
        else:
            self.stats["invalid_quality"] += 1
        self._metadata.update(seq=seq, timestamp=stamp, source_timestamp=source_stamp,
                              session_id=session, quality=quality)

    def get_fingers_data(self):
        """Return a copy of fresh right-hand points, or None on invalid/stale input."""
        if self._closed:
            return {"left_fingers": None, "right_fingers": None}
        for _ in range(512):
            try:
                payload, endpoint = self._socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            self.stats["received"] += 1
            self._accept(payload, endpoint)
        fresh = (self._last is not None and self._mono() - self._last_rx <= self.max_age_s)
        return {"left_fingers": None, "right_fingers": self._last.copy() if fresh else None}

    @property
    def metadata(self):
        result = dict(self._metadata, stats=dict(self.stats))
        age = self._mono() - self._last_rx
        result["receiver_age_ms"] = age * 1000 if math.isfinite(age) else None
        result["fresh"] = self._last is not None and age <= self.max_age_s
        return result

    def cleanup(self):
        if not self._closed:
            self._socket.close()
            self._last = None
            self._closed = True

    close = cleanup

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()
