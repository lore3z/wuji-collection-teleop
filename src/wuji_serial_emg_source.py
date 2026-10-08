#!/usr/bin/env python3
"""Buffered process bridge for the Wavletech serial EMG/IMU receiver."""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import json
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


class SerialEmgSource:
    """Expose native serial EMG samples through the collector source contract."""

    source_name = "Wavletech armband via USB serial receiver"
    emg_layout = "8 signed 24-bit channels"
    emg_unit = "microvolt"
    clip_min = -8_388_608
    clip_max = 8_388_607

    def __init__(
        self,
        python: str,
        tty: str,
        *,
        baud: int = 921600,
        max_queue: int = 262_144,
        silence_timeout_s: float = 0.35,
    ):
        self.python = str(Path(python).expanduser())
        self.tty = str(tty)
        self.baud = int(baud)
        self.silence_timeout_s = float(silence_timeout_s)
        if self.baud <= 0 or self.silence_timeout_s <= 0:
            raise ValueError("serial EMG baud and silence timeout must be positive")
        self.script = Path(__file__).resolve().parents[1] / "runtime" / "wavletech_serial_stream.py"
        self.lock = threading.Lock()
        # timestamp, monotonic source sequence, movement placeholder, EMG,
        # host serial-arrival timestamp
        self.frames: deque[tuple[int, int, int, np.ndarray, int]] = deque(maxlen=max_queue)
        # timestamp, monotonic source sequence, [gx,gy,gz,ax,ay,az], arrival
        self.imu_frames: deque[tuple[int, int, np.ndarray, int]] = deque(maxlen=max_queue)
        self.stderr_tail: deque[str] = deque(maxlen=40)
        self.events: deque[str] = deque(maxlen=12)
        self.process: subprocess.Popen[str] | None = None
        self.stdout_thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None
        self.ready_event = threading.Event()
        self.error = "not started"
        self.connection_state = "not_started"
        self.connection_id = 0
        self.reconnects = 0
        self.resolved_tty = self.tty
        self.driver = "wavletech-serial-v1"
        self.received = 0
        self.imu_received = 0
        self.dropped = 0
        self.last_seq = 0
        self.last_mono = 0.0
        self.last_timestamp_ns = 0
        self.sample_mono_window: deque[float] = deque(maxlen=20_000)
        self.interruption_id = 0
        self.interruption_reason = ""

    def start(
        self,
        timeout_s: float = 20.0,
        stability_s: float = 3.0,
        min_stable_hz: float = 1.0,
        stability_timeout_s: float = 30.0,
        wait_for_stability: bool = True,
    ) -> None:
        if not Path(self.python).is_file():
            raise RuntimeError(f"project Python not found: {self.python}")
        if not self.script.is_file():
            raise RuntimeError(f"serial EMG helper missing: {self.script}")
        self.process = subprocess.Popen(
            [
                self.python, "-u", str(self.script), "--tty", self.tty,
                "--baud", str(self.baud), "--silence-timeout-s", str(self.silence_timeout_s),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        with self.lock:
            self.error = ""
            self.connection_state = "starting"
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()
        if not wait_for_stability:
            return
        deadline = time.monotonic() + timeout_s
        while not self.ready_event.wait(0.1):
            if self.process.poll() is not None:
                raise RuntimeError(self.error or f"serial EMG bridge exited {self.process.returncode}")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"serial EMG did not become ready: {self.diagnostic_tail()}")
        deadline = time.monotonic() + stability_timeout_s
        while time.monotonic() < deadline:
            if self.stable(stability_s, min_stable_hz):
                return
            time.sleep(0.1)
        raise RuntimeError(f"serial EMG did not become stable: {self.status()}")

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.stderr_tail.append(f"bad stdout: {line.strip()}")
                continue
            kind = event.get("kind")
            if kind == "ready":
                with self.lock:
                    next_id = int(event.get("connection_id", self.connection_id + 1))
                    if self.connection_id and next_id != self.connection_id:
                        self.reconnects += 1
                        self.sample_mono_window.clear()
                    self.connection_id = next_id
                    self.connection_state = "connected"
                    self.resolved_tty = str(event.get("tty", self.tty))
                    self.driver = str(event.get("driver", self.driver))
                    self.error = ""
                    self.events.append(f"connected#{next_id} tty={self.resolved_tty}")
                self.ready_event.set()
                continue
            if kind == "state":
                state = str(event.get("state", "unknown"))
                message = str(event.get("message", ""))
                with self.lock:
                    if state in {"recovering", "disconnected"}:
                        self.interruption_id += 1
                        self.interruption_reason = message or state
                        self.sample_mono_window.clear()
                    self.connection_state = state
                    if message:
                        self.error = message
                    self.events.append(f"{state}: {message}".strip())
                continue
            if kind == "gap":
                missing = int(event.get("missing_packets", 0))
                reason = str(event.get("reason", "hardware packet sequence discontinuity"))
                with self.lock:
                    self.dropped += max(missing, 1)
                    self.interruption_id += 1
                    self.interruption_reason = (
                        f"{reason}: {event.get('previous_seq')}->{event.get('hardware_seq')}"
                    )
                    self.events.append(self.interruption_reason)
                continue
            if kind == "error":
                with self.lock:
                    self.error = str(event.get("message", "unknown serial EMG error"))
                self.ready_event.set()
                continue
            if kind == "imu":
                try:
                    values = np.asarray(
                        [*event["gyro_rad_s"], *event["accel_m_s2"]], dtype=np.float32
                    ).reshape(6)
                    frame = (
                        int(event["timestamp_ns"]), int(event["seq"]), values,
                        int(event.get("arrival_timestamp_ns", event["timestamp_ns"])),
                    )
                except Exception:
                    with self.lock:
                        self.dropped += 1
                    continue
                with self.lock:
                    if len(self.imu_frames) == self.imu_frames.maxlen:
                        self.dropped += 1
                    self.imu_frames.append(frame)
                    self.imu_received += 1
                continue
            if kind != "sample":
                continue
            try:
                timestamp_ns = int(event["timestamp_ns"])
                arrival_ns = int(event.get("arrival_timestamp_ns", timestamp_ns))
                seq = int(event["seq"])
                emg = np.asarray(event["emg"], dtype=np.int32).reshape(8)
            except Exception:
                with self.lock:
                    self.dropped += 1
                continue
            with self.lock:
                now_mono = time.monotonic()
                if self.last_mono and now_mono - self.last_mono > self.silence_timeout_s:
                    self.interruption_id += 1
                    self.interruption_reason = (
                        f"serial EMG interrupted for {(now_mono - self.last_mono) * 1000:.0f} ms"
                    )
                    self.sample_mono_window.clear()
                self.last_seq = max(self.last_seq, seq)
                if len(self.frames) == self.frames.maxlen:
                    self.dropped += 1
                self.frames.append((timestamp_ns, seq, 0, emg, arrival_ns))
                self.received += 1
                self.last_mono = now_mono
                self.last_timestamp_ns = timestamp_ns
                self.sample_mono_window.append(now_mono)

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            if line.strip():
                self.stderr_tail.append(line.strip())

    def drain(self) -> list[tuple[int, int, int, np.ndarray, int]]:
        with self.lock:
            values = list(self.frames)
            self.frames.clear()
        return values

    def drain_imu(self) -> list[tuple[int, int, np.ndarray, int]]:
        with self.lock:
            values = list(self.imu_frames)
            self.imu_frames.clear()
        return values

    def clear(self) -> None:
        self.drain()
        self.drain_imu()

    def age_ms(self) -> float:
        with self.lock:
            return (time.monotonic() - self.last_mono) * 1000.0 if self.last_mono else float("inf")

    def current_connection_id(self) -> int:
        with self.lock:
            return self.connection_id

    def interruption_snapshot(self) -> tuple[int, str]:
        with self.lock:
            return self.interruption_id, self.interruption_reason

    def healthy(self, max_age_s: float = 0.5) -> bool:
        with self.lock:
            alive = self.process is not None and self.process.poll() is None
            age = time.monotonic() - self.last_mono if self.last_mono else float("inf")
            return alive and self.connection_state == "connected" and age <= max_age_s

    def stable(self, duration_s: float = 3.0, min_hz: float = 1.0, max_age_s: float = 0.5) -> bool:
        with self.lock:
            now = time.monotonic()
            cutoff = now - duration_s
            recent = [stamp for stamp in self.sample_mono_window if stamp >= cutoff]
            span = recent[-1] - recent[0] if len(recent) >= 2 else 0.0
            rate = (len(recent) - 1) / max(span, 1e-9)
            alive = self.process is not None and self.process.poll() is None
            age = now - self.last_mono if self.last_mono else float("inf")
            return (
                alive and self.connection_state == "connected"
                and span >= duration_s - min(0.02, duration_s * 0.1)
                and rate >= min_hz and age <= max_age_s
            )

    def status(self) -> str:
        with self.lock:
            now = time.monotonic()
            age_ms = (now - self.last_mono) * 1000.0 if self.last_mono else float("inf")
            recent = [stamp for stamp in self.sample_mono_window if now - stamp <= 5.0]
            rate = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-9) if len(recent) >= 2 else 0.0
            alive = self.process is not None and self.process.poll() is None
            return (
                f"emg={self.received}, imu={self.imu_received}, queued={len(self.frames)}, "
                f"dropped={self.dropped}, age_ms={age_ms:.0f}, rate_hz={rate:.1f}, "
                f"state={self.connection_state}, connection={self.connection_id}, "
                f"reconnects={self.reconnects}, alive={alive}, driver={self.driver}, "
                f"tty={self.resolved_tty}, baud={self.baud} {self.error}"
            )

    def diagnostic_tail(self) -> str:
        with self.lock:
            values = [self.events[-1] if self.events else "", self.stderr_tail[-1] if self.stderr_tail else ""]
        return " | ".join(value for value in values if value) or "no bridge diagnostics"

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self.process = None
