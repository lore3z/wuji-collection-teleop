#!/usr/bin/env python3
"""Process bridge buffering native 8-channel Myo RAW EMG without gap filling."""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import json
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


class MyoEmgSource:
    """Own the resilient BLED112 child process and expose its native samples."""

    def __init__(
        self,
        python: str,
        tty: str,
        mac: str,
        *,
        max_queue: int = 131_072,
        silence_timeout_s: float = 0.35,
    ):
        self.python = str(Path(python).expanduser())
        self.tty = str(tty)
        self.mac = str(mac)
        self.resolved_mac = self.mac
        self.silence_timeout_s = float(silence_timeout_s)
        if self.silence_timeout_s <= 0:
            raise ValueError("Myo silence timeout must be positive")
        self.script = Path(__file__).resolve().parents[1] / "runtime" / "myo_200hz_stream.py"
        self.lock = threading.Lock()
        # timestamp, local callback sequence, movement, 8-channel EMG,
        # process-side wall-clock arrival timestamp
        self.frames: deque[tuple[int, int, int, np.ndarray, int]] = deque(maxlen=max_queue)
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
        self.driver = "unknown"
        self.received = 0
        self.dropped = 0
        self.last_seq = 0
        self.last_mono = 0.0
        self.last_timestamp_ns = 0
        self.sample_mono_window: deque[float] = deque(maxlen=4000)
        self.connection_started_mono = 0.0
        self.connection_received_start = 0
        self.interruption_id = 0
        self.interruption_reason = ""

    def _resolve_mac(self) -> str:
        if self.mac.strip().lower() != "auto":
            return self.mac
        scanner = self.script.with_name("myo_bled112_scan.py")
        try:
            result = subprocess.run(
                [
                    self.python,
                    str(scanner),
                    "--tty",
                    self.tty,
                    "--mac",
                    "auto",
                    "--seconds",
                    "12",
                    "--select-mac",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"automatic Myo scan failed: {type(exc).__name__}: {exc}") from exc
        selected = result.stdout.strip()
        if result.returncode != 0:
            detail = result.stderr.strip() or selected or f"scanner exited {result.returncode}"
            raise RuntimeError(f"automatic Myo scan failed: {detail}")
        if not re.fullmatch(r"[0-9A-F]{2}(?::[0-9A-F]{2}){5}", selected):
            raise RuntimeError(f"automatic Myo scan returned an invalid MAC: {selected!r}")
        return selected

    def start(
        self,
        timeout_s: float = 45.0,
        stability_s: float = 10.0,
        min_stable_hz: float = 180.0,
        stability_timeout_s: float = 60.0,
        wait_for_stability: bool = True,
    ) -> None:
        if not Path(self.python).is_file():
            raise RuntimeError(f"Myo Python not found: {self.python}")
        if not self.script.is_file():
            raise RuntimeError(f"Myo stream helper missing: {self.script}")
        self.resolved_mac = self._resolve_mac()
        self.process = subprocess.Popen(
            [
                self.python, "-u", str(self.script), "--tty", self.tty, "--mac", self.resolved_mac,
                "--connect-timeout-s", "12", "--silence-timeout-s", str(self.silence_timeout_s),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # The child is now running. Keep this empty until it reports a useful
        # connection error; otherwise status misleadingly says "not started"
        # throughout genuine background connection attempts.
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
        while not self.ready_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = self.error or "timeout waiting for Myo ready"
                self.close()
                raise RuntimeError(f"Myo did not become ready: {detail}; stderr={list(self.stderr_tail)[-4:]}")
            if self.process.poll() is not None:
                detail = self.error or f"Myo bridge exited with code {self.process.returncode}"
                self.close()
                raise RuntimeError(f"Myo bridge exited before ready: {detail}; stderr={list(self.stderr_tail)[-4:]}")
            self.ready_event.wait(min(0.25, remaining))
        if self.error:
            detail = self.error
            self.close()
            raise RuntimeError(f"Myo start failed: {detail}")
        if min(stability_s, min_stable_hz, stability_timeout_s) <= 0:
            self.close()
            raise ValueError("Myo stability duration and rate must be positive")
        deadline = time.monotonic() + stability_timeout_s
        stable_started: float | None = None
        received_before = 0
        stable_connection_id = 0
        last_rate = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            with self.lock:
                process_alive = self.process is not None and self.process.poll() is None
                state = self.connection_state
                connection_id = self.connection_id
                received = self.received
                age_s = now - self.last_mono if self.last_mono else float("inf")
            healthy_now = process_alive and state == "connected" and age_s <= 0.5
            if not healthy_now:
                stable_started = None
                stable_connection_id = 0
            elif stable_connection_id == 0:
                stable_connection_id = connection_id
                stable_started = now
                received_before = received
            elif connection_id != stable_connection_id:
                stable_started = None
                stable_connection_id = connection_id
                received_before = received
            elif stable_started is None:
                stable_started = now
                received_before = received
            else:
                stable_elapsed = now - stable_started
                last_rate = (received - received_before) / max(stable_elapsed, 1e-9)
                if stable_elapsed >= stability_s:
                    if last_rate >= min_stable_hz:
                        return
                    stable_started = now
                    received_before = received
            time.sleep(0.1)

        detail = self.diagnostic_tail()
        status = self.status()
        self.close()
        raise RuntimeError(
            f"Myo did not achieve a continuous {stability_s:.0f}s stable window "
            f"within {stability_timeout_s:.0f}s: last_rate={last_rate:.1f} Hz; "
            f"{status}; {detail}"
        )

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
                    self.error = ""
                    next_connection_id = int(event.get("connection_id", self.connection_id + 1))
                    if self.connection_id and next_connection_id != self.connection_id:
                        self.reconnects += 1
                        self.sample_mono_window.clear()
                    self.connection_state = "connected"
                    self.connection_id = next_connection_id
                    self.connection_started_mono = time.monotonic()
                    self.connection_received_start = self.received
                    self.resolved_tty = str(event.get("tty", self.resolved_tty))
                    self.driver = str(event.get("driver", self.driver))
                    self.events.append(f"connected#{self.connection_id} tty={self.resolved_tty}")
                self.ready_event.set()
                continue
            if kind == "state":
                state = str(event.get("state", "unknown"))
                message = str(event.get("message", ""))
                with self.lock:
                    if state in {"recovering", "disconnected", "connecting"}:
                        self.interruption_id += 1
                        self.interruption_reason = message or state
                        self.sample_mono_window.clear()
                    self.connection_state = state
                    if message:
                        self.error = message
                    self.events.append(f"{state}: {message}".strip())
                continue
            if kind == "error":
                # A configuration/import error is terminal only before the
                # first successful connection. Runtime disconnects are emitted
                # as state events and are repaired by the bridge itself.
                with self.lock:
                    if not self.ready_event.is_set():
                        self.error = str(event.get("message", "unknown Myo error"))
                self.ready_event.set()
                continue
            if kind != "sample":
                continue
            try:
                timestamp_ns = int(event["timestamp_ns"])
                arrival_timestamp_ns = int(event.get("arrival_timestamp_ns", timestamp_ns))
                seq = int(event["seq"])
                movement = int(event.get("movement", 0))
                emg = np.asarray(event["emg"], dtype=np.int8).reshape(8)
            except Exception:
                with self.lock:
                    self.dropped += 1
                continue
            with self.lock:
                now_mono = time.monotonic()
                if self.last_mono and now_mono - self.last_mono > self.silence_timeout_s:
                    self.interruption_id += 1
                    self.interruption_reason = f"Myo RAW EMG interrupted for {(now_mono - self.last_mono) * 1000:.0f} ms"
                    self.sample_mono_window.clear()
                if self.last_seq and seq > self.last_seq + 1:
                    self.dropped += seq - self.last_seq - 1
                self.last_seq = max(self.last_seq, seq)
                if len(self.frames) == self.frames.maxlen:
                    self.dropped += 1
                self.frames.append((timestamp_ns, seq, movement, emg, arrival_timestamp_ns))
                self.received += 1
                self.last_mono = now_mono
                self.last_timestamp_ns = timestamp_ns
                self.sample_mono_window.append(self.last_mono)

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            line = line.strip()
            if line:
                self.stderr_tail.append(line)

    def drain(self) -> list[tuple[int, int, int, np.ndarray, int]]:
        with self.lock:
            values = list(self.frames)
            self.frames.clear()
        return values

    def clear(self) -> None:
        self.drain()

    def age_ms(self) -> float:
        with self.lock:
            return (time.monotonic() - self.last_mono) * 1000.0 if self.last_mono else float("inf")

    def current_connection_id(self) -> int:
        with self.lock:
            return self.connection_id

    def interruption_snapshot(self) -> tuple[int, str]:
        """Retain outages even when recovery finishes between collector polls."""
        with self.lock:
            return self.interruption_id, self.interruption_reason

    def healthy(self, max_age_s: float = 0.5) -> bool:
        with self.lock:
            process_alive = self.process is not None and self.process.poll() is None
            age_s = time.monotonic() - self.last_mono if self.last_mono else float("inf")
            return process_alive and self.connection_state == "connected" and age_s <= max_age_s

    def stable(self, duration_s: float = 10.0, min_hz: float = 180.0, max_age_s: float = 0.5) -> bool:
        """Whether the current uninterrupted BLE connection has met the start gate."""
        with self.lock:
            now = time.monotonic()
            process_alive = self.process is not None and self.process.poll() is None
            age_s = now - self.last_mono if self.last_mono else float("inf")
            cutoff = now - duration_s
            recent = [sample_mono for sample_mono in self.sample_mono_window if sample_mono >= cutoff]
            span_s = recent[-1] - recent[0] if len(recent) >= 2 else 0.0
            rate_hz = (len(recent) - 1) / max(span_s, 1e-9)
            return (
                process_alive
                and self.connection_state == "connected"
                # Allow only 20 ms of packet/edge scheduling tolerance, not a
                # percentage that would shorten longer stability windows.
                and span_s >= duration_s - min(0.02, duration_s * 0.1)
                and rate_hz >= min_hz
                and all(b - a <= min(max_age_s, self.silence_timeout_s)
                        for a, b in zip(recent, recent[1:]))
                and age_s <= max_age_s
            )

    def status(self) -> str:
        with self.lock:
            now = time.monotonic()
            age_ms = (now - self.last_mono) * 1000.0 if self.last_mono else float("inf")
            recent = [stamp for stamp in self.sample_mono_window if now - stamp <= 5.0]
            if len(recent) >= 2:
                span = recent[-1] - recent[0]
                rate_hz = (len(recent) - 1) / max(span, 1e-9)
            else:
                rate_hz = 0.0
            alive = self.process is not None and self.process.poll() is None
            return (
                f"received={self.received}, queued={len(self.frames)}, dropped={self.dropped}, "
                f"age_ms={age_ms:.0f}, rate_hz={rate_hz:.1f}, state={self.connection_state}, "
                f"connection={self.connection_id}, reconnects={self.reconnects}, alive={alive}, "
                f"driver={self.driver}, tty={self.resolved_tty} (configured={self.tty}) {self.error}"
            )

    def diagnostic_tail(self) -> str:
        """Return the latest bridge event/error without consuming diagnostics."""
        with self.lock:
            event = self.events[-1] if self.events else ""
            stderr = self.stderr_tail[-1] if self.stderr_tail else ""
        details = [value for value in (event, stderr) if value]
        return " | ".join(details) if details else "no bridge diagnostics"

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
