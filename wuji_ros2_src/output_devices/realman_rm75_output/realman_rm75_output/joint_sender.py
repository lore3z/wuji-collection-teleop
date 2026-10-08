"""Producer-consumer joint target buffer and fixed-rate command sender."""

from dataclasses import dataclass
import threading
import time

import numpy as np


def _joints(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError("joint vector must contain seven finite values")
    return result.copy()


@dataclass(frozen=True)
class JointTargetSnapshot:
    joints: np.ndarray
    produced_at: float
    sequence: int


class LatestJointTarget:
    """Single-slot, thread-safe handoff from ~90 Hz IK to the sender."""

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = None
        self._state = "HOLD"
        self._reason = "not started"

    def publish(self, joints, produced_at=None):
        now = time.monotonic() if produced_at is None else float(produced_at)
        with self._lock:
            if self._state in ("STOP", "FAULT"):
                return False
            sequence = 1 if self._snapshot is None else self._snapshot.sequence + 1
            self._snapshot = JointTargetSnapshot(_joints(joints), now, sequence)
            self._state = "ACTIVE"
            self._reason = ""
            return True

    def snapshot(self):
        with self._lock:
            snap = self._snapshot
            state = self._state
            reason = self._reason
        if snap is not None:
            snap = JointTargetSnapshot(snap.joints.copy(), snap.produced_at,
                                       snap.sequence)
        return state, reason, snap

    def stop(self, reason="operator stop"):
        with self._lock:
            self._state = "STOP"
            self._reason = str(reason)

    def fault(self, reason):
        with self._lock:
            self._state = "FAULT"
            self._reason = str(reason)


class JointRateLimiter:
    def __init__(self, initial_joints, max_speed_deg_s=30.0,
                 max_accel_deg_s2=120.0, max_step_deg=0.5,
                 min_position_deg=None, max_position_deg=None):
        self.position = _joints(initial_joints)
        self.velocity = np.zeros(7)
        self.max_speed = float(max_speed_deg_s)
        self.max_accel = float(max_accel_deg_s2)
        self.max_step = float(max_step_deg)
        if min(self.max_speed, self.max_accel, self.max_step) <= 0.0:
            raise ValueError("sender limits must be positive")
        self.min_position = (
            np.full(7, -np.inf) if min_position_deg is None
            else _joints(min_position_deg))
        self.max_position = (
            np.full(7, np.inf) if max_position_deg is None
            else _joints(max_position_deg))
        if np.any(self.min_position > self.position) or np.any(
                self.position > self.max_position):
            raise ValueError("initial joints outside sender position guard")

    def step(self, target, dt):
        target = np.clip(_joints(target), self.min_position, self.max_position)
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("sender dt must be positive")
        error = target - self.position
        desired_velocity = np.clip(error / dt, -self.max_speed, self.max_speed)
        dv = np.clip(desired_velocity - self.velocity,
                     -self.max_accel * dt, self.max_accel * dt)
        self.velocity = np.clip(
            self.velocity + dv, -self.max_speed, self.max_speed)
        delta = np.clip(self.velocity * dt, -self.max_step, self.max_step)
        # Do not overshoot a target component.
        delta = np.where(np.abs(delta) > np.abs(error), error, delta)
        self.position = np.clip(
            self.position + delta, self.min_position, self.max_position)
        return self.position.copy()

    def hold(self):
        self.velocity[:] = 0.0
        return self.position.copy()


class FakeJointBackend:
    def __init__(self):
        self.records = []
        self._lock = threading.Lock()

    def send(self, joints, sent_at):
        with self._lock:
            self.records.append((float(sent_at), _joints(joints)))
        return 0


class JointSender:
    """Absolute-deadline sender loop, suitable for 125 Hz follow=True later."""

    def __init__(self, target_buffer, backend, initial_joints,
                 rate_hz=125.0, max_speed_deg_s=30.0,
                 max_accel_deg_s2=120.0, max_step_deg=0.5,
                 min_position_deg=None, max_position_deg=None,
                 clock=time.monotonic):
        self.target_buffer = target_buffer
        self.backend = backend
        self.rate_hz = float(rate_hz)
        self.period = 1.0 / self.rate_hz
        self.clock = clock
        self.limiter = JointRateLimiter(
            initial_joints, max_speed_deg_s, max_accel_deg_s2, max_step_deg,
            min_position_deg, max_position_deg)
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.period_samples = []
        self.target_age_samples = []
        self.deadline_misses = 0
        self.send_errors = 0
        self.send_duration_samples = []
        self.sent_count = 0

    def start(self):
        if self._thread is not None:
            raise RuntimeError("sender already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=2.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError("sender thread did not stop")

    def current_command(self):
        with self._lock:
            return self.limiter.position.copy()

    def _run(self):
        next_deadline = self.clock()
        previous_send = None
        while not self._stop_event.is_set():
            now = self.clock()
            wait = next_deadline - now
            if wait > 0.0:
                self._stop_event.wait(wait)
                if self._stop_event.is_set():
                    break
            send_at = self.clock()
            if previous_send is not None:
                actual_period = send_at - previous_send
                self.period_samples.append(actual_period)
                # Future RM high-follow requires every command interval <=10ms.
                if actual_period > 0.010:
                    self.deadline_misses += 1
            previous_send = send_at

            state, _reason, snapshot = self.target_buffer.snapshot()
            with self._lock:
                if state == "ACTIVE" and snapshot is not None:
                    command = self.limiter.step(snapshot.joints, self.period)
                    self.target_age_samples.append(
                        max(0.0, send_at - snapshot.produced_at))
                else:
                    command = self.limiter.hold()
            send_started = self.clock()
            code = self.backend.send(command, send_at)
            self.send_duration_samples.append(self.clock() - send_started)
            if code != 0:
                self.send_errors += 1
                self.target_buffer.fault(
                    f"joint backend return code {code}")
            self.sent_count += 1
            next_deadline += self.period
            # Skip stale absolute deadlines without emitting a catch-up burst.
            if send_at - next_deadline > self.period:
                skipped = int((send_at - next_deadline) / self.period) + 1
                next_deadline += skipped * self.period

    @staticmethod
    def _stats(samples):
        if not samples:
            return {key: float("nan") for key in
                    ("mean", "p95", "p99", "max")}
        values = np.asarray(samples, dtype=float) * 1000.0
        return {
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }

    def metrics(self):
        measured_periods = len(self.period_samples)
        return {
            "rate_hz": self.rate_hz,
            "sent_count": self.sent_count,
            "period_ms": self._stats(self.period_samples),
            "target_age_ms": self._stats(self.target_age_samples),
            "send_duration_ms": self._stats(self.send_duration_samples),
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": (
                self.deadline_misses / measured_periods
                if measured_periods else 0.0
            ),
            "send_errors": self.send_errors,
        }
