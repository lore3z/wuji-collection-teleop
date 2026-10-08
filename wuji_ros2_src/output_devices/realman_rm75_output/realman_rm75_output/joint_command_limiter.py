"""Small bounded joint trajectory for the first real RM75 trial."""

import numpy as np


class JointCommandLimiter:
    def __init__(self, initial_joints_deg, max_offset_deg=1.0,
                 max_speed_deg_s=1.0, nominal_period_s=0.02):
        self.initial = self._joints(initial_joints_deg)
        self.current = self.initial.copy()
        self.max_offset = float(max_offset_deg)
        self.max_speed = float(max_speed_deg_s)
        self.period = float(nominal_period_s)
        if min(self.max_offset, self.max_speed, self.period) <= 0.0:
            raise ValueError("joint limiter parameters must be positive")
        self.max_step = self.max_speed * self.period

    @staticmethod
    def _joints(value):
        result = np.asarray(value, dtype=float)
        if result.shape != (7,) or not np.all(np.isfinite(result)):
            raise ValueError("joints must contain seven finite values")
        return result.copy()

    def step(self, desired_joints_deg):
        desired = self._joints(desired_joints_deg)
        bounded = np.clip(
            desired, self.initial - self.max_offset,
            self.initial + self.max_offset)
        delta = np.clip(bounded - self.current, -self.max_step, self.max_step)
        self.current += delta
        return self.current.copy()

    def step_home(self):
        return self.step(self.initial)

    def at_home(self, tolerance_deg=0.01):
        return bool(np.max(np.abs(self.current - self.initial)) <= tolerance_deg)

